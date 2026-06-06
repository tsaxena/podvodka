"""
build_pairs.py

Phase 3 + 4 of the corrected-RM DPO pipeline:
  3. Load candidates and raw RM scores from the caches written by
     generate_and_score.py, then apply corrected_reward (hack / repetition /
     length penalties).
  4. Build diverse DPO pairs with Jaccard deduplication and round-robin budget,
     then write DPO-ready JSONL.

Reads
-----
  --candidates_cache   JSONL written by generate_and_score.py
  --rm_score_cache     JSONL written by generate_and_score.py

Run after
---------
  python preference_collection/generate_and_score.py \\
      --num_prompts 2000 \\
      --candidates_per_prompt 8

Example
-------
  # Default run (up to 5000 pairs):
  python preference_collection/build_pairs.py \\
      --candidates_cache data/sft_candidates_corrected.jsonl \\
      --rm_score_cache   data/rm_scores_corrected.jsonl \\
      --output_path      data/preferences_corrected_rm.jsonl

  # Large run — 20k prompts × 5 candidates, up to 50000 pairs:
  python preference_collection/build_pairs.py \\
      --candidates_cache data/sft_candidates_corrected.jsonl \\
      --rm_score_cache   data/rm_scores_corrected.jsonl \\
      --num_prompts 20000 \\
      --target_pairs 50000 \\
      --max_pairs_per_prompt 5 \\
      --min_score_gap 0.10 \\
      --output_path      data/preferences_corrected_rm.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
from reward_model.corrected_reward_model import (  # noqa: E402
    corrected_reward,
    normalize_scores_per_prompt,
    tokenize as rm_tokenize,
)

SEP = "</s>"


# ============================================================
# Data structures
# ============================================================

@dataclass
class ScoredCandidate:
    prompt_id: int
    candidate_id: int
    prompt: str
    expansion: str
    raw_rm_score: float
    norm_rm_score: float
    corrected_score: float
    audit: dict


# ============================================================
# Load caches
# ============================================================

def load_candidates(cache_path: Path, num_prompts: Optional[int]) -> Tuple[List[str], List[List[str]]]:
    """Return (prompts, candidates) loaded from the candidates JSONL cache."""
    prompts: List[str] = []
    candidates: List[List[str]] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            candidates.append(d["candidates"])
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
        candidates = candidates[:num_prompts]
    return prompts, candidates


def load_raw_scores(
    cache_path: Path,
    num_prompts: int,
    candidates_per_prompt: int,
) -> List[List[float]]:
    """
    Reconstruct a 2-D list [prompt_idx][candidate_idx] → raw RM score.
    Missing entries are filled with NaN.
    """
    cache: Dict[Tuple[int, int], float] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                cache[(d["prompt_id"], d["candidate_id"])] = d["raw_rm_score"]
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"[rm] loaded {len(cache)} raw RM scores from {cache_path}")
    return [
        [cache.get((pi, ci), float("nan")) for ci in range(candidates_per_prompt)]
        for pi in range(num_prompts)
    ]


# ============================================================
# Phase 3: Apply corrected reward
# ============================================================

def apply_corrected_reward(
    prompts: List[str],
    candidates: List[List[str]],
    raw_scores: List[List[float]],
) -> List[List[ScoredCandidate]]:
    result: List[List[ScoredCandidate]] = []

    for pi, (prompt, comps, raw) in enumerate(zip(prompts, candidates, raw_scores)):
        clean_prompt = prompt.rstrip(SEP).strip()

        valid_idx = [i for i, s in enumerate(raw) if not math.isnan(s)]
        if not valid_idx:
            result.append([])
            continue

        valid_raw = [raw[i] for i in valid_idx]
        norm_scores = normalize_scores_per_prompt(valid_raw)

        group: List[ScoredCandidate] = []
        for ci, norm_s in zip(valid_idx, norm_scores):
            expansion = comps[ci]
            corr, audit = corrected_reward(
                user_prompt=clean_prompt,
                candidate=expansion,
                poor_reward_norm=norm_s,
            )
            group.append(
                ScoredCandidate(
                    prompt_id=pi,
                    candidate_id=ci,
                    prompt=clean_prompt,
                    expansion=expansion,
                    raw_rm_score=raw[ci],
                    norm_rm_score=norm_s,
                    corrected_score=corr,
                    audit=audit,
                )
            )
        result.append(group)

    return result


# ============================================================
# Diversity helpers
# ============================================================

def jaccard_similarity(a: str, b: str) -> float:
    ta: Set[str] = set(rm_tokenize(a))
    tb: Set[str] = set(rm_tokenize(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def deduplicate_candidates(
    group: List[ScoredCandidate],
    jaccard_threshold: float,
) -> List[ScoredCandidate]:
    sorted_group = sorted(group, key=lambda x: x.corrected_score, reverse=True)
    kept: List[ScoredCandidate] = []
    for cand in sorted_group:
        is_dup = any(
            jaccard_similarity(cand.expansion, k.expansion) >= jaccard_threshold
            for k in kept
        )
        if not is_dup:
            kept.append(cand)
    return kept


# ============================================================
# Phase 4: Build diverse DPO pairs
# ============================================================

def pairs_for_prompt(
    group: List[ScoredCandidate],
    min_score_gap: float,
    max_pairs: int,
    jaccard_threshold: float,
) -> List[Dict]:
    deduped = deduplicate_candidates(group, jaccard_threshold)
    if len(deduped) < 2:
        return []

    by_score = sorted(deduped, key=lambda x: x.corrected_score, reverse=True)
    n = len(by_score)

    pairs: List[Dict] = []
    seen: Set[Tuple[int, int]] = set()

    for lo in range(n - 1, 0, -1):
        for hi in range(0, lo):
            if len(pairs) >= max_pairs:
                break
            chosen = by_score[hi]
            rejected = by_score[lo]
            gap = chosen.corrected_score - rejected.corrected_score
            if gap < min_score_gap:
                continue
            key = (chosen.candidate_id, rejected.candidate_id)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "prompt": chosen.prompt,
                "chosen": chosen.expansion,
                "rejected": rejected.expansion,
                "chosen_score": chosen.corrected_score,
                "rejected_score": rejected.corrected_score,
                "score_gap": gap,
                "chosen_raw_rm": chosen.raw_rm_score,
                "rejected_raw_rm": rejected.raw_rm_score,
                "chosen_audit": chosen.audit,
                "rejected_audit": rejected.audit,
                "prompt_id": chosen.prompt_id,
            })
        if len(pairs) >= max_pairs:
            break

    return pairs


def build_diverse_pairs(
    scored: List[List[ScoredCandidate]],
    min_score_gap: float,
    max_pairs_per_prompt: int,
    target_pairs: int,
    jaccard_threshold: float,
) -> List[Dict]:
    pool: List[List[Dict]] = [
        pairs_for_prompt(group, min_score_gap, max_pairs_per_prompt, jaccard_threshold)
        if group else []
        for group in scored
    ]

    final: List[Dict] = []
    cursors = [0] * len(pool)

    while len(final) < target_pairs:
        added = 0
        for pi in range(len(pool)):
            if len(final) >= target_pairs:
                break
            c = cursors[pi]
            if c < len(pool[pi]):
                final.append(pool[pi][c])
                cursors[pi] += 1
                added += 1
        if added == 0:
            break

    return final


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(scored: List[List[ScoredCandidate]], pairs: List[Dict]) -> None:
    all_cands = [c for group in scored for c in group]
    if not all_cands:
        print("[diag] no candidates were scored")
        return

    n = len(all_cands)
    hack_vals = [c.audit["hack_word_penalty"] for c in all_cands]
    rep_vals = [c.audit["repetition_penalty"] for c in all_cands]
    len_vals = [c.audit["length_bloat_penalty"] for c in all_cands]
    hard_fails = sum(1 for c in all_cands if c.audit["hard_fail"] > 0.5)
    raw_rm = [c.raw_rm_score for c in all_cands]
    corr = [c.corrected_score for c in all_cands]

    def _fmt(vals: List[float]) -> str:
        m = sum(vals) / len(vals)
        return f"mean={m:.3f}, min={min(vals):.3f}, max={max(vals):.3f}"

    print("\n[diag] ===== Scoring summary =====")
    print(f"  Total candidates scored : {n}")
    print(f"  Hard-gate failures      : {hard_fails}  ({100 * hard_fails / n:.1f}%)")
    print(f"  Hack penalty            : {_fmt(hack_vals)}")
    print(f"  Repetition penalty      : {_fmt(rep_vals)}")
    print(f"  Length bloat penalty    : {_fmt(len_vals)}")
    print(f"  Raw RM score            : {_fmt(raw_rm)}")
    print(f"  Corrected score         : {_fmt(corr)}")

    print(f"\n[diag] ===== Pair summary =====")
    print(f"  DPO pairs built         : {len(pairs)}")
    if not pairs:
        print("  (no pairs — try lowering --min_score_gap)")
        return

    gaps = [p["score_gap"] for p in pairs]
    print(f"  Score gap               : {_fmt(gaps)}")

    prompts_used = len({p["prompt_id"] for p in pairs})
    print(f"  Unique prompts used     : {prompts_used}")

    per_prompt: Dict[int, int] = defaultdict(int)
    for p in pairs:
        per_prompt[p["prompt_id"]] += 1
    counts = list(per_prompt.values())
    print(
        f"  Pairs/prompt            : mean={sum(counts)/len(counts):.1f}, "
        f"min={min(counts)}, max={max(counts)}"
    )

    pairs_rm_agrees = sum(1 for p in pairs if p["chosen_raw_rm"] > p["rejected_raw_rm"])
    rm_agree_pct = 100 * pairs_rm_agrees / len(pairs)
    print(f"\n  Raw RM agrees with corrected ranking: {rm_agree_pct:.1f}% of pairs")
    if rm_agree_pct < 50:
        print("  [warn] Correction frequently overrides the RM — hack / repetition "
              "penalties are dominating. Consider reviewing --min_score_gap.")
    elif rm_agree_pct > 95:
        print("  [warn] Correction barely changes rankings. The RM may not be heavily "
              "hacked in this candidate set, or penalties are too mild.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply corrected reward and build DPO pairs from cached scores."
    )

    ap.add_argument("--candidates_cache", default="data/sft_candidates_corrected.jsonl",
                    help="JSONL cache written by generate_and_score.py.")
    ap.add_argument("--rm_score_cache", default="data/rm_scores_corrected.jsonl",
                    help="JSONL score checkpoint written by generate_and_score.py.")
    ap.add_argument("--num_prompts", type=int, default=None,
                    help="Limit to first N prompts from the cache (default: all).")
    ap.add_argument("--min_score_gap", type=float, default=0.15)
    ap.add_argument("--max_pairs_per_prompt", type=int, default=3)
    ap.add_argument("--target_pairs", type=int, default=5000)
    ap.add_argument("--jaccard_threshold", type=float, default=0.80)
    ap.add_argument("--output_path", default="data/preferences_corrected_rm.jsonl")

    args = ap.parse_args()

    cand_cache = Path(args.candidates_cache)
    rm_cache = Path(args.rm_score_cache)

    for p in (cand_cache, rm_cache):
        if not p.exists():
            sys.exit(f"[error] required cache not found: {p}\n"
                     "Run generate_and_score.py first.")

    # ---- Load caches ----
    prompts, candidates = load_candidates(cand_cache, args.num_prompts)
    print(f"[data] loaded {len(prompts)} prompts from {cand_cache}")

    candidates_per_prompt = len(candidates[0]) if candidates else 0
    raw_scores = load_raw_scores(rm_cache, len(prompts), candidates_per_prompt)

    # ---- Phase 3: corrected reward ----
    print("[reward] applying hack / repetition / length penalties ...")
    scored = apply_corrected_reward(prompts, candidates, raw_scores)
    n_with_cands = sum(1 for g in scored if g)
    print(f"[reward] {n_with_cands} / {len(prompts)} prompts have ≥1 scored candidate")

    # ---- Phase 4: build pairs ----
    print("[pairs] building diverse DPO pairs ...")
    pairs = build_diverse_pairs(
        scored,
        min_score_gap=args.min_score_gap,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        target_pairs=args.target_pairs,
        jaccard_threshold=args.jaccard_threshold,
    )

    # ---- Write output ----
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"[out] wrote {len(pairs)} pairs to {out_path}")

    print_diagnostics(scored, pairs)

    print(f"\nNext step:\n  python rl/train_dpo.py --dataset_path {out_path}")


if __name__ == "__main__":
    main()
