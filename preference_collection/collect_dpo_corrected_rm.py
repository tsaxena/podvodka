"""
collect_dpo_corrected_rm.py

Collect DPO preference data using the existing (flawed) HuggingFace reward model,
with post-hoc correction that penalizes hack words, repetition, and length bloat.

Why not just use the RM directly?
  The HF reward model (toloka/prompts_reward_model) has a learned bias toward
  "magic words" like 8k, octane render, trending on artstation, masterpiece.
  Candidates that stuff these words get high raw RM scores even when they add
  no meaningful content. corrected_reward (reward_model/corrected_reward_model.py)
  subtracts explicit penalties for:
    - hack_word_penalty: magic words added beyond what the user already wrote
    - repetition_penalty: adjacent repeats + n-gram repetition + low unique ratio
    - length_bloat_penalty: excessive expansion beyond 3× the input length
  Hard gates (p_hack > 0.35, p_rep > 0.25, p_len > 0.50) subtract an extra 1.0
  so egregious cases always land below any clean candidate.

Diversity mechanisms:
  - Near-duplicate filtering: Jaccard similarity on token sets; candidates above
    --jaccard_threshold are collapsed to their highest-scoring representative
  - Multiple pairing strategies per prompt: best vs worst, best vs 2nd-worst,
    2nd-best vs worst, etc. — up to --max_pairs_per_prompt
  - Round-robin budget: one pair is taken from each prompt in turn, so no single
    prompt monopolises the training set
  - --min_score_gap: only keep pairs whose corrected scores differ enough to give
    a clear training signal

Pipeline:
  1. Load prompts from train_strings.csv (or plain txt)
  2. Generate K candidates per prompt using the SFT model (or load cache)
  3. Score every (concept, expansion) with the HF reward model in batches
  4. Normalize raw RM scores within each prompt's candidate group → [0, 1]
  5. Apply corrected_reward (hack + repetition + length penalties)
  6. Build diverse DPO pairs per prompt; apply round-robin budget
  7. Write JSONL compatible with rl/train_dpo.py + full audit trail

Auth / setup:
  Set HF_TOKEN in your environment if the reward model is private.

Example
-------
python preference_collection/collect_dpo_corrected_rm.py \\
    --num_prompts 2000 \\
    --candidates_per_prompt 8 \\
    --output_path data/preferences_corrected_rm.jsonl

# Skip generation and scoring, only rebuild pairs from cached scores:
python preference_collection/collect_dpo_corrected_rm.py \\
    --skip_generation \\
    --skip_scoring \\
    --output_path data/preferences_corrected_rm.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------------------------------
# Import the corrected reward utilities from the sibling
# reward_model package regardless of working directory.
# -------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from reward_model.corrected_reward_model import (  # noqa: E402
    corrected_reward,
    normalize_scores_per_prompt,
    tokenize as rm_tokenize,
)

# Separator token used during reward model training (see reward_model/run.py).
SEP = "</s>"


# ============================================================
# Data structures
# ============================================================

@dataclass
class ScoredCandidate:
    prompt_id: int
    candidate_id: int
    prompt: str           # clean user concept (no SEP suffix)
    expansion: str        # SFT-generated expansion text
    raw_rm_score: float   # raw logit from the flawed RM
    norm_rm_score: float  # z-score + sigmoid normalized within prompt group
    corrected_score: float
    audit: dict           # per-component breakdown from corrected_reward


# ============================================================
# Phase 1: Candidate generation
# ============================================================

@torch.no_grad()
def generate_candidates(
    sft_model_path: str,
    prompts: List[str],
    k: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> List[List[str]]:
    from transformers import AutoModelForCausalLM  # local import — not needed if skipping

    print(f"[gen] loading SFT model from {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    out: List[List[str]] = []
    for prompt in tqdm(prompts, desc="generating candidates"):
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = [
            tokenizer.decode(g[ids.shape[1] :], skip_special_tokens=True).strip()
            for g in generated
        ]
        out.append(completions)

    del model
    torch.cuda.empty_cache()
    return out


# ============================================================
# Phase 2: RM scoring (batched, resumable)
# ============================================================

def score_with_rm(
    rm_model_path: str,
    prompts: List[str],
    candidates: List[List[str]],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    checkpoint_path: Path,
) -> List[List[float]]:
    """
    Score every (concept, expansion) pair with the flawed reward model.

    Input format matches training: "{concept}{SEP}{expansion}"
    where SEP = "</s>" (from reward_model/run.py).

    Raw logit values are written to checkpoint_path for resumability.
    Returns a 2-D list [prompt_idx][candidate_idx] → float score.
    NaN indicates a scoring failure.
    """
    # ---- Load existing checkpoint ----
    cache: Dict[Tuple[int, int], float] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cache[(d["prompt_id"], d["candidate_id"])] = d["raw_rm_score"]
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"[rm] loaded {len(cache)} cached RM scores from {checkpoint_path}")

    # ---- Collect items still needing scores ----
    todo: List[Tuple[int, int, str]] = []
    for pi, (prompt, comps) in enumerate(zip(prompts, candidates)):
        clean = prompt.rstrip(SEP).strip()
        for ci, expansion in enumerate(comps):
            if (pi, ci) not in cache:
                todo.append((pi, ci, f"{clean}{SEP}{expansion}"))

    # ---- Score remaining items ----
    if todo:
        print(f"[rm] scoring {len(todo)} items with {rm_model_path} "
              f"(batch_size={batch_size})")
        tokenizer = AutoTokenizer.from_pretrained(rm_model_path)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                rm_model_path, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "a") as f_out:
            for start in tqdm(range(0, len(todo), batch_size), desc="RM scoring"):
                batch = todo[start : start + batch_size]
                pids = [x[0] for x in batch]
                cids = [x[1] for x in batch]
                texts = [x[2] for x in batch]

                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    logits = model(**enc).logits  # (B, 1)
                scores = logits[:, 0].float().cpu().tolist()

                for pi, ci, score in zip(pids, cids, scores):
                    cache[(pi, ci)] = score
                    f_out.write(
                        json.dumps({"prompt_id": pi, "candidate_id": ci,
                                    "raw_rm_score": score}) + "\n"
                    )
                f_out.flush()

        del model
        torch.cuda.empty_cache()
    else:
        print("[rm] all scores already in cache — skipping model load")

    # ---- Reconstruct ordered 2-D list ----
    return [
        [cache.get((pi, ci), float("nan")) for ci in range(len(comps))]
        for pi, comps in enumerate(candidates)
    ]


# ============================================================
# Phase 3: Apply corrected reward
# ============================================================

def apply_corrected_reward(
    prompts: List[str],
    candidates: List[List[str]],
    raw_scores: List[List[float]],
) -> List[List[ScoredCandidate]]:
    """
    For each prompt group:
      1. Drop NaN entries (scoring failures).
      2. Normalize the remaining raw RM scores to [0, 1] via z-score + sigmoid.
      3. Apply hack / repetition / length penalties via corrected_reward.
    """
    result: List[List[ScoredCandidate]] = []

    for pi, (prompt, comps, raw) in enumerate(
        zip(prompts, candidates, raw_scores)
    ):
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
    """Token-level Jaccard similarity between two strings."""
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
    """
    Collapse near-duplicate expansions: iterate in descending corrected-score
    order so the best-scoring representative of each cluster is kept.
    Two expansions are near-duplicates if their Jaccard similarity >= threshold.
    """
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
    """
    Extract up to max_pairs DPO pairs from a single prompt's candidate group.

    Pairs are generated in best-vs-worst priority order:
      (rank-0 vs rank-N-1), (rank-0 vs rank-N-2), (rank-1 vs rank-N-1), ...
    Only pairs whose corrected score gap >= min_score_gap are kept.
    """
    deduped = deduplicate_candidates(group, jaccard_threshold)
    if len(deduped) < 2:
        return []

    by_score = sorted(deduped, key=lambda x: x.corrected_score, reverse=True)
    n = len(by_score)

    pairs: List[Dict] = []
    seen: Set[Tuple[int, int]] = set()

    # Walk outward from the extremes: (0, n-1), (0, n-2), (1, n-1), ...
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
    """
    Round-robin pair extraction across prompts.

    First, collect all available pairs for every prompt (capped at
    max_pairs_per_prompt). Then take one pair at a time from each prompt in
    turn until target_pairs is reached. This ensures diversity: even if one
    prompt has many high-quality pairs, it cannot crowd out other prompts.
    """
    pool: List[List[Dict]] = []
    for group in scored:
        if not group:
            pool.append([])
        else:
            pool.append(
                pairs_for_prompt(group, min_score_gap, max_pairs_per_prompt,
                                  jaccard_threshold)
            )

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
            break  # all prompts exhausted

    return final


# ============================================================
# Diagnostics
# ============================================================

def print_diagnostics(
    scored: List[List[ScoredCandidate]],
    pairs: List[Dict],
) -> None:
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

    # Warn if correction barely changed rankings
    pairs_rm_agrees = sum(
        1 for p in pairs
        if p["chosen_raw_rm"] > p["rejected_raw_rm"]
    )
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
        description="Collect DPO preference data using the corrected RM."
    )

    # ---- Models / data ----
    ap.add_argument(
        "--sft_model", default="tsaxena/gpt2-large-prompt-tags",
        help="HuggingFace SFT model for candidate generation.",
    )
    ap.add_argument(
        "--rm_model", default="toloka/prompts_reward_model",
        help="HuggingFace reward model (the flawed one we correct).",
    )
    ap.add_argument(
        "--train_path", default="data/train_strings.csv",
        help="CSV with a 'text' column formatted as {concept}</s>{expansion}, "
             "or plain txt with one entry per line.",
    )
    ap.add_argument("--num_prompts", type=int, default=2000)
    ap.add_argument("--candidates_per_prompt", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=80)

    # ---- Inference ----
    ap.add_argument("--rm_batch_size", type=int, default=32,
                    help="Batch size for reward model scoring.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="Use fp16 for model inference (default on).")

    # ---- Pair quality ----
    ap.add_argument(
        "--min_score_gap", type=float, default=0.15,
        help="Minimum corrected-score gap required to form a pair. "
             "Higher = cleaner signal, fewer pairs.",
    )
    ap.add_argument(
        "--max_pairs_per_prompt", type=int, default=3,
        help="Cap on pairs extracted per prompt. Limits prompt crowding.",
    )
    ap.add_argument(
        "--target_pairs", type=int, default=5000,
        help="Desired total number of DPO pairs in the output.",
    )
    ap.add_argument(
        "--jaccard_threshold", type=float, default=0.80,
        help="Jaccard similarity above which two expansions are considered "
             "near-duplicates and collapsed (keep best-scoring one).",
    )

    # ---- Paths ----
    ap.add_argument("--output_path", default="data/preferences_corrected_rm.jsonl")
    ap.add_argument(
        "--candidates_cache", default="data/sft_candidates_corrected.jsonl",
        help="Cache file for generated SFT candidates.",
    )
    ap.add_argument(
        "--rm_score_cache", default="data/rm_scores_corrected.jsonl",
        help="Checkpoint file for raw RM scores (append-only, enables resuming).",
    )

    # ---- Resumability ----
    ap.add_argument(
        "--skip_generation", action="store_true",
        help="Skip candidate generation; load from --candidates_cache.",
    )
    ap.add_argument(
        "--skip_scoring", action="store_true",
        help="Skip RM scoring entirely; use scores already in --rm_score_cache.",
    )

    args = ap.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    if not torch.cuda.is_available():
        print("[warn] CUDA not available — running on CPU (slow)")
        device = "cpu"

    # -------------------------------------------------------
    # Load prompts
    # -------------------------------------------------------
    train_path = Path(args.train_path)
    if not train_path.exists():
        sys.exit(f"[error] --train_path not found: {train_path}")

    if train_path.suffix == ".csv":
        df = pd.read_csv(train_path)
        raw_prompts = [row.split(SEP)[0].strip() for row in df["text"]]
    else:
        # Plain text: one entry per line.
        # Handles both "[BOS]concept = expansion<endoftext>" and "concept</s>..."
        with open(train_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        raw_prompts = []
        for line in lines:
            if SEP in line:
                raw_prompts.append(line.split(SEP)[0].strip())
            elif " = " in line:
                # "[BOS]concept = expansion<endoftext>" format
                concept = line.lstrip("[BOS]").split(" = ")[0].strip()
                raw_prompts.append(concept)
            else:
                raw_prompts.append(line)

    prompts = raw_prompts[: args.num_prompts]
    print(f"[data] loaded {len(prompts)} prompts from {train_path}")

    # -------------------------------------------------------
    # Phase 1: Candidate generation (or load from cache)
    # -------------------------------------------------------
    cand_cache = Path(args.candidates_cache)
    candidates: Optional[List[List[str]]] = None

    if args.skip_generation or cand_cache.exists():
        try:
            loaded: List[List[str]] = []
            with open(cand_cache) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        loaded.append(json.loads(line)["candidates"])
            if len(loaded) >= len(prompts):
                candidates = loaded[: len(prompts)]
                print(f"[gen] loaded {len(candidates)} candidate sets from {cand_cache}")
            else:
                print(
                    f"[gen] cache has {len(loaded)} entries but need {len(prompts)} "
                    "— will regenerate"
                )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"[gen] could not load cache ({e}) — will regenerate")

    if candidates is None:
        if args.skip_generation:
            sys.exit(
                f"[error] --skip_generation set but {cand_cache} is missing "
                "or incomplete."
            )
        candidates = generate_candidates(
            args.sft_model,
            prompts,
            k=args.candidates_per_prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=dtype,
        )
        cand_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cand_cache, "w") as f:
            for p_text, comps in zip(prompts, candidates):
                f.write(json.dumps({"prompt": p_text, "candidates": comps}) + "\n")
        print(f"[gen] cached {len(candidates)} candidate sets to {cand_cache}")

    # -------------------------------------------------------
    # Phase 2: RM scoring (batched, checkpointed)
    # -------------------------------------------------------
    rm_cache = Path(args.rm_score_cache)

    if args.skip_scoring and not rm_cache.exists():
        sys.exit(
            f"[error] --skip_scoring set but {rm_cache} does not exist."
        )

    raw_scores = score_with_rm(
        rm_model_path=args.rm_model,
        prompts=prompts,
        candidates=candidates,
        batch_size=args.rm_batch_size,
        device=device,
        dtype=dtype,
        checkpoint_path=rm_cache,
    )

    # -------------------------------------------------------
    # Phase 3: Apply corrected reward
    # -------------------------------------------------------
    print("[reward] applying hack / repetition / length penalties ...")
    scored = apply_corrected_reward(prompts, candidates, raw_scores)

    n_groups_with_cands = sum(1 for g in scored if g)
    print(f"[reward] {n_groups_with_cands} / {len(prompts)} prompts have ≥1 scored candidate")

    # -------------------------------------------------------
    # Phase 4: Build diverse DPO pairs
    # -------------------------------------------------------
    print("[pairs] building diverse DPO pairs ...")
    pairs = build_diverse_pairs(
        scored,
        min_score_gap=args.min_score_gap,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        target_pairs=args.target_pairs,
        jaccard_threshold=args.jaccard_threshold,
    )

    # -------------------------------------------------------
    # Write output
    # -------------------------------------------------------
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"[out] wrote {len(pairs)} pairs to {out_path}")

    print_diagnostics(scored, pairs)

    print(f"\nNext step:")
    print(f"  python rl/train_dpo.py --dataset_path {out_path}")


if __name__ == "__main__":
    main()
