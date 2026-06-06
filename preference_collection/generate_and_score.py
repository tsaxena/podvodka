"""
generate_and_score.py

Phase 1 + 2 of the corrected-RM DPO pipeline:
  1. Generate K candidate expansions per prompt using the SFT model
     (or load from --candidates_cache to skip).
  2. Score every (concept, expansion) pair with the HF reward model in batches,
     writing scores to a resumable checkpoint (--rm_score_cache).

Outputs
-------
  --candidates_cache   JSONL, one line per prompt:
                         {"prompt": "...", "candidates": ["...", ...]}
  --rm_score_cache     JSONL, one line per (prompt, candidate):
                         {"prompt_id": i, "candidate_id": j, "raw_rm_score": f}

Example
-------
  # Full run (generate + score):
  python preference_collection/generate_and_score.py \\
      --num_prompts 20000 \\
      --candidates_per_prompt 5

  # Resume scoring only (candidates already cached):
  python preference_collection/generate_and_score.py \\
      --skip_generation \\
      --num_prompts 2000

  # Generate candidates only, skip RM scoring:
  python preference_collection/generate_and_score.py \\
      --skip_scoring \\
      --num_prompts 2000 \\
      --candidates_per_prompt 8

Run next
--------
  python preference_collection/build_pairs.py \\
      --candidates_cache data/sft_candidates_corrected.jsonl \\
      --rm_score_cache   data/rm_scores_corrected.jsonl \\
      --output_path      data/preferences_corrected_rm.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEP = "</s>"


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
    from transformers import AutoModelForCausalLM

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
            tokenizer.decode(g[ids.shape[1]:], skip_special_tokens=True).strip()
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
) -> None:
    """
    Score every (concept, expansion) pair with the flawed reward model and
    append results to checkpoint_path.  Already-scored pairs are skipped.
    """
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

    todo: List[Tuple[int, int, str]] = []
    for pi, (prompt, comps) in enumerate(zip(prompts, candidates)):
        clean = prompt.rstrip(SEP).strip()
        for ci, expansion in enumerate(comps):
            if (pi, ci) not in cache:
                todo.append((pi, ci, f"{clean}{SEP}{expansion}"))

    if not todo:
        print("[rm] all scores already in cache — skipping model load")
        return

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
            batch = todo[start: start + batch_size]
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
                logits = model(**enc).logits
            scores = logits[:, 0].float().cpu().tolist()

            for pi, ci, score in zip(pids, cids, scores):
                f_out.write(
                    json.dumps({"prompt_id": pi, "candidate_id": ci,
                                "raw_rm_score": score}) + "\n"
                )
            f_out.flush()

    del model
    torch.cuda.empty_cache()
    print(f"[rm] scores written to {checkpoint_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate SFT candidates and score them with the reward model."
    )

    ap.add_argument("--sft_model", default="tsaxena/gpt2-large-prompt-tags")
    ap.add_argument("--rm_model", default="toloka/prompts_reward_model")
    ap.add_argument("--train_path", default="data/train_strings.csv")
    ap.add_argument("--num_prompts", type=int, default=2000)
    ap.add_argument("--candidates_per_prompt", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--rm_batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--candidates_cache", default="data/sft_candidates_corrected.jsonl")
    ap.add_argument("--rm_score_cache", default="data/rm_scores_corrected.jsonl")
    ap.add_argument(
        "--skip_generation", action="store_true",
        help="Load candidates from --candidates_cache instead of generating.",
    )
    ap.add_argument(
        "--skip_scoring", action="store_true",
        help="Skip RM scoring (useful if only regenerating candidates).",
    )

    args = ap.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    device = args.device
    if not torch.cuda.is_available():
        print("[warn] CUDA not available — running on CPU (slow)")
        device = "cpu"

    # ---- Load prompts ----
    train_path = Path(args.train_path)
    if not train_path.exists():
        sys.exit(f"[error] --train_path not found: {train_path}")

    if train_path.suffix == ".csv":
        df = pd.read_csv(train_path)
        raw_prompts = [row.split(SEP)[0].strip() for row in df["text"]]
    else:
        with open(train_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        raw_prompts = []
        for line in lines:
            if SEP in line:
                raw_prompts.append(line.split(SEP)[0].strip())
            elif " = " in line:
                raw_prompts.append(line.lstrip("[BOS]").split(" = ")[0].strip())
            else:
                raw_prompts.append(line)

    prompts = raw_prompts[: args.num_prompts]
    print(f"[data] loaded {len(prompts)} prompts from {train_path}")

    # ---- Phase 1: candidates ----
    cand_cache = Path(args.candidates_cache)
    candidates: List[List[str]] = []

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

    if not candidates:
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

    # ---- Phase 2: RM scoring ----
    if args.skip_scoring:
        print("[rm] --skip_scoring set — skipping reward model scoring")
    else:
        score_with_rm(
            rm_model_path=args.rm_model,
            prompts=prompts,
            candidates=candidates,
            batch_size=args.rm_batch_size,
            device=device,
            dtype=dtype,
            checkpoint_path=Path(args.rm_score_cache),
        )

    print(
        f"\nNext step:\n"
        f"  python preference_collection/build_pairs.py"
        f" --candidates_cache {cand_cache}"
        f" --rm_score_cache {args.rm_score_cache}"
    )


if __name__ == "__main__":
    main()
