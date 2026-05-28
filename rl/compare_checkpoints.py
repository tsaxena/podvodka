"""
Compare multiple PPO checkpoints (and the base SFT model) on the same prompts.

For each checkpoint:
- Generate K completions per prompt (K samples to estimate variance)
- Score every generation with the reward model
- Report mean/std reward + a side-by-side CSV of generations

Use this to detect reward hacking: if `final` scores far above `mid` but
the actual text quality is worse, that's the signature. Always read the CSV
afterwards — numbers alone won't catch RM gaming.

Example
-------
python compare_checkpoints.py \\
    --checkpoints \\
        base:tsaxena/gpt2-large-prompt-tags \\
        early:/workspace/podvodka/models/gpt2-large-rl-prompt-writing/step-000100 \\
        mid:/workspace/podvodka/models/gpt2-large-rl-prompt-writing/step-000200 \\
        best:/workspace/podvodka/models/gpt2-large-rl-prompt-writing/best \\
    --val_path /workspace/podvodka/data/val_strings.csv \\
    --num_prompts 30 \\
    --samples_per_prompt 2 \\
    --output_csv /workspace/podvodka/checkpoint_comparison.csv
"""

import argparse
import gc
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_checkpoints(items: List[str]) -> Dict[str, str]:
    """Parse 'name:path' pairs from CLI into a dict, preserving order."""
    out = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"--checkpoints entry must be 'name:path', got {item!r}")
        name, path = item.split(":", 1)
        out[name.strip()] = path.strip()
    return out


@torch.no_grad()
def generate_for_checkpoint(
    path: str,
    prompts: List[str],
    samples_per_prompt: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
) -> List[List[str]]:
    """Generate K samples per prompt for the model at `path`. Returns list-of-lists."""
    print(f"  [load] {path}")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    all_samples: List[List[str]] = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            num_return_sequences=samples_per_prompt,
            pad_token_id=tok.pad_token_id,
        )
        decoded = [
            tok.decode(o[ids.shape[1]:], skip_special_tokens=False)
            for o in out
        ]
        all_samples.append(decoded)
        if (i + 1) % 10 == 0:
            print(f"    generated {i+1}/{len(prompts)}")

    # Free GPU memory before loading next checkpoint
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    return all_samples


@torch.no_grad()
def score_all(
    reward_pipeline,
    prompts: List[str],
    samples_by_ckpt: Dict[str, List[List[str]]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Score every (prompt, sample) under every checkpoint with the reward model.
    Returns a dict mapping checkpoint name -> array of shape (num_prompts, samples_per_prompt).
    """
    scores_by_ckpt = {}
    for name, samples in samples_by_ckpt.items():
        print(f"  [score] {name}")
        # Flatten (prompt, sample) into a single list for batched scoring
        flat_texts = [
            p + "</s>" + s
            for p, sample_list in zip(prompts, samples)
            for s in sample_list
        ]
        outs = reward_pipeline(
            flat_texts,
            function_to_apply="none",
            batch_size=batch_size,
            truncation=True,
        )
        flat_scores = np.array([o["score"] for o in outs], dtype=np.float32)
        n_prompts = len(samples)
        k = len(samples[0])
        scores_by_ckpt[name] = flat_scores.reshape(n_prompts, k)
    return scores_by_ckpt


def print_summary(scores_by_ckpt: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Print a summary table and return it as a DataFrame."""
    rows = []
    for name, scores in scores_by_ckpt.items():
        per_prompt_mean = scores.mean(axis=1)  # mean across K samples per prompt
        rows.append({
            "checkpoint": name,
            "rm_mean": float(scores.mean()),
            "rm_std_overall": float(scores.std()),
            "rm_std_across_prompts": float(per_prompt_mean.std()),
            "rm_min": float(scores.min()),
            "rm_max": float(scores.max()),
            "frac_above_+0.5": float((scores > 0.5).mean()),
            "frac_at_ceiling_+0.9": float((scores > 0.9).mean()),
        })
    df = pd.DataFrame(rows).set_index("checkpoint")
    print("\n" + "=" * 80)
    print("REWARD MODEL SCORES BY CHECKPOINT")
    print("=" * 80)
    print(df.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("=" * 80)
    return df


def detect_hacking_flags(df: pd.DataFrame) -> None:
    """Heuristic warnings — not definitive, just signals worth investigating."""
    names = df.index.tolist()
    if len(names) < 3:
        return
    print("\nHEURISTIC FLAGS (not definitive — read the generations!):")

    means = df["rm_mean"].values
    # Check for accelerating reward gain at the end (classic hacking signature)
    if len(means) >= 3:
        last_gap = means[-1] - means[-2]
        prev_gap = means[-2] - means[-3]
        if last_gap > 1.5 * prev_gap and last_gap > 0.1:
            print(f"  ⚠ Accelerating reward at the end: "
                  f"Δ({names[-2]}→{names[-1]})={last_gap:+.2f} vs "
                  f"Δ({names[-3]}→{names[-2]})={prev_gap:+.2f}")

    # Std compression (mode-collapse-y)
    stds = df["rm_std_overall"].values
    if stds[-1] < 0.5 * stds[0]:
        print(f"  ⚠ Std collapsed by >50%: {stds[0]:.3f} → {stds[-1]:.3f}")

    # Ceiling saturation
    ceiling = df["frac_at_ceiling_+0.9"].values
    if ceiling[-1] > 0.3 and ceiling[0] < 0.05:
        print(f"  ⚠ {ceiling[-1]:.0%} of {names[-1]} generations score >+0.9 "
              f"(vs {ceiling[0]:.0%} for {names[0]}) — possible RM saturation exploit")

    if all(stds[-1] >= 0.5 * stds[0] for s in [None]) and ceiling[-1] <= 0.3:
        print("  (no strong flags)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more 'name:path' entries, in order from earliest to latest.")
    p.add_argument("--val_path", required=True,
                   help="CSV with a 'text' column containing prompts (with </s> markers).")
    p.add_argument("--reward_model_path", default="toloka/prompts_reward_model")
    p.add_argument("--num_prompts", type=int, default=30,
                   help="Number of distinct prompts to evaluate on.")
    p.add_argument("--samples_per_prompt", type=int, default=2,
                   help="K — sample K completions per prompt per checkpoint.")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--reward_batch_size", type=int, default=32)
    p.add_argument("--output_csv", default="checkpoint_comparison.csv",
                   help="Path for the side-by-side generations CSV.")
    p.add_argument("--summary_csv", default=None,
                   help="Optional path for the numeric summary CSV.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Need a GPU for this comparison."
    device = "cuda:0"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoints = parse_checkpoints(args.checkpoints)
    print(f"Comparing {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")

    # Load prompts (same set for every checkpoint — controls for prompt difficulty)
    val_df = pd.read_csv(args.val_path)
    prompts = [l.split("</s>")[0] + "</s>" for l in val_df["text"]][: args.num_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.val_path}")

    # Phase 1: generate from every checkpoint, one at a time (memory-safe)
    print("\n--- Phase 1: generation ---")
    samples_by_ckpt: Dict[str, List[List[str]]] = {}
    for name, path in checkpoints.items():
        print(f"[ckpt] {name}")
        # Re-seed before each so prompts use the same RNG sequence per checkpoint
        torch.manual_seed(args.seed)
        samples_by_ckpt[name] = generate_for_checkpoint(
            path=path,
            prompts=prompts,
            samples_per_prompt=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=dtype,
        )

    # Phase 2: score everything with the reward model
    print("\n--- Phase 2: reward scoring ---")
    reward_pipeline = pipeline(
        "text-classification",
        model=args.reward_model_path,
        device=0,
        torch_dtype=dtype,
    )
    scores_by_ckpt = score_all(
        reward_pipeline,
        prompts,
        samples_by_ckpt,
        batch_size=args.reward_batch_size,
    )

    # Phase 3: summary + flags
    print("\n--- Phase 3: summary ---")
    summary_df = print_summary(scores_by_ckpt)
    detect_hacking_flags(summary_df)
    if args.summary_csv:
        summary_df.to_csv(args.summary_csv)
        print(f"\nWrote summary to {args.summary_csv}")

    # Phase 4: side-by-side generations CSV (this is the thing you actually read)
    print("\n--- Phase 4: writing side-by-side generations ---")
    rows = []
    for i, prompt in enumerate(prompts):
        for k in range(args.samples_per_prompt):
            row = {"prompt": prompt, "sample_idx": k}
            for name in checkpoints:
                row[f"{name}__gen"] = samples_by_ckpt[name][i][k]
                row[f"{name}__rm_score"] = float(scores_by_ckpt[name][i, k])
            rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output_csv}")

    print("\nNext step: open the CSV and READ a sample of rows. "
          "Numbers can't catch RM gaming — your eyes can.")


if __name__ == "__main__":
    main()
