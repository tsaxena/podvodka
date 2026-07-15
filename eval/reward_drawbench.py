"""
reward_drawbench.py — Run the reward model on all 200 DrawBench prompts.

Loads the DrawBench benchmark (shunk031/DrawBench, 11 categories, 200 prompts),
optionally generates SD-tag completions from a fine-tuned checkpoint, then scores
each (prompt, completion) pair with the toloka/prompts_reward_model.

This gives a reward-model signal on a fixed, standard benchmark — the same 200
prompts every run — so reward differences across checkpoints reflect real training
progress rather than prompt-difficulty variance.

Usage:
    # Baseline: score raw DrawBench prompts (no enrichment)
    python eval/reward_drawbench.py

    # Score completions from a fine-tuned checkpoint:
    python eval/reward_drawbench.py \\
        --checkpoint-path /path/to/checkpoint \\
        --step-label v7_step_7177

    # Override defaults:
    python eval/reward_drawbench.py \\
        --checkpoint-path /path/to/checkpoint \\
        --num-samples 8 \\
        --reward-model toloka/prompts_reward_model \\
        --output my_results.csv

Results are appended to --output across runs so evaluating multiple checkpoints
over time builds a single directly-comparable table (same prompts, same reward
model, only the checkpoint changes).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline


# ---------------------------------------------------------------------------
# DrawBench loader (mirrors dsg.py's _load_drawbench_prompts)
# ---------------------------------------------------------------------------

def load_drawbench_prompts() -> list[str]:
    """Load all 200 DrawBench benchmark prompts from shunk031/DrawBench."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "Error: the `datasets` package is required.\n"
            "Install with: pip install datasets"
        )
    try:
        ds = load_dataset("shunk031/DrawBench", split="test")
    except Exception:
        try:
            ds = load_dataset("shunk031/DrawBench", split="test", trust_remote_code=True)
        except RuntimeError as e:
            sys.exit(
                f"Error loading DrawBench: {e}\n"
                "The installed 'datasets' version no longer supports dataset scripts.\n"
                "Fix: pip install 'datasets<3.0'"
            )
    return [row["prompts"] for row in ds]


# ---------------------------------------------------------------------------
# Completion generation (mirrors eval_checkpoint.py)
# ---------------------------------------------------------------------------

def generate_completions(
    checkpoint_path: str,
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> list[dict]:
    """Generate SD-tag completions for each prompt using a fine-tuned checkpoint.

    Each prompt produces `num_samples` independent completions.
    Returns a flat list of {'prompt': str, 'completion': str} dicts.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit(
            "Error: transformers and torch are required for generation.\n"
            "Install with: pip install transformers torch"
        )

    print(f"Loading checkpoint: {checkpoint_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct left-padded causal generation

    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, torch_dtype=model_dtype
    ).to(device)
    model.eval()

    rows = []
    for i, prompt in enumerate(prompts):
        # Format mirrors the training convention: description + </s> as the input prefix.
        prompt_text = prompt + "</s>"
        inputs = tokenizer(
            [prompt_text] * num_samples, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for out in output_ids:
            completion = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            rows.append({"prompt": prompt, "completion": completion})

        if (i + 1) % 20 == 0 or (i + 1) == len(prompts):
            print(f"  Generated completions for {i + 1}/{len(prompts)} prompts")

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score all 200 DrawBench prompts with the reward model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path", default=None,
        help="Path or HuggingFace repo ID for a fine-tuned causal-LM checkpoint. "
             "When given, the model generates SD-tag completions that are appended "
             "to each DrawBench prompt before reward scoring. "
             "When omitted, the raw DrawBench prompts are scored as the SD prompt "
             "(no-enrichment baseline).",
    )
    parser.add_argument(
        "--reward-model", default="toloka/prompts_reward_model",
        help="HuggingFace model ID or local path for the reward model "
             "(default: toloka/prompts_reward_model).",
    )
    parser.add_argument(
        "--step-label", default=None,
        help="Label recorded in the results CSV (e.g. the training step this checkpoint "
             "corresponds to). Defaults to the checkpoint dir/repo name, or 'baseline' "
             "when --checkpoint-path is not given.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=4,
        help="Number of completions sampled per prompt when --checkpoint-path is given "
             "(default: 4). Higher values give a more stable per-prompt estimate.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Max new tokens when generating completions (default: 80).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for completion generation (default: 1.0).",
    )
    parser.add_argument(
        "--gen-device", default="cuda",
        help="Device for the generation checkpoint (default: cuda; falls back to cpu if "
             "CUDA is unavailable).",
    )
    parser.add_argument(
        "--reward-device", default="cuda",
        help="Device for the reward model (default: cuda; use 'cpu' if no GPU).",
    )
    parser.add_argument(
        "--reward-batch-size", type=int, default=32,
        help="Batch size for reward model inference (default: 32).",
    )
    parser.add_argument(
        "--output", default="reward_drawbench_results.csv",
        help="CSV file that results are appended to across runs, building a "
             "cross-checkpoint comparison table (default: reward_drawbench_results.csv).",
    )
    args = parser.parse_args()

    step_label = args.step_label
    if step_label is None:
        step_label = Path(args.checkpoint_path).name if args.checkpoint_path else "baseline"

    print(f"[DrawBench] step_label = {step_label!r}")

    prompts = load_drawbench_prompts()
    print(f"[DrawBench] Loaded {len(prompts)} prompts")

    # ---- Generate or prepare completions ----
    if args.checkpoint_path:
        gen_device = args.gen_device if torch.cuda.is_available() else "cpu"
        if gen_device != args.gen_device:
            print(f"  (CUDA not available — using cpu for generation)")
        rows = generate_completions(
            args.checkpoint_path,
            prompts,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=gen_device,
        )
    else:
        # Baseline: no enrichment — use the raw DrawBench prompt as the SD prompt.
        # One row per prompt, empty completion.
        print("No checkpoint provided — scoring raw DrawBench prompts (baseline).")
        rows = [{"prompt": p, "completion": ""} for p in prompts]

    # ---- Load reward model ----
    use_gpu = args.reward_device.startswith("cuda") and torch.cuda.is_available()
    reward_device = 0 if use_gpu else -1
    if not use_gpu and args.reward_device.startswith("cuda"):
        print("  (CUDA not available — using cpu for reward model)")
    print(f"Loading reward model: {args.reward_model} ...")
    reward_pipeline = pipeline(
        "text-classification",
        model=args.reward_model,
        device=reward_device,
    )

    # ---- Score ----
    # Format mirrors the GRPO training convention (run_grpo.py):
    # prompt already ends in </s> → reward input is prompt</s></s>completion.
    print(f"Scoring {len(rows)} (prompt, completion) pairs ...")
    reward_texts = [r["prompt"] + "</s>" + r["completion"] for r in rows]
    outputs = reward_pipeline(
        reward_texts,
        function_to_apply="none",
        batch_size=args.reward_batch_size,
        truncation=True,
    )

    # ---- Collect results ----
    all_rows = []
    for row, out in zip(rows, outputs):
        all_rows.append({
            "step_label": step_label,
            "checkpoint_path": args.checkpoint_path or "",
            "prompt": row["prompt"],
            "completion": row["completion"],
            "reward": out["score"],
        })

    df = pd.DataFrame(all_rows)

    # ---- Print per-prompt summary ----
    summary = df.groupby("prompt")["reward"].mean().sort_values()
    print(f"\n{'Prompt':<50} {'Mean Reward':>12}")
    print("-" * 63)
    for prompt, mean_reward in summary.items():
        print(f"{prompt[:48]:<50} {mean_reward:>+12.3f}")

    overall_mean = df["reward"].mean()
    print(
        f"\n[{step_label}] Overall mean reward: {overall_mean:+.4f} "
        f"(n={len(df)}, {len(prompts)} prompts"
        + (f", {args.num_samples} samples/prompt)" if args.checkpoint_path else ")")
    )

    # ---- Append to CSV ----
    results_path = Path(args.output)
    write_header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=write_header, index=False)
    print(f"Results appended to {results_path}")

    # ---- Show trend across all evaluations so far ----
    if results_path.exists():
        history = pd.read_csv(results_path)
        if history["step_label"].nunique() > 1:
            print()
            print("=== Trend across all evaluations (200 DrawBench prompts) ===")
            print(history.groupby("step_label")["reward"].mean().to_string())


if __name__ == "__main__":
    main()
