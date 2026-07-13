"""
eval_checkpoint.py — Score a FIXED, reusable test-prompt set against any saved GRPO checkpoint.

Why this exists: every completions comparison so far has used whatever prompts happened to be
in that training step's live-logged sample, which differ every time — so reward differences
between samples are confounded with prompt-difficulty differences, not just training progress.
This script removes that confound: the same prompts, generated and scored the same way, every
time you run it. Any reward difference across checkpoints is then real signal.

Usage:
    # Evaluate the current best-so-far checkpoint from an in-progress run:
    python eval_checkpoint.py \
        --checkpoint_path /workspace/podvodka/models/gpt2-large-grpo-prompt-writing-v2/best \
        --step_label 4656

    # Later, evaluate a further-along checkpoint the same way:
    python eval_checkpoint.py \
        --checkpoint_path /workspace/podvodka/models/gpt2-large-grpo-prompt-writing-v2/best \
        --step_label 8000

Results append to --results_csv (default eval_results.csv), so running this against several
checkpoints over time builds a clean, directly-comparable reward trend — the same idea as the
step-335/2353/5402/10000 comparison for v2, but without the different-prompts confound.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Fixed test-prompt set — deliberately hardcoded, not resampled from the training CSV, so it
# NEVER changes between runs of this script. That fixed-ness is the entire point. Includes a
# few prompts that already appeared in earlier live-sample checks (v2's "a hooded guy with a
# gas mask..." / "a young blonde girl", v3's "a movie monster from 80s" / "maps of boston and
# seattle"), so results here can also be loosely cross-referenced against that earlier data.
TEST_PROMPTS = [
    "a movie monster from 80s",
    "maps of boston and seattle",
    "a 2d art background from gta videogame",
    "a skyline of Lyon in summer at 8 am",
    "a sniper taking aim",
    "a 3d colorful cat",
    "a hooded guy with a gas mask holding a vape",
    "a young blonde girl",
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GRPO checkpoint on a fixed prompt set.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a saved checkpoint directory (e.g. .../best or .../final). "
                             "Must contain a full model + tokenizer (as produced by this "
                             "project's save_pretrained calls).")
    parser.add_argument("--reward_model_path", type=str, default="toloka/prompts_reward_model")
    parser.add_argument("--step_label", type=str, default=None,
                        help="Label recorded in the results table (e.g. the training step this "
                             "checkpoint corresponds to). Defaults to the checkpoint dir name.")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Completions sampled per prompt — matches --num_generations from "
                             "training, for a comparably stable per-prompt reward estimate.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--results_csv", type=str, default="eval_results.csv",
                        help="Results are appended here across runs, building a running "
                             "cross-checkpoint comparison table.")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — run this on the GPU pod."
    device = "cuda"
    step_label = args.step_label or Path(args.checkpoint_path).name

    print(f"Loading checkpoint from {args.checkpoint_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation with a causal LM

    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, torch_dtype=model_dtype
    ).to(device)
    model.eval()

    print(f"Loading reward model from {args.reward_model_path} ...")
    reward_pipeline = pipeline("text-classification", model=args.reward_model_path, device=0)

    # Match the training script's prompt formatting exactly (prompt_list = text + "</s>"), so
    # scores here are directly comparable to what was logged during training.
    prompts_with_eos = [p + "</s>" for p in TEST_PROMPTS]

    all_rows = []
    for prompt, prompt_text in zip(TEST_PROMPTS, prompts_with_eos):
        inputs = tokenizer(
            [prompt_text] * args.num_samples, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]  # same for every row: left-padded to equal length
        completions = [
            tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in output_ids
        ]

        reward_texts = [prompt_text + "</s>" + c for c in completions]
        outputs = reward_pipeline(
            reward_texts, function_to_apply="none", batch_size=8, truncation=True
        )
        rewards = [o["score"] for o in outputs]

        for c, r in zip(completions, rewards):
            all_rows.append({
                "checkpoint": args.checkpoint_path,
                "step_label": step_label,
                "prompt": prompt,
                "completion": c,
                "reward": r,
            })

        print(f"  [{prompt[:40]:40s}] mean reward: {sum(rewards) / len(rewards):+.3f}")

    df = pd.DataFrame(all_rows)
    results_path = Path(args.results_csv)
    write_header = not results_path.exists()
    df.to_csv(results_path, mode="a", header=write_header, index=False)

    print()
    print(f"Overall mean reward @ {step_label}: {df['reward'].mean():+.3f}")
    print(f"Results appended to {results_path}")

    if results_path.exists():
        history = pd.read_csv(results_path)
        if history["step_label"].nunique() > 1:
            print()
            print("=== Trend across all checkpoints evaluated so far (same fixed prompts) ===")
            print(history.groupby("step_label")["reward"].mean().to_string())


if __name__ == "__main__":
    main()
