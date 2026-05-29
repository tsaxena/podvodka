"""
DPO (Direct Preference Optimization) training for prompt-writing.

Two phases:
  1. Build a preference dataset from the SFT model + reward model
     (or load a pre-built one with --dataset_path).
  2. Train the policy with DPOTrainer.

Why DPO instead of PPO here:
  - No RL loop, much simpler.
  - No reward model in the training inner loop, so no PPO-style reward hacking
    (different failure modes apply — see notes at bottom).
  - The reference model (SFT) constrains drift via the DPO loss directly.

Example
-------
# Build a fresh preference dataset and train
python train_dpo.py \\
    --build_dataset \\
    --num_prompts 1000 \\
    --candidates_per_prompt 8 \\
    --output_path /workspace/podvodka/models/gpt2-large-dpo-prompt-tags

# Or train on an existing dataset
python train_dpo.py \\
    --dataset_path /workspace/podvodka/data/preferences.jsonl \\
    --output_path /workspace/podvodka/models/gpt2-large-dpo-prompt-tags
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import DPOTrainer


# ============================================================
# PHASE 1: Build a preference dataset
# ============================================================

@torch.no_grad()
def generate_candidates(model, tokenizer, prompts: List[str], k: int,
                        max_new_tokens: int, device: str) -> List[List[str]]:
    """Generate k completions per prompt using the SFT model."""
    out = []
    for i, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt", truncation=True,
                        max_length=1024).input_ids.to(device)
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
            tokenizer.decode(g[ids.shape[1]:], skip_special_tokens=False)
            for g in generated
        ]
        out.append(completions)
        if (i + 1) % 50 == 0:
            print(f"  generated for {i+1}/{len(prompts)} prompts")
    return out


def score_completions(reward_pipeline, prompts: List[str],
                      completions: List[List[str]],
                      batch_size: int = 32) -> List[List[float]]:
    """Score every (prompt, completion) with the reward model."""
    flat_texts = [
        p + "</s>" + c
        for p, comps in zip(prompts, completions)
        for c in comps
    ]
    print(f"  scoring {len(flat_texts)} (prompt, completion) pairs")
    outs = reward_pipeline(flat_texts, function_to_apply="none",
                           batch_size=batch_size, truncation=True)
    flat_scores = [o["score"] for o in outs]

    # Reshape back to per-prompt lists
    k = len(completions[0])
    return [flat_scores[i * k:(i + 1) * k] for i in range(len(prompts))]


def build_preference_pairs(prompts: List[str],
                           completions: List[List[str]],
                           scores: List[List[float]],
                           min_score_gap: float = 0.3) -> List[Dict]:
    """For each prompt, form a chosen/rejected pair from highest/lowest scoring
    completions. Skip prompts where the gap is too small (noisy preferences)."""
    pairs = []
    skipped_small_gap = 0
    for p, comps, scs in zip(prompts, completions, scores):
        best_idx = max(range(len(scs)), key=lambda i: scs[i])
        worst_idx = min(range(len(scs)), key=lambda i: scs[i])
        gap = scs[best_idx] - scs[worst_idx]
        if gap < min_score_gap:
            skipped_small_gap += 1
            continue
        pairs.append({
            "prompt": p,
            "chosen": comps[best_idx],
            "rejected": comps[worst_idx],
            "chosen_score": scs[best_idx],
            "rejected_score": scs[worst_idx],
            "score_gap": gap,
        })
    print(f"  built {len(pairs)} pairs "
          f"({skipped_small_gap} skipped: gap < {min_score_gap})")
    return pairs


def build_dataset(args):
    print("Phase 1: building preference dataset")
    device = "cuda:0"
    dtype = torch.float16

    # Load SFT model for candidate generation
    print(f"  loading SFT model: {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sft = AutoModelForCausalLM.from_pretrained(
        args.sft_model, torch_dtype=dtype
    ).to(device).eval()

    # Load prompts
    train_df = pd.read_csv(args.train_path)
    prompts = [l.split("</s>")[0] + "</s>" for l in train_df["text"]]
    prompts = prompts[:args.num_prompts]
    print(f"  using {len(prompts)} prompts from {args.train_path}")

    # Generate
    print("  generating candidates...")
    completions = generate_candidates(
        sft, tokenizer, prompts,
        k=args.candidates_per_prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    # Free SFT memory before loading reward pipeline
    del sft
    torch.cuda.empty_cache()

    # Score
    print(f"  loading reward model: {args.reward_model}")
    rm_pipe = pipeline("text-classification", model=args.reward_model,
                       device=0, torch_dtype=dtype)
    scores = score_completions(rm_pipe, prompts, completions,
                                batch_size=args.reward_batch_size)
    del rm_pipe
    torch.cuda.empty_cache()

    # Build pairs
    pairs = build_preference_pairs(prompts, completions, scores,
                                    min_score_gap=args.min_score_gap)

    # Save
    out_path = Path(args.dataset_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  wrote {len(pairs)} pairs to {out_path}")
    return pairs


# ============================================================
# PHASE 2: Train with DPO
# ============================================================

def load_dataset_from_jsonl(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # DPOTrainer expects 'prompt', 'chosen', 'rejected' columns
            rows.append({
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            })
    return Dataset.from_list(rows)


def train_dpo(args):
    print("Phase 2: training with DPO")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Both policy and reference start from the SFT checkpoint.
    # The reference stays frozen; the policy is what gets trained.
    print(f"  loading policy from {args.sft_model}")
    policy = AutoModelForCausalLM.from_pretrained(args.sft_model)
    print(f"  loading reference from {args.sft_model} (frozen)")
    reference = AutoModelForCausalLM.from_pretrained(args.sft_model)

    dataset = load_dataset_from_jsonl(args.dataset_path)
    print(f"  dataset: {len(dataset)} pairs")

    # Optional: hold out a small eval set
    split = dataset.train_test_split(test_size=min(200, len(dataset) // 10),
                                      seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # trl 0.11.4's DPOConfig is imported from trl.trainer
    from trl import DPOConfig

    dpo_config = DPOConfig(
        output_dir=args.output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        evaluation_strategy="steps",
        beta=args.beta,                  # DPO temperature; 0.1 is standard
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to=["wandb"] if not args.no_wandb else [],
        run_name=args.wandb_run_name,
        bf16=True,                       # A100 handles this well
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"  saved DPO model to {args.output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()

    # Mode
    p.add_argument("--build_dataset", action="store_true",
                   help="Build the preference dataset before training.")

    # Models
    p.add_argument("--sft_model", default="tsaxena/gpt2-large-prompt-tags",
                   help="SFT model; used as both initial policy and frozen reference.")
    p.add_argument("--reward_model", default="toloka/prompts_reward_model",
                   help="RM used to score candidates when building pairs.")

    # Data
    p.add_argument("--train_path",
                   default="/workspace/podvodka/data/train_strings.csv")
    p.add_argument("--dataset_path",
                   default="/workspace/podvodka/data/preferences.jsonl",
                   help="Where preference pairs are saved/loaded.")
    p.add_argument("--num_prompts", type=int, default=1000)
    p.add_argument("--candidates_per_prompt", type=int, default=8)
    p.add_argument("--min_score_gap", type=float, default=0.3,
                   help="Skip pairs where chosen-rejected gap is below this.")
    p.add_argument("--reward_batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=80)

    # DPO training
    p.add_argument("--output_path",
                   default="/workspace/podvodka/models/gpt2-large-dpo-prompt-tags")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="DPO is sensitive to LR. 1e-6 to 5e-6 is typical.")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature. Higher = stronger pull toward chosen, "
                        "but also stronger push away from reference. 0.1 is a "
                        "common starting point.")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)

    # Logging
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_run_name", default="dpo-gpt2-large-prompts")

    args = p.parse_args()

    if args.build_dataset:
        build_dataset(args)

    train_dpo(args)


if __name__ == "__main__":
    main()
