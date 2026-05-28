"""
GRPO training script — critic-free complement to run.py (PPO).

Key differences vs PPO:
  - No value head: uses AutoModelForCausalLM, not AutoModelForCausalLMWithValueHead.
  - Group-relative advantage: for each prompt, G completions are sampled;
    advantage = (reward − group_mean) / group_std — no separate critic needed.
  - Controlled by --num_generations (group size G) instead of --vf_coef.

Requires trl >= 0.12.0 (GRPOTrainer / GRPOConfig).
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    pipeline,
)
import pandas as pd
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------

def save_snapshot(model, tokenizer, out_dir: Path, step: int, best_reward: float):
    """Save model weights + tokenizer + a small metadata file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    torch.save({"step": step, "best_reward": best_reward}, out_dir / "trainer_state.pt")


class BestRewardCheckpointer(TrainerCallback):
    """Saves the model whenever mean reward exceeds the running best (after warmup)."""

    def __init__(self, out_root: Path, tokenizer, save_best_after: int):
        self.out_root = out_root
        self.tokenizer = tokenizer
        self.save_best_after = save_best_after
        self.best_reward = float("-inf")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if logs is None:
            return
        step = state.global_step
        # TRL logs GRPO mean reward as "rewards/mean"; fall back to "reward" for older builds.
        reward = logs.get("rewards/mean", logs.get("reward", None))
        if reward is None or step < self.save_best_after:
            return
        if reward > self.best_reward:
            self.best_reward = reward
            model = kwargs.get("model")
            if model is not None:
                best_dir = self.out_root / "best"
                save_snapshot(model, self.tokenizer, best_dir, step, reward)
                print(f"[ckpt] new best reward={reward:+.3f} at step {step} → {best_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning — drop-in complement to run.py (PPO)."
    )

    # ---- Core hyperparameters ----
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--beta", type=float, default=0.05,
                        help="KL penalty coefficient (analogous to init_kl_coef in PPO).")
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Completions per prompt (group size G). "
                             "Total samples per step = batch_size × num_generations.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Prompts per gradient step.")
    parser.add_argument("--grpo_epochs", type=int, default=1,
                        help="Inner optimisation epochs per rollout batch (num_iterations).")
    parser.add_argument("--cliprange", type=float, default=0.2,
                        help="Ratio clip range (ε in GRPO, analogous to PPO cliprange).")
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--reward_batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=10000)

    # ---- Generation ----
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. 1.0 = top_p sampling as in PPO run.")

    # ---- Paths ----
    parser.add_argument("--train_path", type=str,
                        default="/workspace/podvodka/data/train_strings.csv")
    parser.add_argument("--val_path", type=str,
                        default="/workspace/podvodka/data/val_strings.csv")
    parser.add_argument("--output_path", type=str,
                        default="/workspace/podvodka/models/gpt2-large-grpo-prompt-writing")
    parser.add_argument("--reward_model_path", type=str,
                        default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str,
                        default="tsaxena/gpt2-large-prompt-tags")

    # ---- Checkpointing ----
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save a checkpoint every N steps (0 = disabled).")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="Keep only the most recent N periodic checkpoints.")
    parser.add_argument("--save_best_after", type=int, default=20,
                        help="Start tracking best-reward checkpoints after this step.")

    # ---- W&B ----
    parser.add_argument("--wandb_project", type=str, default="podvodka-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=["grpo", "gpt2-large"])
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — fix the environment before training."
    reward_device = int(os.environ.get("LOCAL_RANK", 0))

    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model (no value head — GRPO is critic-free) ----
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    for param in model.parameters():
        param.requires_grad = False
    for block in list(model.transformer.h)[-args.num_layers_unfrozen:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # ---- Reward pipeline ----
    reward_pipeline = pipeline(
        "text-classification",
        model=args.reward_model_path,
        device=reward_device,
    )

    @torch.no_grad()
    def reward_fn(completions: List[str], prompt: List[str] = None, **kwargs) -> List[float]:
        # prompt column is repeated G times by GRPOTrainer (one entry per completion).
        # Format mirrors run.py: p already ends in </s>, producing p + </s> + completion.
        reward_texts = [p + "</s>" + c for p, c in zip(prompt, completions)]
        outputs = reward_pipeline(
            reward_texts,
            function_to_apply="none",
            batch_size=args.reward_batch_size,
            truncation=True,
        )
        return [o["score"] for o in outputs]

    # ---- Dataset ----
    train_df = pd.read_csv(args.train_path)
    prompt_list = [l.split("</s>")[0] + "</s>" for l in train_df["text"]]
    train_dataset = Dataset.from_dict({"prompt": prompt_list})

    # ---- W&B (init before GRPOConfig so entity/tags attach to the correct run) ----
    report_to = "none" if args.no_wandb else "wandb"
    if report_to == "wandb":
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                tags=args.wandb_tags or [],
                config=vars(args),
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    # ---- GRPO config ----
    grpo_config = GRPOConfig(
        output_dir=str(out_root),
        # Optimiser — match PPO run defaults
        learning_rate=args.lr,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
        weight_decay=1.0e-6,
        lr_scheduler_type="cosine",
        warmup_steps=0,
        # Training loop
        per_device_train_batch_size=args.batch_size,
        max_steps=args.num_steps,
        num_generations=args.num_generations,
        num_iterations=args.grpo_epochs,
        # GRPO-specific
        beta=args.beta,
        epsilon=args.cliprange,
        # Generation
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=0,
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        # Logging / saving
        logging_steps=1,
        report_to=report_to,
        save_steps=args.save_every if args.save_every > 0 else args.num_steps + 1,
        save_total_limit=args.keep_last_n,
    )

    best_reward_cb = BestRewardCheckpointer(
        out_root=out_root,
        tokenizer=tokenizer,
        save_best_after=args.save_best_after,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[best_reward_cb],
    )

    print("=" * 50)
    print("policy model device :", next(model.parameters()).device)
    print("reward pipe device  :", reward_pipeline.device)
    print("num_generations (G) :", args.num_generations)
    print("effective batch     :", args.batch_size * args.num_generations, "completions/step")
    print("=" * 50)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[interrupt] Saving emergency checkpoint before exit...")
        trainer.save_model(str(out_root / "interrupted"))
        tokenizer.save_pretrained(str(out_root / "interrupted"))
        raise

    print("[ckpt] saving final checkpoint")
    trainer.save_model(str(out_root / "final"))
    tokenizer.save_pretrained(str(out_root / "final"))


if __name__ == "__main__":
    main()
