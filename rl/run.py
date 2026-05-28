import argparse
import os
from typing import List

import torch
from transformers import AutoTokenizer, pipeline
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--init_kl_coef", type=float, default=0.05)
    parser.add_argument("--vf_coef", type=float, default=1)
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--gen_batch_size", type=int, default=32)
    parser.add_argument("--reward_batch_size", type=int, default=32)
    parser.add_argument("--train_path", type=str, default="/workspace/podvodka/data/train_strings.csv")
    parser.add_argument("--val_path", type=str, default="/workspace/podvodka/data/val_strings.csv")
    parser.add_argument("--output_path", type=str, default="/workspace/podvodka/models/gpt2-large-rl-prompt-writing")
    parser.add_argument("--reward_model_path", type=str, default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str, default="tsaxena/gpt2-large-prompt-tags")

    # ---- W&B logging ----
    parser.add_argument("--wandb_project", type=str, default="podvodka-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name. If unset, wandb auto-generates one.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B team/user. Leave unset to use your default.")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=["ppo", "gpt2-large"])
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging (useful for local debugging).")
    parser.add_argument("--text_log_every", type=int, default=10,
                        help="Log query/response/reward text table every N steps "
                             "(set to 1 to log every step; high values save W&B storage).")

    args = parser.parse_args()

    log_with = None if args.no_wandb else "wandb"
    tracker_kwargs = {}
    if log_with == "wandb":
        wandb_kwargs = {"tags": args.wandb_tags}
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        tracker_kwargs["wandb"] = wandb_kwargs

    assert torch.cuda.is_available(), "CUDA not available — fix the environment before training."
    reward_device = int(os.environ.get("LOCAL_RANK", 0))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.base_model_path)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.base_model_path)

    # Freeze all layers except the last num_layers_unfrozen transformer blocks and value head
    for param in model.pretrained_model.parameters():
        param.requires_grad = False
    for block in list(model.pretrained_model.transformer.h)[-args.num_layers_unfrozen:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.v_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-6,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=args.lr * 0.1)

    ppo_config = PPOConfig(
        model_name=args.base_model_path,
        learning_rate=args.lr,
        batch_size=args.num_rollouts,
        mini_batch_size=args.chunk_size,
        gradient_accumulation_steps=1,
        ppo_epochs=4,
        init_kl_coef=args.init_kl_coef,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=args.vf_coef,
        accelerator_kwargs={"cpu": False},
        # ---- W&B ----
        log_with=log_with,
        tracker_project_name=args.wandb_project,
        tracker_kwargs=tracker_kwargs,
    )

    reward_pipeline = pipeline(
        "text-classification",
        model=args.reward_model_path,
        device=reward_device,
    )

    @torch.no_grad()
    def score_batch(texts: List[str]) -> List[torch.Tensor]:
        outputs = reward_pipeline(
            texts,
            function_to_apply="none",
            batch_size=args.reward_batch_size,
            truncation=True,
        )
        return [torch.tensor(o["score"], dtype=torch.float) for o in outputs]

    train_df = pd.read_csv(args.train_path)
    prompts = [l.split("</s>")[0] + "</s>" for l in train_df["text"]]

    val_df = pd.read_csv(args.val_path)
    eval_prompts = [l.split("</s>")[0] + "</s>" for l in val_df["text"]][:100]

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
    )

    # Log all CLI args as the run's hyperparameter config (visible on the W&B Overview tab).
    if log_with == "wandb" and ppo_trainer.accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(vars(args), allow_val_change=True)
        except Exception as e:
            print(f"[wandb] config.update skipped: {e}")

    # Device sanity check
    device = ppo_trainer.accelerator.device
    print("=" * 50)
    print("accelerator device :", device)
    print("policy model device :", next(model.parameters()).device)
    print("reward pipe device  :", reward_pipeline.device)
    print("=" * 50)
    assert device.type == "cuda", (
        f"PPOTrainer is on {device}, not GPU. "
        "Delete ~/.cache/huggingface/accelerate/default_config.yaml and retry."
    )

    gen_kwargs = dict(
        max_new_tokens=80,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    for step in tqdm(range(10000)):
        indices = torch.randint(0, len(prompts), (args.num_rollouts,))
        batch_prompts = [prompts[i] for i in indices]

        query_tensors = [
            tokenizer(p, return_tensors="pt", truncation=True, max_length=1024)
            .input_ids.squeeze(0)
            .to(device)
            for p in batch_prompts
        ]

        full_sequences = ppo_trainer.generate(
            query_tensors, batch_size=args.gen_batch_size, **gen_kwargs
        )
        response_tensors = [r[len(q):] for q, r in zip(query_tensors, full_sequences)]
        batch_responses = [tokenizer.decode(r, skip_special_tokens=False) for r in response_tensors]

        reward_texts = [p + "</s>" + r for p, r in zip(batch_prompts, batch_responses)]
        rewards = score_batch(reward_texts)

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Always log scalar PPO stats. Only attach the heavy text table every N steps
        # so W&B doesn't store 10k copies of full query/response strings.
        if step % args.text_log_every == 0:
            text_batch = {"query": batch_prompts, "response": batch_responses}
        else:
            text_batch = {
                "query": [""] * len(batch_prompts),
                "response": [""] * len(batch_responses),
            }
        ppo_trainer.log_stats(stats, text_batch, rewards)

    ppo_trainer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()