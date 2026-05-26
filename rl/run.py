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
    parser.add_argument("--train_path", type=str, default="/mnt/data/train_strings.csv")
    parser.add_argument("--val_path", type=str, default="/mnt/data/val_strings.csv")
    parser.add_argument("--output_path", type=str, default="/mnt/models/gpt2-large-rl-prompt-writing")
    parser.add_argument("--reward_model_path", type=str, default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str, default="toloka/gpt2-large-supervised-prompt-writing")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

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
    scheduler = CosineAnnealingLR(optimizer, T_max=10000, eta_min=args.lr)

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
        cliprange_reward=10,
    )

    reward_pipeline = pipeline("text-classification", model=args.reward_model_path, device=device)

    @torch.no_grad()
    def score(text):
        return reward_pipeline(text, function_to_apply="none")[0]["score"]

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
            tokenizer(p, return_tensors="pt", truncation=True, max_length=1024).input_ids.squeeze(0)
            for p in batch_prompts
        ]

        full_sequences = ppo_trainer.generate(query_tensors, **gen_kwargs)
        response_tensors = [r[len(q):] for q, r in zip(query_tensors, full_sequences)]

        batch_responses = [tokenizer.decode(r, skip_special_tokens=False) for r in response_tensors]

        rewards = [
            torch.tensor(score(p + "</s>" + r), dtype=torch.float)
            for p, r in zip(batch_prompts, batch_responses)
        ]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, {"query": batch_prompts, "response": batch_responses}, rewards)

    ppo_trainer.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
