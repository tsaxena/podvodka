import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from trl import DPOConfig, DPOTrainer


def build_preference_dataset(train_path, base_model_path, reward_model_path, device,
                              n_candidates=4, max_new_tokens=80):
    """
    Build preference pairs by:
    1. Sampling n_candidates prompts from the SFT model for each description.
    2. Scoring all candidates with the reward model.
    3. Using the highest- and lowest-scored candidates as (chosen, rejected).
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    gen_model.eval()
    if device >= 0:
        gen_model = gen_model.cuda(device)

    reward_pipeline = pipeline("text-classification", model=reward_model_path, device=device)

    @torch.no_grad()
    def score(text):
        return reward_pipeline(text, function_to_apply="none")[0]["score"]

    df = pd.read_csv(train_path)
    descriptions = []
    for text in df["text"]:
        text = text.strip()
        if "</s>" in text:
            descriptions.append(text.split("</s>", 1)[0].strip())
    descriptions = list(dict.fromkeys(descriptions))  # deduplicate, preserve order

    preference_rows = []
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    for desc in tqdm(descriptions, desc="Building preference pairs"):
        query = desc + "</s>"
        input_ids = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).input_ids
        if device >= 0:
            input_ids = input_ids.cuda(device)

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids.expand(n_candidates, -1), **gen_kwargs
            )

        candidates = []
        for out in outputs:
            generated = out[input_ids.shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            if text:
                candidates.append(text)

        if len(candidates) < 2:
            continue

        scored = [(c, score(query + "</s>" + c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen = scored[0][0]
        rejected = scored[-1][0]
        if chosen != rejected:
            preference_rows.append({"prompt": query, "chosen": chosen, "rejected": rejected})

    del gen_model
    return pd.DataFrame(preference_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient (higher = stay closer to reference)")
    parser.add_argument("--num_layers_unfrozen", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--n_candidates", type=int, default=4,
                        help="Candidates to generate per description when building pairs")
    parser.add_argument("--train_path", type=str, default="/mnt/data/train_strings.csv")
    parser.add_argument("--preference_path", type=str, default=None,
                        help="CSV with columns: prompt, chosen, rejected. "
                             "If omitted, pairs are built automatically from --train_path.")
    parser.add_argument("--output_path", type=str, default="/mnt/models/gpt2-large-dpo-prompt-writing")
    parser.add_argument("--reward_model_path", type=str, default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str, default="toloka/gpt2-large-supervised-prompt-writing")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    if args.preference_path is not None:
        pref_df = pd.read_csv(args.preference_path)
    else:
        print("No --preference_path given; building preference pairs from training data...")
        pref_df = build_preference_dataset(
            args.train_path, args.base_model_path, args.reward_model_path,
            device, n_candidates=args.n_candidates,
        )

    dataset = Dataset.from_pandas(pref_df[["prompt", "chosen", "rejected"]])

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
    ref_model = AutoModelForCausalLM.from_pretrained(args.base_model_path)

    # Freeze all layers except the last num_layers_unfrozen transformer blocks and LM head
    for param in model.parameters():
        param.requires_grad = False
    for block in list(model.transformer.h)[-args.num_layers_unfrozen:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    dpo_config = DPOConfig(
        output_dir=args.output_path,
        beta=args.beta,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
        weight_decay=1.0e-6,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_path)


if __name__ == "__main__":
    main()
