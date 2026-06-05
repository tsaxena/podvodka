"""
UDRL (Upside-Down Reinforcement Learning) training script.

Reference: Srivastava et al. (2019) "Training Agents using Upside-Down RL"
           Schmidhuber (2019) "Reinforcement Learning Upside Down"

Core idea: reframe RL as supervised learning by conditioning the policy on a
*desired return* command token prepended to every input.  The model learns the
mapping  "[reward: +R] description</s>" → high-quality SD prompt, so at
inference time we simply command a high R to elicit good completions.

Algorithm (one phase):
  1. COLLECTION — sample prompts, prepend the current desired-return command,
     generate completions, score with the frozen reward model, push
     (prompt, completion, actual_reward) triples into the replay buffer.
  2. TRAINING   — sample the top `top_frac` of the buffer by reward, form
     supervised examples  "[reward: +actual_R] prompt completion",  mask
     command+prompt tokens from the cross-entropy loss, and run
     `num_train_steps` gradient updates.
  Repeat for `num_phases` phases.  The commanded desired return is linearly
  annealed from `desired_return_init` to `desired_return_final` over phases
  (or set --use_adaptive_command to track the top-fraction buffer mean).

Command format injected into the input:
  "[reward: +0.75] {description}</s>{completion}"

Compare with PPO (run.py)   — same reward model, same base model, same data.
Compare with GRPO (run_grpo.py) — also critic-free; GRPO uses group-relative
advantage estimation whereas UDRL uses reward-conditioned imitation.
"""

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single (prompt, completion, reward) experience triple."""
    prompt: str      # "description</s>"  — already ends with </s>
    completion: str  # model-generated text (may contain <|endoftext|>)
    reward: float    # score from toloka/prompts_reward_model


class ReplayBuffer:
    """Fixed-capacity circular buffer with top-fraction sampling for UDRL."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: List[Episode] = []

    def add(self, new_episodes: List[Episode]) -> None:
        self.episodes.extend(new_episodes)
        if len(self.episodes) > self.capacity:
            # Keep the most recent entries (FIFO overflow).
            self.episodes = self.episodes[-self.capacity:]

    def sample_top(self, k: int, top_frac: float) -> List[Episode]:
        """Sample k episodes uniformly from the top `top_frac` by reward."""
        if not self.episodes:
            return []
        sorted_ep = sorted(self.episodes, key=lambda e: e.reward, reverse=True)
        cutoff = max(k, int(len(sorted_ep) * top_frac))
        pool = sorted_ep[:cutoff]
        return random.sample(pool, min(k, len(pool)))

    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.reward for e in self.episodes) / len(self.episodes)

    def top_mean_reward(self, top_frac: float = 0.5) -> float:
        """Mean reward of the top `top_frac` fraction of the buffer."""
        if not self.episodes:
            return 0.0
        sorted_ep = sorted(self.episodes, key=lambda e: e.reward, reverse=True)
        cutoff = max(1, int(len(sorted_ep) * top_frac))
        return sum(e.reward for e in sorted_ep[:cutoff]) / cutoff

    def __len__(self) -> int:
        return len(self.episodes)


# ---------------------------------------------------------------------------
# Command formatting
# ---------------------------------------------------------------------------

def format_command(reward: float) -> str:
    """Return the text prefix that conditions the model on `reward`."""
    return f"[reward: {reward:+.2f}] "


# ---------------------------------------------------------------------------
# Supervised training dataset
# ---------------------------------------------------------------------------

class CommandDataset(TorchDataset):
    """
    Each sample encodes:
      full_ids : token ids for  "[reward: +R] description</s> completion"
      labels   : same ids but with -100 on the command+description prefix
                 so that only the completion tokens contribute to the loss.
    """

    def __init__(self, episodes: List[Episode], tokenizer, max_length: int):
        self.samples = []
        for ep in episodes:
            cmd = format_command(ep.reward)
            prefix_text = cmd + ep.prompt          # "[reward: +R] description</s>"
            full_text = prefix_text + ep.completion

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prefix_enc = tokenizer(
                prefix_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            full_ids = full_enc.input_ids.squeeze(0)
            prefix_len = prefix_enc.input_ids.shape[1]

            labels = full_ids.clone()
            labels[:prefix_len] = -100  # mask command + prompt from loss

            self.samples.append((full_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_id: int):
    """Pad a batch of (input_ids, labels) pairs to the same length."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(x.size(0) for x in input_ids_list)
    B = len(input_ids_list)
    padded_input = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        n = inp.size(0)
        padded_input[i, :n] = inp
        padded_labels[i, :n] = lbl
    return padded_input, padded_labels


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, tokenizer, optimizer, scheduler,
    out_dir: Path, step: int, best_reward: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_reward": best_reward,
        },
        out_dir / "trainer_state.pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UDRL training — supervised complement to PPO (run.py) and GRPO (run_grpo.py)."
    )

    # ---- UDRL loop ----
    parser.add_argument("--num_phases", type=int, default=20,
                        help="Number of collect→train alternation phases.")
    parser.add_argument("--num_collection_prompts", type=int, default=256,
                        help="Prompts sampled per collection phase.")
    parser.add_argument("--num_train_steps", type=int, default=100,
                        help="Gradient updates per training phase.")
    parser.add_argument("--buffer_size", type=int, default=8192,
                        help="Maximum replay buffer capacity (FIFO overflow).")
    parser.add_argument("--top_frac", type=float, default=0.5,
                        help="Train on the top `top_frac` of the buffer by reward.")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Minibatch size for supervised gradient updates.")

    # ---- Desired-return command schedule ----
    parser.add_argument("--desired_return_init", type=float, default=0.5,
                        help="Desired return used as collection command in phase 0.")
    parser.add_argument("--desired_return_final", type=float, default=1.0,
                        help="Target desired return by the final phase (linear anneal).")
    parser.add_argument("--use_adaptive_command", action="store_true",
                        help="Replace the linear schedule with the top_frac buffer mean.")

    # ---- Model / optimiser ----
    parser.add_argument("--lr", type=float, default=1.4e-5)
    parser.add_argument("--num_layers_unfrozen", type=int, default=2,
                        help="Unfreeze the last N transformer blocks (+ lm_head).")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max tokens for the full command+prompt+completion sequence.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--gen_batch_size", type=int, default=32,
                        help="Prompts per generation forward pass.")
    parser.add_argument("--reward_batch_size", type=int, default=32)

    # ---- Paths ----
    parser.add_argument("--train_path", type=str,
                        default="/workspace/podvodka/data/train_strings.csv")
    parser.add_argument("--val_path", type=str,
                        default="/workspace/podvodka/data/val_strings.csv")
    parser.add_argument("--output_path", type=str,
                        default="/workspace/podvodka/models/gpt2-large-udrl-prompt-writing")
    parser.add_argument("--reward_model_path", type=str,
                        default="toloka/prompts_reward_model")
    parser.add_argument("--base_model_path", type=str,
                        default="tsaxena/gpt2-large-prompt-tags")

    # ---- Checkpointing ----
    parser.add_argument("--save_every", type=int, default=5,
                        help="Periodic checkpoint every N phases (0 = disabled).")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="Number of periodic phase-* checkpoints to retain.")
    parser.add_argument("--save_best_after", type=int, default=3,
                        help="Begin tracking best-reward checkpoints after this phase.")

    # ---- W&B ----
    parser.add_argument("--wandb_project", type=str, default="podvodka-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=["udrl", "gpt2-large"])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--text_log_every", type=int, default=2,
                        help="Log sample generations to W&B every N phases.")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — fix the environment before training."
    device = torch.device("cuda")
    reward_device = int(os.environ.get("LOCAL_RANK", 0))

    # ---- W&B ----
    use_wandb = not args.no_wandb
    if use_wandb:
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
            use_wandb = False

    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Policy model (no value head — UDRL is critic-free) ----
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path).to(device)

    # Freeze everything, then unfreeze the last N blocks and the lm_head.
    for param in model.parameters():
        param.requires_grad = False
    for block in list(model.transformer.h)[-args.num_layers_unfrozen:]:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")

    # ---- Optimiser + cosine LR schedule ----
    total_grad_steps = args.num_phases * args.num_train_steps
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=1.0e-6,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_grad_steps, eta_min=args.lr * 0.1)

    # ---- Reward model (frozen throughout) ----
    reward_pipeline = pipeline(
        "text-classification",
        model=args.reward_model_path,
        device=reward_device,
    )

    @torch.no_grad()
    def score_batch(texts: List[str]) -> List[float]:
        outputs = reward_pipeline(
            texts,
            function_to_apply="none",
            batch_size=args.reward_batch_size,
            truncation=True,
        )
        return [o["score"] for o in outputs]

    # ---- Data ----
    train_df = pd.read_csv(args.train_path)
    all_prompts = [row.split("</s>")[0] + "</s>" for row in train_df["text"]]

    val_df = pd.read_csv(args.val_path)
    eval_prompts = [row.split("</s>")[0] + "</s>" for row in val_df["text"]][:100]

    print("=" * 50)
    print("policy model device :", next(model.parameters()).device)
    print("reward pipe device  :", reward_pipeline.device)
    print("train prompts       :", len(all_prompts))
    print("eval  prompts       :", len(eval_prompts))
    print("=" * 50)
    assert next(model.parameters()).device.type == "cuda", (
        "Policy model is not on GPU. "
        "Delete ~/.cache/huggingface/accelerate/default_config.yaml and retry."
    )

    # ---- Generation config ----
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ---- Replay buffer ----
    buffer = ReplayBuffer(args.buffer_size)

    # ---------------------------------------------------------------------------
    # Collection phase
    # ---------------------------------------------------------------------------

    def collect_episodes(desired_return: float, prompts: List[str]) -> List[Episode]:
        """
        Generate completions conditioned on `desired_return`, score with the RM,
        and return the resulting Episode list.

        Left-padding is used so all inputs in a batch share the same tensor length
        and out[:, input_len:] cleanly extracts the generated tokens.
        """
        model.eval()
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        episodes: List[Episode] = []
        for i in tqdm(range(0, len(prompts), args.gen_batch_size),
                      desc="  collecting", leave=False):
            batch_prompts = prompts[i : i + args.gen_batch_size]
            commanded = [format_command(desired_return) + p for p in batch_prompts]

            enc = tokenizer(
                commanded,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            input_len = enc.input_ids.shape[1]

            with torch.no_grad():
                out = model.generate(
                    enc.input_ids,
                    attention_mask=enc.attention_mask,
                    **gen_kwargs,
                )

            completions = [
                tokenizer.decode(out[j, input_len:], skip_special_tokens=False)
                for j in range(out.shape[0])
            ]

            # Reward scoring format matches run.py: "description</s></s>completion"
            reward_texts = [p + "</s>" + c for p, c in zip(batch_prompts, completions)]
            rewards = score_batch(reward_texts)

            for prompt, completion, reward in zip(batch_prompts, completions, rewards):
                episodes.append(Episode(prompt=prompt, completion=completion, reward=reward))

            del out, enc
            torch.cuda.empty_cache()

        tokenizer.padding_side = prev_padding_side
        return episodes

    # ---------------------------------------------------------------------------
    # Training phase
    # ---------------------------------------------------------------------------

    def run_training_phase(train_episodes: List[Episode]) -> float:
        """
        Supervised UDRL update: condition on actual reward achieved, train to
        reproduce the completion.  Returns mean cross-entropy loss.
        """
        model.train()
        dataset = CommandDataset(train_episodes, tokenizer, max_length=args.max_length)
        if len(dataset) == 0:
            return 0.0

        loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=tokenizer.pad_token_id),
            drop_last=False,
        )

        total_loss = 0.0
        n_steps = 0
        loader_iter = iter(loader)

        # Run exactly num_train_steps gradient steps, cycling the loader if needed.
        for _ in range(args.num_train_steps):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps += 1

        return total_loss / max(n_steps, 1)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(desired_return: float) -> float:
        """Generate on eval_prompts and return mean RM reward."""
        model.eval()
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        all_rewards: List[float] = []
        for i in range(0, len(eval_prompts), args.gen_batch_size):
            batch = eval_prompts[i : i + args.gen_batch_size]
            commanded = [format_command(desired_return) + p for p in batch]
            enc = tokenizer(
                commanded,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            input_len = enc.input_ids.shape[1]
            out = model.generate(
                enc.input_ids,
                attention_mask=enc.attention_mask,
                **gen_kwargs,
            )
            completions = [
                tokenizer.decode(out[j, input_len:], skip_special_tokens=False)
                for j in range(out.shape[0])
            ]
            reward_texts = [p + "</s>" + c for p, c in zip(batch, completions)]
            all_rewards.extend(score_batch(reward_texts))

            del out, enc
            torch.cuda.empty_cache()

        tokenizer.padding_side = prev_padding_side
        return sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    # ---------------------------------------------------------------------------
    # Main UDRL loop
    # ---------------------------------------------------------------------------

    best_reward = float("-inf")
    global_train_step = 0

    try:
        for phase in range(args.num_phases):
            # --- Desired-return schedule ---
            if args.use_adaptive_command and len(buffer) > 0:
                desired_return = buffer.top_mean_reward(args.top_frac)
            else:
                t = phase / max(args.num_phases - 1, 1)
                desired_return = (
                    args.desired_return_init
                    + t * (args.desired_return_final - args.desired_return_init)
                )

            print(f"\n{'='*55}")
            print(f"Phase {phase + 1}/{args.num_phases}  |  "
                  f"desired_return={desired_return:+.3f}  |  "
                  f"buffer={len(buffer)}")
            print(f"{'='*55}")

            # 1. Collection
            sample_prompts = random.sample(
                all_prompts, min(args.num_collection_prompts, len(all_prompts))
            )
            new_episodes = collect_episodes(desired_return, sample_prompts)
            buffer.add(new_episodes)

            col_rewards = [e.reward for e in new_episodes]
            mean_col = sum(col_rewards) / len(col_rewards)
            print(f"  [collect] mean={mean_col:+.3f}  "
                  f"max={max(col_rewards):+.3f}  "
                  f"buffer={len(buffer)}")

            # 2. Training (on the top-fraction of the buffer)
            train_episodes = buffer.sample_top(
                k=min(args.train_batch_size * args.num_train_steps, len(buffer)),
                top_frac=args.top_frac,
            )
            mean_loss = run_training_phase(train_episodes)
            global_train_step += args.num_train_steps
            print(f"  [train]   loss={mean_loss:.4f}  steps={args.num_train_steps}")

            # 3. Evaluation (always command the final desired return)
            eval_reward = evaluate(desired_return=args.desired_return_final)
            print(f"  [eval]    mean_reward={eval_reward:+.3f}")

            # 4. W&B logging
            if use_wandb:
                try:
                    import wandb
                    log_dict = {
                        "phase": phase + 1,
                        "desired_return": desired_return,
                        "collection/mean_reward": mean_col,
                        "collection/max_reward": max(col_rewards),
                        "buffer/size": len(buffer),
                        "buffer/mean_reward": buffer.mean_reward(),
                        "buffer/top_mean_reward": buffer.top_mean_reward(args.top_frac),
                        "train/loss": mean_loss,
                        "eval/mean_reward": eval_reward,
                        "lr": scheduler.get_last_lr()[0],
                    }
                    wandb.log(log_dict, step=global_train_step)

                    if phase % args.text_log_every == 0:
                        table = wandb.Table(
                            columns=["phase", "prompt", "completion", "reward"]
                        )
                        for ep in new_episodes[:3]:
                            table.add_data(phase + 1, ep.prompt, ep.completion, ep.reward)
                        wandb.log({"samples": table}, step=global_train_step)
                except Exception as e:
                    print(f"[wandb] log failed: {e}")

            # 5. Checkpointing
            phase_id = phase + 1

            if args.save_every > 0 and phase_id % args.save_every == 0:
                ckpt_dir = out_root / f"phase-{phase_id:04d}"
                save_checkpoint(model, tokenizer, optimizer, scheduler,
                                ckpt_dir, global_train_step, best_reward)
                print(f"  [ckpt] phase {phase_id} → {ckpt_dir}")

                phase_dirs = sorted(out_root.glob("phase-*"))
                for old in phase_dirs[:-args.keep_last_n]:
                    print(f"  [ckpt] removing {old}")
                    shutil.rmtree(old, ignore_errors=True)

            if phase_id >= args.save_best_after and eval_reward > best_reward:
                best_reward = eval_reward
                best_dir = out_root / "best"
                save_checkpoint(model, tokenizer, optimizer, scheduler,
                                best_dir, global_train_step, best_reward)
                print(f"  [ckpt] new best eval_reward={best_reward:+.3f} → {best_dir}")

    except KeyboardInterrupt:
        print("\n[interrupt] Saving emergency checkpoint before exit...")
        save_checkpoint(model, tokenizer, optimizer, scheduler,
                        out_root / "interrupted", global_train_step, best_reward)
        raise

    print("[ckpt] saving final checkpoint")
    save_checkpoint(model, tokenizer, optimizer, scheduler,
                    out_root / "final", global_train_step, best_reward)

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
