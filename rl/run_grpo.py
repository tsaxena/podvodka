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
import sys
import threading
from collections import deque
from pathlib import Path
from statistics import median
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
import trl.trainer.grpo_trainer as _trl_grpo_trainer_module

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# KL-explosion guard
# ---------------------------------------------------------------------------

class GuardedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that skips (zeroes) the update for a microbatch if its KL explodes.

    TRL's GRPO KL estimator (k3 / Schulman estimator) is computed per-token as
        exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
    and the batch-mean of this feeds directly into the loss as `beta * kl`. Because the
    estimator sits inside an exp(), a single token where the current policy has collapsed
    to near-zero probability relative to the reference (e.g. a degenerate short completion,
    or a bf16 precision artifact on a low-probability token) is enough to send that step's
    mean KL — and therefore the loss and gradient — into the thousands or tens of thousands.
    A single such step can badly corrupt the policy in a way training can't recover from.

    This subclass tracks a rolling median of "normal" (non-exploded) per-step KL and, if a
    step's KL exceeds max(kl_explosion_floor, kl_explosion_multiple * rolling_median), it
    multiplies that microbatch's loss by 0 before returning it. Backward() still runs (so
    gradient accumulation and the training loop proceed normally), but this microbatch
    contributes exactly zero gradient — a no-op update instead of a potentially destructive
    one. The true (unzeroed) loss/KL/grad-norm are still recorded under separate metric keys
    (kl_guard/*) so explosions remain visible in your logs rather than being silently hidden.

    This is a safety net, not a fix for whatever is causing the collapse in the first place —
    if it's triggering often, treat that as a signal to investigate (e.g. bf16 log-prob
    precision, degenerate generations, beta/lr too high), not just to trust the guard.
    """

    def __init__(
        self,
        *args,
        kl_explosion_multiple: float = 15.0,
        kl_explosion_floor: float = 2000.0,
        kl_history_len: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kl_explosion_multiple = kl_explosion_multiple
        self.kl_explosion_floor = kl_explosion_floor
        self._kl_recent = deque(maxlen=kl_history_len)
        self.num_skipped_steps = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

        if self.beta == 0.0:
            return loss  # no KL term is computed at all — nothing to guard against

        mode = "eval" if self.control.should_evaluate else "train"
        kl_log = self._metrics[mode].get("kl")
        if not kl_log:
            return loss
        current_kl = kl_log[-1]

        threshold = self.kl_explosion_floor
        if self._kl_recent:
            threshold = max(threshold, self.kl_explosion_multiple * median(self._kl_recent))
        exploded = current_kl > threshold

        if mode != "train":
            return loss

        if not exploded:
            # Only feed "normal" steps into the rolling baseline, so one spike doesn't drag
            # the threshold up and mask the next one.
            self._kl_recent.append(current_kl)
            self._metrics[mode]["kl_guard/triggered"].append(0.0)
            return loss

        self.num_skipped_steps += 1
        self._metrics[mode]["kl_guard/triggered"].append(1.0)
        self._metrics[mode]["kl_guard/raw_loss"].append(loss.detach().item())
        self._metrics[mode]["kl_guard/raw_kl"].append(current_kl)
        raw_grad_norm = self._compute_raw_grad_norm(model, loss)
        if raw_grad_norm is not None:
            self._metrics[mode]["kl_guard/raw_grad_norm"].append(raw_grad_norm)
        baseline = median(self._kl_recent) if self._kl_recent else float("nan")
        grad_norm_str = f"{raw_grad_norm:.1f}" if raw_grad_norm is not None else "n/a"
        print(
            f"[kl-guard] step {self.state.global_step}: kl={current_kl:.1f} exceeded "
            f"threshold={threshold:.1f} (rolling median={baseline:.1f}); raw_grad_norm="
            f"{grad_norm_str}; zeroing this microbatch's update "
            f"({self.num_skipped_steps} skipped so far)."
        )
        # Preserve the autograd graph (so backward() still runs cleanly and gradient
        # accumulation isn't disrupted) but contribute exactly zero gradient.
        return loss * 0.0

    def _compute_raw_grad_norm(self, model, loss):
        """What grad_norm *would* have been for this step's raw (unzeroed) loss.

        This exists because once we zero the loss, the applied gradient — and therefore
        the normal `grad_norm` metric the Trainer logs — is exactly 0 on an exploded step,
        which hides how bad the raw gradient actually was. We recompute it here.

        Uses torch.autograd.grad (functional) rather than loss.backward(), which is
        critical: .backward() would accumulate into each parameter's .grad, and since the
        loss we ultimately return (loss * 0.0) contributes zero on top of whatever's
        already there, any leftover .grad from a naive diagnostic backward() would flow
        straight into the optimizer step — silently reintroducing the exact explosion this
        guard exists to prevent. autograd.grad returns gradients as plain tensors without
        touching .grad, so this is purely read-only.
        """
        try:
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            sq_sum = sum(g.detach().float().pow(2).sum() for g in grads if g is not None)
            return sq_sum.sqrt().item()
        except Exception as e:
            print(f"[kl-guard] raw_grad_norm computation failed (non-fatal): {e}")
            return None


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------
#
# The naive version of this (call model.save_pretrained() straight from the on_log
# callback) blocks the training loop for the full duration of the disk write — on
# network-attached storage that's easily 10-100x slower than local NVMe, and a save
# can visibly stall training for a long time even for a "small" ~774M param model.
# Two changes fix that:
#   1. Only the fast part (copying weights GPU -> CPU, a few seconds over PCIe) happens
#      synchronously. The slow part (serializing to disk) happens in a background thread,
#      so training resumes immediately after the CPU copy.
#   2. A minimum step interval between best-reward saves, since early in training reward
#      is noisy enough that "new best" can fire on nearly consecutive steps — each save
#      is a full ~1.5-3GB write, so back-to-back saves compound the stall.
#
# We rebuild a fresh CPU-only model shell (AutoModelForCausalLM.from_config) and
# load_state_dict into it inside the background thread, then call the real
# save_pretrained() on that shell. This — rather than hand-rolling a safetensors writer —
# is what correctly preserves tied weights (e.g. GPT-2's wte/lm_head tying); save_pretrained
# already contains the logic to detect and dedupe tied parameters, so reusing it avoids
# subtly producing a checkpoint that fails to reload.

def _background_save(model_config, state_dict: dict, tokenizer, out_dir: Path, step: int, reward: float, label: str):
    try:
        target_dtype = next(iter(state_dict.values())).dtype
        # Build the shell in default dtype, then cast explicitly — from_config's torch_dtype
        # kwarg support has varied across transformers versions, so don't rely on it alone.
        shell = AutoModelForCausalLM.from_config(model_config).to(dtype=target_dtype)
        shell.load_state_dict(state_dict)
        out_dir.mkdir(parents=True, exist_ok=True)
        shell.save_pretrained(str(out_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(out_dir))
        torch.save({"step": step, "best_reward": reward}, out_dir / "trainer_state.pt")
        print(f"[ckpt] ({label}) background save finished: step {step}, reward={reward:+.3f} → {out_dir}")
    except Exception as e:
        print(f"[ckpt] ({label}) background save FAILED at step {step}: {e}")


class BestRewardCheckpointer(TrainerCallback):
    """Saves the model whenever mean reward exceeds the running best (after warmup).

    Saving happens on a background thread (see _background_save) so it doesn't block
    the training loop, and is throttled by save_best_min_interval so noisy early-training
    reward spikes don't trigger a pile-up of back-to-back saves.
    """

    def __init__(self, out_root: Path, tokenizer, save_best_after: int, save_best_min_interval: int = 100):
        self.out_root = out_root
        self.tokenizer = tokenizer
        self.save_best_after = save_best_after
        self.save_best_min_interval = save_best_min_interval
        self.best_reward = float("-inf")
        self.last_save_step = -save_best_min_interval  # so the first eligible step can always save
        self._save_lock = threading.Lock()
        self._threads: List[threading.Thread] = []

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
        if reward <= self.best_reward:
            return
        if step - self.last_save_step < self.save_best_min_interval:
            # Still a genuine new best — just too soon after the last save to act on it.
            # Update best_reward anyway so we don't re-trigger on this same value later.
            self.best_reward = reward
            return
        if not self._save_lock.acquire(blocking=False):
            print(f"[ckpt] new best reward={reward:+.3f} at step {step}, but a save is already "
                  f"in progress — skipping (next new-best will catch up).")
            return

        self.best_reward = reward
        self.last_save_step = step
        model = kwargs.get("model")
        if model is None:
            self._save_lock.release()
            return

        # Fast, synchronous part: copy weights off the GPU. This is a PCIe memcpy
        # (seconds, not the tens of seconds a network-disk write can take), so it's fine
        # to do inline. Everything slow happens in the background thread below.
        state_dict = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        config = model.config
        best_dir = self.out_root / "best"

        def _run():
            try:
                _background_save(config, state_dict, self.tokenizer, best_dir, step, reward, "best")
            finally:
                self._save_lock.release()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        self._threads.append(thread)
        print(f"[ckpt] new best reward={reward:+.3f} at step {step} — saving in background → {best_dir}")

    def join(self, timeout: Optional[float] = None):
        """Block until all in-flight background saves finish. Call before process exit."""
        for t in self._threads:
            t.join(timeout=timeout)


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
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Linear LR warmup steps before the cosine schedule kicks in. "
                             "0 disables warmup (not recommended — early steps have the "
                             "noisiest advantage estimates).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm, applied before the optimizer step.")
    parser.add_argument("--reward_clip", type=float, default=5.0,
                        help="Clip raw reward-model logits to [-reward_clip, +reward_clip] "
                             "before group-relative advantage is computed. Guards against a "
                             "single outlier in a small generation group (num_generations) "
                             "distorting the whole group's advantage estimate. Set <= 0 to disable.")
    parser.add_argument("--kl_guard_floor", type=float, default=2000.0,
                        help="Absolute KL threshold above which a step is treated as an "
                             "explosion and its update is zeroed, regardless of recent history.")
    parser.add_argument("--kl_guard_multiple", type=float, default=15.0,
                        help="A step is also treated as an explosion if its KL exceeds this "
                             "multiple of the rolling median of recent non-exploded steps.")
    parser.add_argument("--kl_guard_disable", action="store_true",
                        help="Disable the KL-explosion guard entirely (use plain GRPOTrainer).")

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
    parser.add_argument("--local_scratch_dir", type=str, default="/tmp/grpo_scratch",
                        help="Directory passed as GRPOConfig's output_dir. TRL writes to this "
                             "directory internally on every logging_steps interval — periodic "
                             "HF Trainer checkpoints, and (critically) a completions/*.parquet "
                             "file on every log() call when --log_completions is set. With "
                             "logging_steps=1 that's a disk write roughly every 2-3 seconds for "
                             "the whole run; on slow/network-mounted storage this can eventually "
                             "surface as file-handle or I/O errors (e.g. pyarrow 'error closing "
                             "file'). Pointing this at local fast disk avoids that. This is "
                             "SEPARATE from --output_path: best/final/interrupted checkpoints "
                             "(the ones you actually care about) always go to --output_path "
                             "regardless of this setting.")

    # ---- Checkpointing ----
    parser.add_argument("--save_every", type=int, default=2000,
                        help="Save a checkpoint every N steps (0 = disabled).")
    parser.add_argument("--keep_last_n", type=int, default=1,
                        help="Keep only the most recent N periodic checkpoints.")
    parser.add_argument("--save_best_after", type=int, default=200,
                        help="Start tracking best-reward checkpoints after this step.")
    parser.add_argument("--save_best_min_interval", type=int, default=100,
                        help="Minimum steps between best-reward saves. Prevents noisy "
                             "early-training reward spikes from triggering back-to-back "
                             "full-model saves.")

    # ---- Debugging / visibility ----
    parser.add_argument("--log_completions", action="store_true",
                        help="Log actual prompt/completion/reward samples to W&B (and print "
                             "them to console). Essential for spotting reward hacking / "
                             "degenerate completions that scalar metrics alone won't show.")
    parser.add_argument("--print_completions_console", action="store_true",
                        help="With --log_completions, TRL also prints a full Rich table of "
                             "every sample to the console every logging_steps (=1 here, so "
                             "every step) — useful briefly, but overwhelming over a long run. "
                             "Off by default: completions still go to the W&B table, just not "
                             "the terminal. Pass this to get the console dump back too.")

    # ---- W&B ----
    parser.add_argument("--wandb_project", type=str, default="podvodka-rl")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=["grpo", "gpt2-large"])
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    if args.log_completions and not args.print_completions_console:
        # TRL's _generate_and_score_completions gates the Rich console dump on
        # is_rich_available() (imported into this module's namespace), separately from the
        # W&B table logging. Patching it to False here suppresses only the terminal spam —
        # the W&B completions table (which is where you actually want to browse this data
        # over a long run) is untouched.
        _trl_grpo_trainer_module.is_rich_available = lambda: False

    assert torch.cuda.is_available(), "CUDA not available — fix the environment before training."
    reward_device = int(os.environ.get("LOCAL_RANK", 0))

    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    scratch_root = Path(args.local_scratch_dir)
    scratch_root.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Model (no value head — GRPO is critic-free) ----
    # Loading in bf16 directly (rather than fp32-then-autocast) roughly halves both GPU
    # memory and checkpoint size/write time, which is most of what was making saves slow.
    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=model_dtype)

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
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        # prompts is repeated G times by GRPOTrainer (one entry per completion).
        # Format mirrors run.py: p already ends in </s>, producing p + </s> + completion.
        reward_texts = [p + "</s>" + c for p, c in zip(prompts, completions)]
        outputs = reward_pipeline(
            reward_texts,
            function_to_apply="none",
            batch_size=args.reward_batch_size,
            truncation=True,
        )
        scores = [o["score"] for o in outputs]
        if args.reward_clip > 0:
            # A single outlier logit inside a small generation group (num_generations)
            # can otherwise dominate the group_mean/group_std used for advantages.
            scores = [max(-args.reward_clip, min(args.reward_clip, s)) for s in scores]
        return scores

    # ---- Dataset ----
    train_df = pd.read_csv(args.train_path)
    prompt_list = [l.split("</s>")[0] + "</s>" for l in train_df["text"]]
    train_dataset = Dataset.from_dict({"prompt": prompt_list})

    # ---- W&B (init before GRPOConfig so entity/tags attach to the correct run) ----
    report_to = "none" if args.no_wandb else "wandb"
    if report_to == "wandb":
        try:
            import wandb
            # wandb.init defaults `dir` to the current working directory, which here is on
            # the same network-mounted (MooseFS) filesystem as everything else. wandb stages
            # media artifacts (e.g. the completions Table this script logs) as temp files
            # under /tmp, then shutil.move()s them into its run dir — that's a cross-device
            # move from local /tmp to network storage, which falls back to copy+delete, which
            # can itself fail (or hang) against a flaky network mount. Pointing wandb's local
            # dir at the same local scratch disk as GRPOConfig's output_dir keeps the source
            # and destination on the same filesystem, avoiding this entirely. Uploads to the
            # actual W&B servers are HTTP, not filesystem, so this doesn't affect what you see
            # in the dashboard — only where local staging files live before being uploaded.
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                tags=args.wandb_tags or [],
                config=vars(args),
                dir=str(scratch_root),
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    # ---- GRPO config ----
    grpo_config = GRPOConfig(
        # Deliberately scratch_root, NOT out_root: TRL writes to this directory internally on
        # every logging_steps interval (periodic checkpoints, and completions/*.parquet when
        # log_completions is set). Keeping that high-frequency I/O on local disk avoids the
        # "OSError: error closing file" failures that can happen writing that often to
        # network-mounted storage. The checkpoints you actually care about (best/final/
        # interrupted) are saved explicitly to out_root elsewhere in this script, unaffected
        # by this setting.
        output_dir=str(scratch_root),
        # Optimiser — match PPO run defaults
        learning_rate=args.lr,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1.0e-8,
        weight_decay=1.0e-6,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
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
        log_completions=args.log_completions,
        report_to=report_to,
        save_steps=args.save_every if args.save_every > 0 else args.num_steps + 1,
        save_total_limit=args.keep_last_n,
        # Keep the 'prompt' column so it reaches the reward function.
        # TrainingArguments defaults remove_unused_columns=True, which would
        # drop 'prompt' after tokenisation — leaving reward_fn(prompt=None)
        # and crashing the zip inside it.
        remove_unused_columns=False,
    )

    best_reward_cb = BestRewardCheckpointer(
        out_root=out_root,
        tokenizer=tokenizer,
        save_best_after=args.save_best_after,
        save_best_min_interval=args.save_best_min_interval,
    )

    trainer_cls = GRPOTrainer if args.kl_guard_disable else GuardedGRPOTrainer
    trainer_kwargs = dict(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[best_reward_cb],
    )
    if trainer_cls is GuardedGRPOTrainer:
        trainer_kwargs["kl_explosion_floor"] = args.kl_guard_floor
        trainer_kwargs["kl_explosion_multiple"] = args.kl_guard_multiple
    trainer = trainer_cls(**trainer_kwargs)

    print("=" * 50)
    print("policy model device :", next(model.parameters()).device)
    print("policy model dtype  :", next(model.parameters()).dtype)
    print("reward pipe device  :", reward_pipeline.device)
    print("checkpoints (best/final) → :", out_root)
    print("TRL internal I/O (completions/periodic ckpts) → :", scratch_root)
    print("num_generations (G) :", args.num_generations)
    print("effective batch     :", args.batch_size * args.num_generations, "completions/step")
    print("warmup_steps        :", args.warmup_steps)
    print("max_grad_norm       :", args.max_grad_norm)
    print("reward_clip         :", args.reward_clip if args.reward_clip > 0 else "disabled")
    print("log_completions     :", args.log_completions,
          f"(console printing {'on' if args.print_completions_console else 'off — W&B table only'})"
          if args.log_completions else "")
    print("save_best_min_interval:", args.save_best_min_interval)
    if args.kl_guard_disable:
        print("kl_guard            : disabled")
    else:
        print(f"kl_guard            : floor={args.kl_guard_floor}, multiple={args.kl_guard_multiple}x rolling median")
    print("=" * 50)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[interrupt] Waiting for any in-flight background checkpoint save to finish...")
        best_reward_cb.join(timeout=300)
        print("[interrupt] Saving emergency checkpoint before exit...")
        trainer.save_model(str(out_root / "interrupted"))
        tokenizer.save_pretrained(str(out_root / "interrupted"))
        raise

    print("[ckpt] waiting for any in-flight background best-checkpoint save to finish...")
    best_reward_cb.join(timeout=300)

    print("[ckpt] saving final checkpoint")
    trainer.save_model(str(out_root / "final"))
    tokenizer.save_pretrained(str(out_root / "final"))

    if isinstance(trainer, GuardedGRPOTrainer):
        print(f"[kl-guard] total microbatches skipped: {trainer.num_skipped_steps}")


if __name__ == "__main__":
    main()