# GRPO Training Run — Analysis & Fixes Log

**Model:** gpt2-large (base: `tsaxena/gpt2-large-prompt-tags`), last 2 transformer blocks + `lm_head` unfrozen
**Reward model:** `toloka/prompts_reward_model`
**Framework:** TRL `GRPOTrainer` / `GRPOConfig`
**Script:** `run_grpo.py`

This document summarizes everything found and fixed across the diagnostic process for this GRPO
run, from the initial training instability through a completed 10,000-step run, plus the changes
queued up for the next run.

---

## 1. Initial problem: catastrophic loss/gradient spikes

The first runs (`beta=0.05`, default TRL settings) showed wildly unstable training:

- `grad_norm` spiking as high as **410,000+**
- `loss` spiking into the thousands
- `kl` occasionally spiking to **~62,000** in a single step

### Root cause

Confirmed directly from TRL's `GRPOTrainer.compute_loss` source: the KL estimator (Schulman's
k3 estimator) is computed per-token as

```
per_token_kl = exp(ref_logp - policy_logp) - (ref_logp - policy_logp) - 1
```

Because this sits inside an `exp()`, a single token where the current policy's probability has
collapsed relative to the reference model (e.g. a near-immediate EOS, or a bf16 precision
artifact on a low-probability token) is enough to send the **batch-mean** KL — and therefore the
loss and gradient — into the thousands or tens of thousands. One such step can corrupt the policy
irrecoverably.

### Isolation test

Running with **`beta=0`** (no KL penalty at all) produced clean, bounded `grad_norm` (single-to-low
double digits) for the same number of steps — confirming the KL term specifically, not the reward
path, LR, or clipping mechanics, was the source of instability.

---

## 2. Fixes implemented

All of the following are in `run_grpo.py`, with CLI flags and sensible defaults.

### Training stability
- **`--warmup_steps`** (default 50): linear LR ramp instead of hitting full LR on step 1, when
  advantage estimates are noisiest.
- **`--max_grad_norm`** (default 1.0): explicit gradient clipping, previously relying on an
  implicit default.
- **`--reward_clip`** (default 5.0): clips raw reward-model logits before the group-relative
  advantage calculation, so one outlier in a group of `num_generations` can't dominate the
  advantage estimate for the whole group.

### KL-explosion guard (`GuardedGRPOTrainer`)
A `GRPOTrainer` subclass that:
- Tracks a rolling median of "normal" (non-exploded) per-step KL.
- If a step's KL exceeds `max(kl_guard_floor, kl_guard_multiple × rolling_median)`
  (defaults: floor **2000**, multiple **15×**), the step's loss is multiplied by `0.0` before
  return — `backward()` still runs (so gradient accumulation isn't disrupted) but the exploded
  microbatch contributes **exactly zero gradient**, a no-op instead of a destructive update.
- Logs the true (unzeroed) values under separate keys so explosions stay visible instead of being
  silently hidden: `kl_guard/triggered`, `kl_guard/raw_loss`, `kl_guard/raw_kl`,
  `kl_guard/raw_grad_norm`.
- `raw_grad_norm` is computed via `torch.autograd.grad` (functional), **not** `.backward()` —
  verified by direct test that this cannot leak the explosive gradient back into the real
  (zeroed) update.
- This is a safety net, not a root-cause fix. If it fires frequently, that's a signal to
  investigate further (e.g. fp32 log-prob computation), not to rely on indefinitely.

### Checkpointing
- **Async saves**: `BestRewardCheckpointer` now only does the fast part (GPU→CPU weight copy)
  synchronously; the actual disk write happens on a background thread, so training doesn't stall
  for the duration of a slow/network-disk write.
- **`--save_best_min_interval`** (default 100 steps): throttles back-to-back saves during noisy
  early-training reward spikes.
- Lock-guarded, non-blocking: a save already in flight causes the next new-best to skip rather
  than queue.
- Background threads are joined before process exit (normal and `KeyboardInterrupt`) so a
  checkpoint can't be killed mid-write.
- **bf16 model loading** (`torch_dtype=torch.bfloat16`): roughly halves both GPU memory and
  checkpoint size/write time. Trade-off: trainable params now update in native bf16 rather than
  fp32-with-autocast — common for partial fine-tunes, but a real precision trade-off worth
  watching on long runs.
- Tied-weight correctness (GPT-2's `wte`/`lm_head` tying) verified via a full save→reload test
  using a real `AutoModelForCausalLM.from_config` shell + `load_state_dict`, rather than a
  hand-rolled safetensors writer that could silently break tying.

### Infrastructure — two separate crash/hang root causes found and fixed
1. **`OSError: error closing file`** (pyarrow, mid-run crash): TRL writes a completions parquet
   file to `{output_dir}/completions/` on *every* `log()` call — with `logging_steps=1`, that's a
   write every ~2.5s to network-mounted storage. **Fix:** `--local_scratch_dir` (default
   `/tmp/grpo_scratch`) — `GRPOConfig.output_dir` now points at local scratch disk, separate from
   `--output_path` (used only for the checkpoints we actually keep: best/final/interrupted).
2. **Multi-hour silent hang**, no crash, no progress: `wandb.init()` defaults its local run
   directory to the current working directory (same network mount). wandb stages completions
   Table media as temp files under `/tmp`, then `shutil.move()`s them into its run dir — a
   cross-device move that falls back to copy, which itself failed/hung against the network mount.
   **Fix:** `wandb.init(..., dir=str(scratch_root))` — wandb's local files now live on the same
   local disk as the source temp files.
3. Root cause of both: distinguish TRL/wandb's own high-frequency internal I/O (→ local scratch)
   from the checkpoints you actually want kept (→ real network output path).

### Visibility
- **`--log_completions`**: logs actual prompt/completion/reward samples to a W&B table —
  essential for catching reward hacking or degenerate completions that scalar metrics alone
  don't show.
- Console spam suppressed by default (TRL's Rich table print, which would otherwise fire every
  step): `--print_completions_console` to re-enable.

---

## 3. Full 10,000-step run — results

### Reward: real but decelerating improvement
Confirmed via four independent completions samples (aggregate scalar `reward` curve was too
noisy to show any trend by eye):

| step | 335 | 2353 | 5402 | ~10,000 |
|---|---|---|---|---|
| mean reward | -0.357 | -0.158 | -0.102 | -0.091 |

Clear early learning (335→2353: +0.20) that largely plateaus in the back half of the run
(5402→10k: +0.01). Consistent with the KL penalty (`beta=0.05`) increasingly counterbalancing
further reward-driven movement. Completion length *decreased* slightly across the same span
(264 → 158-220 chars) — no sign of reward hacking via padding.

### One severe KL explosion, caught cleanly
Around step ~2500: `raw_kl ≈ 1.2×10⁸`, `raw_grad_norm ≈ 1×10⁹` — roughly 1,000-4,000× larger than
any other explosion in the run. The guard zeroed it correctly; training continued with no
lasting damage. Guard fired at a fairly steady ~4-5% rate throughout, no clear worsening trend
over the run.

### Reward-hacking check: mostly clean, one soft pattern
- No length-padding exploit.
- Top completions were genuinely relevant/coherent to their prompts, not generic filler.
- Soft pattern noted: some (not all) high-reward completions repeated quality buzzwords
  ("highly detailed" ×2-4) — present but not a reliable predictor of reward on its own, so
  treated as a minor watch-item rather than a confirmed exploit.
- **Confirmed reward-function gap:** one empty-string completion scored **+0.114** — the reward
  model is out-of-distribution on degenerate/empty input and shouldn't be trusted there.

### Completion length: persistent bimodal pattern
`min_length` repeatedly hit 1 token; `max_length` frequently pinned at the 80-token cap, for the
full run — never converged to one consistent stopping behavior. Direct inspection of the longest
(clipped) completions at step ~10k showed genuine degeneration into repetition (e.g. "ghost ghost
with stylized skeleton"); reward correctly penalized these (all in the -0.19 to -0.65 range), so
this wasn't being exploited, but it's a real, recurring failure mode worth addressing directly.

### Unresolved: `step_time` drift
Roughly doubled over the run (~2.5s → 5-6s/step), separate from the four isolated
checkpoint-save spikes. GPU memory was confirmed **not** the cause (stable at ~8.8GB of 80GB
throughout). Root cause not fully confirmed mid-run, but a strong hypothesis emerged after the
run finished: TRL writes **one new completions parquet file per step** (confirmed directly from
the uploaded filenames — `completions_00001.parquet` through `completions_10000.parquet`),
accumulating 10,000 files in a single local directory — a known pattern for causing gradual
per-file-operation slowdown on many filesystems.

---

## 4. Fixes queued for the next run

Three findings, all addressed in `run_grpo.py`, each individually unit-tested against real
completion samples from this run before shipping:

1. **Empty-completion reward override** — `--empty_completion_penalty` (default -1.0). Forces a
   fixed worst-case reward for empty/whitespace-only completions instead of trusting the reward
   model on out-of-distribution input.
2. **Repetition penalty** — `--repetition_penalty_weight` (default 0.15). Penalizes reward in
   proportion to the fraction of repeated word-trigrams. Verified against real samples: ~0 on
   clean completions, 0.086 on the mildly-repetitive real sample, 0.75 on a deliberately
   degenerate test case.
3. **Clipped-completion penalty** — `--clipped_completion_penalty` (default 0.1). Flat penalty
   when a completion's retokenized length hits `max_new_tokens`, nudging the policy toward
   wrapping up within budget rather than rambling to the cap.
4. **Completions-directory pruning** — new `CompletionsPruner` callback,
   `--completions_keep_last_n` (default 20). Best-evidence fix for the step_time drift: keeps
   only the most recent N local completions parquet files (full history is unaffected — it's
   already in the W&B table, which doesn't share this local-directory-growth problem).

All new flags default to the recommended values; no extra flags are required to get these fixes
on the next run.

---

## 5. Key metrics to monitor (reference)

| Category | Metrics | What to watch |
|---|---|---|
| Reward (the objective) | `reward`, `rewards/reward_fn/mean` | Trend over hundreds of steps, not step-to-step — the aggregate curve can hide real movement (as it did here). Cross-check with sampled completions periodically. |
| | `reward_std` / `rewards/reward_fn/std` | Spread within each generation group; too low = nothing to learn from. |
| | `frac_reward_zero_std` | Fraction of groups with zero learning signal. |
| KL / loss | `kl` | Normal: single-to-low-double digits. Hundreds-to-thousands = explosion. |
| | `loss` | `policy_loss + beta * kl` — a spike is often just KL passing through; check both together. |
| Gradient stability | `grad_norm` | Should stay bounded after clipping; early warning of instability. |
| | `clip_ratio/*` | Structurally ~0 when `num_iterations=1` (sampling policy == updated policy) — not a bug. |
| Generation health | `completions/clipped_ratio`, `min/max_length` | High clipped_ratio or persistent bimodal lengths = policy not converging on stopping behavior. |
| | `entropy` | Climbing with no KL anchor → drifting; collapsing → possible mode collapse. |
| Guard (custom) | `kl_guard/triggered`, `raw_kl`, `raw_loss`, `raw_grad_norm` | Rate and magnitude trends — rising = underlying instability worsening, not just bad luck. |

**Golden rule from this session:** never read one metric in isolation. Loss spikes are only
meaningful alongside `kl`; a flat reward curve is only meaningful alongside an actual completions
sample — aggregate noise repeatedly hid real signal that direct sampling caught.
