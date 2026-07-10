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

**Update:** these three fixes were later suspected of causing a training-quality regression in
the next run — see Section 5 for the full investigation and its resolution. Short version: they
didn't. Keep them; the empty-completion override is worth revisiting for a different reason
(mechanism, not outcome) — also covered in Section 6.

---

## 5. The v3 "regression" investigation — and why it wasn't one

After the fixes in Section 4 shipped, the next run (**v3**) showed a flat, non-improving reward
trend where the original run (**v2**) had shown steady improvement. This section documents the
investigation into why, and its eventual resolution — which turned out to be a measurement
artifact, not a real regression.

### The apparent problem
- v3's smoothed `train/reward` curve stayed flat around -0.22 to -0.25 across 10,000 steps, where
  v2's had climbed from about -0.25 to -0.09 over the same range.
- v3 also showed more severe (though not more frequent) KL-guard explosions early on: 4 events
  above 500K raw_kl in its first 1,245 steps, including two above 5M, versus v2's roughly one
  severe outlier across its entire run.

### Ruling out hypotheses, one at a time
1. **Beta.** Ruled out immediately — identical (`0.05`) across every run compared.
2. **The three new reward-shaping penalties** (repetition, clipped-completion, empty-completion).
   Tested via an ablation run (**v4**, ` --repetition_penalty_weight 0 --clipped_completion_penalty 0`,
   empty-completion penalty still active) — still plateaued, statistically indistinguishable from
   v3 at matched steps. Partial evidence against the penalties, but incomplete: the
   empty-completion override couldn't be disabled by a flag yet, so it remained a live variable.
3. **KL-explosion severity.** Checked directly via exported `kl_guard/raw_kl` data for the
   ablation run: only 1 severe event (>500K) across 3,089 steps — calm, like v2, not like v3.
   Ruled out as the differentiator, since a "calm" run still plateaued.
4. **The empty-completion override specifically.** Re-examined more carefully: it isn't just a
   scoring correction — it participates in the **group-relative advantage calculation**. Forcing
   one group member's reward to a fixed value shifts that group's mean/std, and therefore the
   advantage (and gradient) for every *other* completion in the same group. Added
   `--disable_empty_completion_penalty` to allow a true ablation of this specific mechanism.
5. **Full ablation (v5)**: all three penalties disabled/bypassed, otherwise identical to v3/v4.
   v5's smoothed training curve visually tracked v2's closely across all 10,000 steps, and its
   `best_reward` kept improving throughout the run (reaching 0.289, versus v3's stalled 0.193 and
   v4's stalled 0.218) — strong circumstantial evidence the empty-completion override had been
   the cause.

### The confound that undid all of the above
Every completions comparison for v2 (the step 335/2353/5402/10000 samples in Section 3) used
whatever prompts happened to appear in that step's live training batch — different prompts each
time. Every comparison for v3/v4/v5 used a new tool, **`eval_checkpoint.py`**, built specifically
to eliminate that confound: a fixed, hardcoded set of 8 prompts, generated and scored identically
regardless of checkpoint. This made v3/v4/v5 directly comparable *to each other* — but **v2 was
never once evaluated with this tool**. Every "v5 vs v2" comparison up to this point was actually
comparing two different measurement methods on two different prompt sets — precisely the
confound `eval_checkpoint.py` existed to remove, reintroduced by omission.

### Resolution
v2's final checkpoint (uploaded to `tsaxena/gpt2-large-grpo-prompt-writing`) was evaluated with
`eval_checkpoint.py` on the same fixed 8 prompts used throughout:

| checkpoint | mean reward (fixed 8-prompt eval) |
|---|---|
| **v2 final (true baseline, evaluated properly for the first time)** | **-0.206** |
| v3 @ step 1898 | -0.174 |
| v3 @ step 6000 | -0.170 |
| v4 (ablation) @ step 4000 | -0.183 |
| v5 (full ablation) @ step 7177 | -0.164 |

**v2's real score on this prompt set is statistically indistinguishable from v3, v4, and v5** —
all five numbers fall within noise of each other (std ~0.31-0.40 on n=64 each), and v5 is
actually marginally *better* than v2's own baseline. **There was no real performance gap to
explain.** v2's apparent climb to -0.09 was real for the specific prompts in its own live
training batches, but never generalized to this fixed prompt set — the entire multi-run
investigation was chasing a difference that was an artifact of inconsistent measurement
methodology, not a property of any training run.

### What still stands, and what to take from this
- The three reward-shaping penalties (Section 4) are **cleared** — no evidence they hurt training
  quality. Keep them.
- The empty-completion override's *mechanism* (distorting group-relative advantage for other
  completions in the group) was still real and worth fixing on principle, even though it turned
  out not to be causing the effect it was originally blamed for. See Section 6 — it was fixed,
  and the fix turned out to help more than expected.
- **Lesson for future comparisons:** pin down identical measurement methodology *before* comparing
  runs, with the same rigor applied to the training config. A fixed-prompt eval tool
  (`eval_checkpoint.py`) is only a valid comparison baseline if every run being compared —
  including the original "reference" run — is evaluated with it. Retrofitting it onto new runs
  while leaving the baseline measured a different way silently reintroduces the exact confound
  the tool was built to eliminate.

---

## 6. Fixing the empty-completion mechanism — a real improvement, found by accident

With the plateau mystery resolved (Section 5), the empty-completion override's group-distortion
side effect was revisited on its own merits — not because it was expected to move the needle, but
because it was a known-real mechanism-level flaw worth fixing regardless.

### The fix: group-median substitution
New flag `--empty_completion_median`. Instead of forcing empty/whitespace-only completions to a
fixed `--empty_completion_penalty` (default -1.0) — which injects an extreme value into the
group's mean/std and therefore distorts the advantage of every *other* completion sharing that
prompt — this substitutes the group's own median reward (computed from its non-empty members).
Empty output is still discouraged relative to its peers, without poisoning their advantage signal.
Falls back to the hard override only if an entire group is empty (no median available to take).

Implemented as a standalone, unit-tested helper (`_substitute_empty_with_group_median`), verified
against: a normal mixed group (empty correctly replaced with the median of its non-empty peers),
a group with no empties (fully untouched), an all-empty group (correctly falls through to the
hard-penalty fallback), and confirmed not to mutate its input list in place.

### Result: a real, confirmed improvement — not just a neutral fix
Evaluated with `eval_checkpoint.py` (same fixed 8-prompt set used throughout) at a step count
(7,177) matched exactly against the best prior configuration:

| configuration | step | `best_reward` | fixed-prompt eval (mean) |
|---|---|---|---|
| v2 (original, hard override, true baseline) | — | — | **-0.206** |
| v5 (empty-completion protection disabled entirely) | 7,177 | 0.289 | -0.164 |
| v6 (hard override off, rep+clip on) | 4,000 | 0.247* | **-0.159** |
| **v7 (group-median substitution, rep+clip on)** | **7,177** | **0.302** | **-0.119** |

\* v6's `best_reward` peak is not directly comparable to the others — see the note below on why
`best_reward` misled the investigation mid-stream.

**v7 is the best configuration found across the entire investigation on the one measurement
method proven reliable in this project.** The median-substitution fix wasn't just safer than the
hard override (as hoped) — it measurably improved policy quality, at the same step count as the
next-best comparison (v5). Final recommended config going forward: `beta=0.05`,
`--repetition_penalty_weight 0.15`, `--clipped_completion_penalty 0.1`,
`--empty_completion_median` (in place of the old hard override).

### A cautionary footnote on `best_reward`
Mid-investigation, v6's `best_reward` appeared stuck at a low peak (0.247 at step 724, never
improving afterward) while its fixed-prompt eval was actually the best result seen up to that
point (-0.159). The explanation: `best_reward` tracks a *maximum* over noisy per-batch rewards,
and repetition/clipped penalties fire somewhat randomly per-completion — a batch that happened to
dodge them early can set a ceiling that's statistically hard for *any* later, larger, more
representative batch to beat, independent of whether the underlying policy is actually improving.
`best_reward` is a fragile signal under these penalties; the fixed-prompt eval mean (a stable
64-sample average, immune to this ceiling effect) is the metric to trust when the two disagree.

---

## 7. Key metrics to monitor (reference)

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
