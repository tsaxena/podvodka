# RLHF on GPT-2-Large for Prompt-Writing — Project Summary

> **Where things stand:** SFT base ships. PPO and DPO experiments both produced models *worse than the base* — different failure modes, same root cause. The next experiment (DPO with a reasoning-model judge) is the test of whether the problem can be fixed without abandoning RLHF on this RM.

## TL;DR

The project tried two RLHF algorithms (PPO, then DPO) on the same SFT model with the same reward model, on a stable-diffusion prompt-writing task. Both algorithms achieved their training objectives — both got their reward metrics to climb — and both produced policies that are qualitatively worse than the SFT base they started from. The root cause is the reward model, not the algorithms.

| | SFT base | PPO | DPO (RM-labeled) | DPO (reasoning judge) |
|---|---|---|---|---|
| RM score | −0.17 | +0.55 | +0.48 | TBD |
| Layer 3 verdict | Best | Reward hacking (severe) | Reward hacking (mild) | Pending |
| Recommended for production | ✅ | ❌ | ❌ | TBD |

**The single sentence:** *both PPO and DPO faithfully optimized a gameable reward signal; the algorithms differ in how they fail, but neither can produce a better policy than the SFT base when the labeler is broken.*

---

## The Big Picture

**Goal:** improve a GPT-2-large model that expands short concepts into stable-diffusion-style prompts (e.g., "Moscow street" → "Moscow street, autumn, style of Konstantin Korovin, dawn light"). The SFT base (`tsaxena/gpt2-large-prompt-tags`) does this competently. The hypothesis: PPO/DPO against a reward model should improve it further.

**Reality:** the only available reward model (`toloka/prompts_reward_model`) is gameable. Any algorithm that optimizes against it produces magic-word stuffing instead of better prompts. The project became a study in *how* different algorithms fail under the same flawed signal.

**Outcome of the project so far:**
- A clearly-better understanding of how PPO and DPO differ in practice on real (flawed) data
- Three trained models (PPO, DPO-RM, all confirmed broken via Layer 3) and the SFT base (recommended for use)
- A reusable comparison and sanity-check pipeline (`compare_checkpoints.py`, `rm_sanity.py`, etc.)
- One open thread: DPO with a reasoning model as judge, which has a real chance of beating the SFT base if the judge prompt accurately captures what "good" means

---

## Story Part 1: PPO

### What we did

Pinned `trl==0.11.4` for the classic `PPOTrainer`/`PPOConfig` API. Loaded the SFT model as both policy and reference (via `AutoModelForCausalLMWithValueHead`). Froze 34 of 36 transformer blocks, kept the top 2 + value head trainable. Used a frozen `toloka/prompts_reward_model` pipeline for reward scoring. Trained on an A100 80GB.

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 1.4e-5 |
| `batch_size` (rollouts) | 128 |
| `mini_batch_size` | 128 |
| `ppo_epochs` | 4 |
| `init_kl_coef` | 0.05 |
| `cliprange` | 0.2 |
| `num_layers_unfrozen` | 2 |
| Steps trained | ~330 (plateau at ~200) |

### The environment fight

Roughly 60% of the PPO phase was spent on environment compatibility — a long catalog of version conflicts:

| Problem | Fix |
|---|---|
| `trl` API rewrote between versions | Pin `trl==0.11.4` |
| `cliprange_reward` is from trlX, not trl | Remove |
| torch ↔ torchvision mismatch (`nms` op missing) | Align both to torch 2.4.1 |
| CUDA wheels not on PyPI | `--index-url https://download.pytorch.org/whl/cu124` |
| `transformers` too new for torch 2.4 | Pin `transformers==4.45.2` |
| `datasets` ↔ `fsspec` upper bound | Loosen fsspec |
| Container root disk only 20GB on RunPod | Redirect `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR` to `/workspace` |
| Training silently on CPU despite GPU available | `accelerator_kwargs={"cpu": False}` in `PPOConfig` |
| 50s/step (reward scored one sample at a time) | Batch `reward_pipeline` calls |
| Default `trl.generate(batch_size=4)` | Bump to 32 |
| Pod resets restored older Python images | `pip freeze > requirements.lock.txt` |

The accelerate-CPU-silent issue was the biggest single waste of time. A device assertion right after `PPOTrainer` is now a permanent part of the script.

### What the metrics showed

PPO trained without drama. The W&B metrics all looked good:

- `env/reward_mean`: −0.27 → +0.55, plateau around step 200
- `env/reward_std`: 0.38 → 0.28 (mild decline, considered healthy)
- `ppo/loss/value`: 0.15 → 0.005 (smooth, critic converged)
- `ppo/policy/clipfrac`: ~0.001 (clip never bound, very conservative updates)
- `ppo/policy/approxkl`: ~0.0002 (tiny per-step KL)

At the time, all of this was read as "PPO worked, run successful." The verdict was wrong.

### What Layer 3 revealed

After the run, generated 3 samples per prompt for 30 prompts from base and PPO, scored with the same RM, and **read the top-scoring PPO outputs**. The pattern was unmistakable — "art by artgerm and greg rutkowski" verbatim in 4 of 5 top outputs, regardless of prompt context:

| Prompt | PPO output (top-scoring) |
|---|---|
| Valley in lauterbrunnen | "atmospheric, **greg rutkowski**, fantastic art, mystical, fantasy, intricate, highly detailed, digital painting, **artstation**, **octane render**, **8k**, **wlop**, **artgerm**" |
| Moscow street | "surrealistic, hyper detailed, digital art, depth of field, **unreal engine 5**, **hd**, **8k**, **art by artgerm and greg rutkowski**" |

Compare to base SFT, which picked `style of konstantin korovin` (a real Russian Impressionist) for the Moscow prompt — a prompt-specific stylistic choice. PPO learned to slap the same magic-word stack onto everything, because the RM consistently scored those phrases high. Reward went up; quality went down.

### Why this happened mechanically

Three things combined:
- **The RM has a learned correlation** between SD magic words and "quality" (because its training data had them in high-quality examples).
- **PPO's KL anchor was weak** (`init_kl_coef=0.05`). The policy could drift far if reward gain was worth it. It was.
- **Conservative updates aren't a defense.** `clipfrac=0.001` meant each update was tiny, but tiny updates toward a misaligned objective still converge to that objective — they just take longer.

The conservative-PPO regime bought stability but did not prevent the hacking. It just made it slower and smoother to discover.

---

## Story Part 2: DPO with RM-labeled data

### Motivation

Given that PPO directly optimizes against the (gameable) RM on every step, the natural next experiment is an algorithm that doesn't have the RM in the inner loop. DPO trains on a fixed preference dataset and never queries the RM during training — but the dataset still has to be labeled somehow, and we used the same RM to do it. Question: does that change anything?

Theoretical reasons it might help:
- **Stronger reference anchor.** DPO's loss is a log-ratio against the reference policy. Large coordinated changes (like "always append 80 tokens of magic words") get penalized structurally.
- **Bounded optimization.** Once the policy can correctly discriminate chosen from rejected, the loss gradient vanishes. PPO has no such ceiling.

Theoretical reasons it might not:
- **The dataset still encodes the RM's bias.** If "chosen" examples all contain magic words (because that's what the RM scored highest), DPO learns to produce magic words.

### What we did

Built a preference dataset from the SFT model: 1000 prompts × 8 candidates each, scored with the RM, paired the best/worst with a `min_score_gap=0.3` filter. Got ~970 pairs (875 train, 97 eval). Trained `DPOTrainer` from `trl==0.11.4`.

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 5e-6 |
| `batch_size` (effective) | 16 |
| `epochs` | 2 |
| `β` | 0.1 |
| Steps trained | 108 |
| Wall-clock | **~70 seconds** |

Compare to PPO's hours. DPO is dramatically faster (no rollouts, no reward scoring during training).

### What the metrics showed

All healthy:
- `train/rewards/chosen`: −0.71 → +0.5 (rising, good)
- `train/rewards/rejected`: −4.35 → −6.5 (falling, good)
- `train/rewards/margins`: 3.6 → 7.0 (widening, good)
- `train/rewards/accuracies`: 67.5% → 95%
- `eval/rewards/accuracies`: 89.4%

One yellow flag: `grad_norm` spiked to 549 in step 2 (normal is 1–10). Fix for next time: `max_grad_norm=1.0`, `warmup_ratio=0.2`. Run survived it.

### What Layer 3 revealed

Same comparison script, this time with three checkpoints. Numeric summary:

| Metric | base | PPO | **DPO** |
|---|---|---|---|
| `rm_mean` | −0.17 | +0.55 | **+0.48** |
| `rm_std_overall` | 0.33 | 0.25 | **0.30** |
| `rm_std_across_prompts` | 0.18 | 0.17 | **0.17** |
| `frac_above_+0.5` | 3.3% | 63.3% | **46.7%** |
| `frac_at_ceiling_+0.9` | 0.0% | 5.6% | **7.8%** |

DPO captured ~93% of PPO's RM-score gain, with less std compression. Metric-level, looked mildly better than PPO.

But Layer 3 reading of top-scoring DPO outputs revealed a **different** failure mode:

| Prompt | DPO output (top-scoring) |
|---|---|
| Tall glass tower at dusk | "= an extremely detailed photo of a tall glass tower, viewed from the ground at dusk, hight resolution, 8k photography, artstation, realistic" |
| Valley in lauterbrunnen | "= lucid breathtaking valley in lauterbrunnen image, ultrarealistic, cinematic lighting, future tech" |

Key observations:
- **No "art by artgerm and greg rutkowski"** in any DPO output. DPO's reference anchor did partially work — the worst PPO failure mode was dodged.
- **DPO has its own milder pattern**: restate the prompt + 3-4 generic modifiers + short overall length. Less verbose, less artist-stuffed, but also less specific.
- **Base SFT still picks better stylistic references** than either RLHF model.
- **"Future tech" for a Swiss alpine valley** is the kind of bad-context modifier choice that DPO falls into — generic adjective stacks that don't fit the prompt.

### Why DPO failed differently from PPO

The mechanism is interesting and worth understanding:

PPO's KL penalty (`init_kl_coef=0.05`) is a *soft* suggestion. The policy can drift arbitrarily far if the reward gain is large enough. Over 200 steps, it found the global attractor: stacking the most-rewarded magic words on every output.

DPO's reference-ratio loss is a *structural* constraint. Every token's log-ratio enters the loss. Large coordinated changes (like always appending the same 80-token stack) are expensive to make — the policy would have to restructure its output distribution, paying loss on many other completions. So DPO settled on small, low-cost changes: prepend "= ", restate the prompt, append a handful of modifiers.

**Same RM bias, different optimization path, different end-state.** PPO went to the extreme; DPO took a partial step in the same direction.

---

## The Pattern That Emerged

Three observations converge to one conclusion:

1. **PPO maximally exploited the RM bias.** Long magic-word stacks on every output.
2. **DPO partially exploited the RM bias.** Shorter outputs, fewer modifiers, no signature artists, but generic adjectives that don't fit.
3. **The base SFT (no RLHF)** picks varied, prompt-specific stylistic references and is qualitatively best.

The reward model is the upstream cause of both failures. Any algorithm that uses it as the source of preference signal will inherit some version of its bias. The algorithm only changes the *path* and *severity* of the exploitation, not whether it happens.

**This is the central finding of the project.** It's not a story about PPO being bad or DPO being better. It's a story about the labeler being the bottleneck, and the algorithm choice being downstream of that.

---

## Story Part 3: DPO with a reasoning-model judge (pending)

### The proposed experiment

Replace the RM as labeler. Generate candidates from the SFT model, then ask a strong reasoning model (DeepSeek R1, Claude Sonnet 4.5 with thinking, o3-mini) to score each candidate using a judge prompt that **explicitly penalizes the failure modes we observed**:

- Generic "quality" tokens (8k, octane render, unreal engine, trending on artstation)
- Famous artist name-dropping that doesn't fit the subject
- Prompt restatement without genuine additions
- Verbose modifier stacks on every output regardless of context

Build the DPO dataset from these labels and train. Same algorithm, same hyperparameters, **different labels**.

### Why this is the right next experiment

The hypothesis "the dataset is the bottleneck, not the algorithm" makes a specific prediction: changing the labeler should change the result more than changing the algorithm did. PPO vs DPO with the same labeler produced two flavors of the same failure. DPO with a different labeler should either (a) produce a model that genuinely beats the SFT base, or (b) produce a model with new and different failure modes — both outcomes are informative.

### What I expect

A reasoning model can read the prompt ("Moscow street"), read the candidate ("art by artgerm and greg rutkowski"), and **notice the mismatch**. The RM can't notice anything — it just maps text to a scalar. So the judge should give low scores to the obvious failures, and DPO trained on those labels should avoid them.

Realistic prediction: a model meaningfully better than DPO-RM, plausibly competitive with the SFT base, with some new and (probably) milder bias I haven't yet anticipated. The next Layer 3 read will reveal what the new bias is.

### Why this isn't a clean fix either

- **The judge has its own biases.** Reasoning models trained on internet text have their own ideas about what makes a "good" SD prompt. Those overlap with the RM's biases but not perfectly.
- **The judge prompt only catches anticipated failures.** New failure modes you didn't think to ban will still get exploited.
- **DPO still amplifies whatever bias is in the labels** — the dataset is always the bottleneck, just a less broken one with a better labeler.

The framing for this next run: it's a test of whether the bottleneck can be moved, not whether it can be removed.

### Scripts ready to run

- `build_dpo_dataset_openrouter.py` — uses OpenRouter, defaults to a free model. Throttled for free-tier limits.
- The same `train_dpo.py` script consumes the resulting JSONL with no changes needed.
- `compare_checkpoints.py` will run a four-way comparison once the new model exists.

---

## Key Insights from the Whole Project

### 1. The reward curve told the truth about optimization; only reading generations told the truth about quality.

Originally I wrote "the reward curve is the only metric that matters." That was almost right and dangerously wrong. The reward curve answers "is the policy optimizing the RM?" — which is not the same question as "is the policy doing the task well." When the RM is imperfect (always), those are different things. Layer 3 reading is the only way to find out which one you got.

### 2. Loss curves in PPO are misleading in three different ways at once.

- **Policy loss is negative when training succeeds.** trl reports `-L^CLIP` so the optimizer can minimize it. Negative = good. People panic at negative losses and shouldn't.
- **Total loss is dominated by value loss** when `vf_coef=1`. The total going down beautifully (0.15 → 0.005) just means the critic learned to predict returns. It says nothing about policy quality.
- **Loss can fall while the policy gets worse.** Reward-hacking failure mode: policy games the RM, reward rises, loss falls, text quality collapses. Loss never warns.

### 3. `std_across_prompts` is not a complete mode-collapse detector.

Specifically, it fails to catch *head/tail mode collapse*: when outputs have a prompt-specific component and a generic component, and only the generic component collapses. Standard variance metrics average over both and stay roughly flat.

In both PPO and DPO runs, `rm_std_across_prompts` was nearly identical to base (~0.17 across all three checkpoints). The metric "told us" no collapse happened — but Layer 3 reading revealed that yes, the modifier tails *had* collapsed onto a small set of high-RM-score patterns. The prompt-specific descriptive prefix kept the variance metric flat.

Better detection ideas:
- **N-gram overlap across batch** (cheap, catches this directly)
- **Position-wise variance** (measure variance at end of generation separately)
- **Just read 5 samples every 50 steps** (the cheapest and most decisive)

### 4. PPO converges much faster than the defaults suggest.

The original `num_steps=10000` was a magic number from a tutorial. Reward plateaued around step 200; 98% of planned compute was past the plateau. Stop on the reward curve, not on a step count.

But: stopping earlier wouldn't have fixed the hacking. Magic-word discovery probably happened in the first 100 steps. The reward plateau and hacking saturation arrived together. **Fewer steps is wall-clock savings; KL anchor is the actual hacking defense.**

### 5. The conservative regime worked exactly as advertised — and still hacked.

`clipfrac ≈ 0.001`, `approxkl ≈ 0.0002`, every update was tiny. The PPO clip almost never bound. **And the policy still found the magic-word exploit.** Slow optimization toward a misaligned objective is still optimization toward a misaligned objective. Being conservative bought time but didn't prevent the underlying failure.

### 6. Algorithm choice matters more than the pre-experiment intuition.

The hypothesis going into DPO was "both PPO and DPO will reproduce the RM's bias, no real difference." That was half right: both produced worse-than-base models, but the failure modes were qualitatively different — PPO went to extremes (verbose magic-word soup), DPO took a partial step in the same direction (short generic adjective stacks). The structural difference between weak-KL-on-rewards and reference-ratio-loss is real and observable in the outputs.

But: algorithm choice didn't solve the problem. DPO is *better than PPO* but *worse than base SFT*. The bottleneck is upstream.

### 7. DPO is dramatically easier to run than PPO.

| | PPO | DPO |
|---|---|---|
| Inner-loop work | Rollouts + RM scoring + advantage + clip + update | Forward pass on (prompt, chosen, rejected), backward, update |
| Hyperparams to tune | ~15 | ~3 |
| Failure modes during training | Many | Few |
| Wall-clock | Hours | ~70 seconds |

DPO is the better engineering choice. Neither is the right choice for production with a broken RM.

### 8. Environment setup is the project.

60% of total effort was version compatibility. The matrix of compatible versions is narrow and shifts under your feet between releases. Pin everything via `requirements.lock.txt` from day one. Set `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR` to `/workspace` immediately on RunPod (container root is tiny).

### 9. The RM is the problem, regardless of algorithm.

Both experiments converge on this: `toloka/prompts_reward_model` is gameable, and its bias dominates any RLHF run that uses it. PPO found the bias fast and exploited it fully. DPO found it slowly and exploited it partially. The path forward is not "try a third RLHF algorithm" — it's "fix or replace the labeling signal."

---

## What I'd Do Differently (Cumulative)

In rough priority order, drawing from both experiments:

1. **Use a stronger labeler.** A reasoning model with a judge prompt that explicitly penalizes the failure modes we observed. This is the next experiment.
2. **Build Layer 3 into the training loop.** Every N steps, sample 5 generations, compute n-gram overlap across them and against a known magic-word list, log as a metric. Catch hacking as it emerges, not in postmortem.
3. **Stronger KL coefficient for PPO** (`init_kl_coef=0.5+`). Soft KL is too permissive when the RM has known correlations.
4. **DPO over PPO by default.** Faster, simpler, fewer failure modes during training. Even with bad data, milder failure mode.
5. **Validate RM input format before training.** The original script concatenated `p + "</s>" + r` but `p` already ends in `</s>`, producing `</s></s>` between prompt and response. Possibly off-distribution input to the RM, possibly weakened training signal.
6. **Smaller `num_steps`, more checkpoints.** Plateau in ~200 steps; deploy `best/`. Save every 50 steps so you can roll back to a pre-hacking version.
7. **Pin the environment.** `requirements.lock.txt` from day one.
8. **Diversity / length penalties as auxiliary losses.** Crude but effective. Directly target the magic-word attractor.
9. **Consider LoRA over layer-freezing.** Modern equivalent of "limit how much the policy can change." Better capacity/stability tradeoff.

---

## Current Status / What to Deploy

| Model | Status | Use |
|---|---|---|
| `tsaxena/gpt2-large-prompt-tags` (SFT) | ✅ Recommended for production | Best qualitative outputs |
| `tsaxena/gpt2-large-ppo-prompt-tags` (PPO) | ❌ Research artifact only | Magic-word reward hacking |
| DPO (RM-labeled), local checkpoint | ❌ Research artifact only | Milder magic-word problem |
| DPO (reasoning-judge), pending | 🔄 Not yet trained | Next experiment |

The HF model cards for the PPO and (eventually) DPO models should carry warnings about the reward hacking before being used downstream.

---

## Reusable Artifacts

The project produced several scripts that are worth keeping for future RLHF work:

| File | Purpose |
|---|---|
| `run.py` | PPO training with all environment/CPU/scoring fixes |
| `train_dpo.py` | DPO training from a JSONL preference file |
| `rm_sanity.py` | Pre-training reward-model sanity check |
| `rm_preflight.py` | Extended RM pre-flight (good/bad pairs, length bias, etc.) |
| `compare_checkpoints.py` | Layer 2/3 multi-checkpoint comparison |
| `build_dpo_dataset_openrouter.py` | Build preference data via OpenRouter judge |
| `upload_to_hf.py` | Upload trained checkpoint to HF Hub with model card |

The Layer 3 reading pattern (filter top-scoring outputs, read side-by-side, look for cross-prompt template repetition) is the single most valuable diagnostic technique from the project. It's the thing that caught both the PPO and DPO failures when every metric said the runs were healthy.

---

## The Story In One Paragraph

We tried PPO on GPT-2-large with a frozen reward model. After a long environment fight, training looked beautiful by every standard PPO metric. Layer 3 reading revealed severe reward hacking — the policy had learned to append "art by artgerm and greg rutkowski, octane render, 8k" to every prompt regardless of context, because the RM scored those phrases high. We switched to DPO, hoping the algorithm change would help. DPO finished training in 70 seconds (vs hours for PPO) and produced a model that dodged PPO's worst failure mode (no famous-artist stuffing) but exhibited a milder version of the same pathology (short outputs that restate the prompt plus a small set of generic modifiers). Both algorithms were doing what they were told; the reward model was the upstream problem. The base SFT model remains the best policy for actual deployment. The next experiment — DPO with a reasoning-model judge that explicitly penalizes the failure modes we observed — is the test of whether the bottleneck can be moved by changing the labeler instead of the algorithm. If it works, we have a path forward; if it doesn't, the conclusion is that this particular RM-labeled task isn't amenable to RLHF improvement over the SFT base, full stop.
