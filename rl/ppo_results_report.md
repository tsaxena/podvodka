# PPO Fine-Tuning of GPT-2-Large — Project Summary

> **Result:** The PPO run looked successful by every standard PPO metric — and was, in fact, reward hacking. The base SFT model is the better policy. This document is honest about both what looked good and what was actually wrong.

## Outcome

A PPO RLHF run on a single A100 80GB that **achieved its training objective and failed its real one**.

- **Base model**: `tsaxena/gpt2-large-prompt-tags` (SFT'd GPT-2-large for stable-diffusion-style prompt expansion)
- **Reward model**: `toloka/prompts_reward_model` (frozen; cannot retrain)
- **Training result**: mean reward improved from **−0.27 → +0.55** over ~200 PPO steps
- **Qualitative result**: PPO model learned to append the canonical SD magic-word stack (`art by artgerm and greg rutkowski`, `octane render`, `unreal engine`, `8k`, `trending on artstation`) to **every** prompt regardless of context. Base SFT model produces more varied, prompt-specific outputs.
- **Uploaded to HF** (as a research artifact, not for production): `tsaxena/gpt2-large-ppo-prompt-tags`
- **Recommended production model**: the SFT base, not the PPO output

## The Journey (Compressed)

The environment fight was longer than the training fight.

| Stage | What went wrong | Fix |
|---|---|---|
| `trl` import | Module missing | `pip install trl` |
| API mismatch | Tutorial used classic `PPOTrainer` API; recent `trl` rewrote it | Pinned `trl==0.11.4` |
| `cliprange_reward` | Not a `trl` param — that's **trlX** | Removed; would manually clip if needed |
| torch ↔ torchvision | `RuntimeError: operator torchvision::nms does not exist` (mismatched compiled ops) | Aligned both to torch 2.4.1 |
| CUDA wheels | `torch==2.4.1+cu124` not on PyPI | `--index-url https://download.pytorch.org/whl/cu124` |
| transformers too new | `torch.distributed.tensor.device_mesh` doesn't exist in torch 2.4 | Downgraded `transformers` to 4.45.2 |
| Disk full | 20 GB container root vs 266 TB `/workspace` on RunPod | Redirected `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR` |
| `datasets` ↔ `fsspec` | `datasets` pins `fsspec<=2026.2.0` | Unpinned fsspec |
| **Training silently on CPU** | A100 sat at 0% util, model never on GPU | `accelerator_kwargs={"cpu": False}` in `PPOConfig` |
| 50s/step | Reward pipeline scored one sample at a time | Batched to `score_batch()` with `batch_size=32` |
| Slow generation | `trl`'s `generate()` defaults to `batch_size=4` | Bumped to 32 |
| LR schedule no-op | `eta_min=lr` made cosine flat | Fixed to `lr * 0.1` |
| No mid-run checkpoints | Single save at end → one crash = total loss | Added `step-*`, `best/`, `final/`, `interrupted/` checkpoints |

## Final Configuration

```python
PPOConfig(
    learning_rate=1.4e-5,
    batch_size=128,                  # rollouts per PPO step
    mini_batch_size=128,
    ppo_epochs=4,
    init_kl_coef=0.05,               # too low in hindsight — see "What I'd Do Differently"
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=1.0,
    accelerator_kwargs={"cpu": False},  # CRITICAL — was silently CPU
)

# Partial fine-tuning: only 2 of 36 transformer blocks unfrozen + value head
# Generation: max_new_tokens=80, do_sample=True, top_p=1.0
# Reward scoring and generation: batched, 32 at a time
```

## Training Health Snapshot (as observed *during* training)

These metrics all looked good in W&B. **Every single one was consistent with reward hacking that was happening simultaneously.** That is the lesson.

| Metric | Reading | Apparent verdict |
|---|---|---|
| `env/reward_mean` | −0.27 → +0.55, plateau | "PPO worked" |
| `env/reward_std` | 0.38 → 0.28, modest decline | "Convergence, not collapse" |
| `env/reward_dist` | Shifted up, width preserved | "Healthy translation" |
| `ppo/loss/value` | 0.15 → 0.005, smooth | "Critic converged" |
| `ppo/loss/policy` | ~−0.004 steady | "Healthy (negative = correct)" |
| `ppo/policy/clipfrac` | ~0.001 | "Updates extremely conservative" |
| `ppo/policy/approxkl` | ~0.0002 | "Same" |

These verdicts were all wrong-in-retrospect. The metrics describe a *successful optimization*, not a *good outcome*. Those are different things.

## Reward Hacking Sanity Check (3-Layer Verification)

PPO can quietly fail by *gaming the reward model* — producing outputs that score high without being better. Loss curves can't catch this. Even rising reward can't catch it. The policy is literally optimizing for the RM, so high RM scores are exactly what reward hacking looks like.

Three layers of evidence, in increasing order of how definitive they are.

### Layer 1: Metric-pattern check (during training)

Looked for two failure-mode fingerprints in the W&B charts:

**Hacking-event pattern:** simultaneous spike in `reward_mean`, `objective/kl`, `clipfrac`; drop in `entropy` and `reward_std`. **Not observed** — the reward curve was smooth.

**Gradual-drift pattern:** slow `reward_std` compression, distribution piling against the RM ceiling, bottom of distribution disappearing. **Mild signs:** std declined ~25%, top edge of `reward_dist` sat against +1.0, but bulk of distribution was healthy-looking.

Layer 1 verdict at the time: "weak metric-level evidence of hacking, mild yellow flag, proceed to Layer 2."

### Layer 2: Quantitative checkpoint comparison

Used `compare_checkpoints.py` on 30 prompts × 3 samples each, scoring everything with the same RM.

| Metric | base | ppo | Interpretation at the time |
|---|---|---|---|
| `rm_mean` | −0.167 | +0.553 | "+0.72 gap. Real shift." |
| `rm_std_overall` | 0.330 | 0.254 | "~23% drop. Mild compression." |
| **`rm_std_across_prompts`** | **0.182** | **0.174** | **"Essentially unchanged — policy still differentiating between prompts."** |
| `rm_min` | −0.977 | −0.289 | "Bottom lifted — bad outputs eliminated." |
| `rm_max` | +0.758 | +1.108 | "Top extended past the original ceiling." |
| `frac_above_+0.5` | 3.3% | 63.3% | "Bulk of distribution moved." |
| `frac_at_ceiling_+0.9` | 0.0% | 5.6% | "Well below the 30% saturation threshold." |

Layer 2 verdict at the time: "numbers strongly suggest healthy improvement."

**This verdict was wrong**, for reasons documented in the "Key Insights" section below.

### Layer 3: Read the generations — the verdict that mattered

Filtered the output CSV to PPO generations scoring above +0.9 and read them. **Five of five top-scoring PPO outputs followed the same template**: a brief descriptive prefix, then a near-identical SD magic-word soup.

| Prompt | PPO modifier tail |
|---|---|
| valley in lauterbrunnen | `atmospheric, greg rutkowski, fantastic art, mystical, fantasy, intricate, highly detailed, digital painting, artstation, octane render, 8k, wlop, artgerm` |
| tall glass tower at dusk | `intricate, highly detailed, photorealistic, octane render, 4k, unreal engine, cinematic, concept art, 8k, art by artgerm and greg rutkowski` |
| sci-fi vietnam marines | `concept art, glowing lights, unreal engine, highly detailed, octane render, trending on artstation, unreal engine 5, masterpiece` |
| Moscow street | `surrealistic, hyper detailed, digital art, depth of field, unreal engine 5, hd, 8k, art by artgerm and greg rutkowski` |
| tribal village at sunset | `biro, cyberpunk, artstation, ultra detailed, cinematic light, unreal engine, hd, 8k, art by artgerm and greg rutkowski` |

**"art by artgerm and greg rutkowski" verbatim in 4 of 5 outputs.** A Russian street, a Swiss valley, a tribal village, and a sci-fi battlefield all get the same modifier stack.

**Compare to the base SFT model for the same prompts:**

| Prompt | Base SFT modifier choice |
|---|---|
| Moscow street | `style of konstantin korovin` (a real Russian Impressionist — *fits the prompt*) |
| tall glass tower at dusk | `style of david lazar and salvador dali, muted colors, lateral perspective, sun glare` |
| tribal village at sunset | `liminal land mine hunting machine, blue nebulae in cowboy hats` (weird, but specific) |

Base picks *prompt-relevant* artists and varies stylistically. PPO pastes the same stack onto everything.

**Layer 3 verdict: confirmed reward hacking. The base SFT model is the better policy for actual use.**

### Why Layer 2 missed it

The single number I trusted most was `rm_std_across_prompts: 0.182 → 0.174`. I reasoned: if every prompt got the same response, this would crash. It didn't crash, so the policy must still differentiate.

The error: **the policy *is* still differentiating between prompts — but only in the descriptive prefix.** "Surreal mountain setting" vs "Russian river" vs "futuristic space ship" varies per-prompt. The *modifier tail* is identical. Since the prefix dominates the per-prompt-mean variance, `std_across_prompts` stays flat even though the tail has fully collapsed.

This is a real and previously-unflagged failure mode of the metric: **head/tail mode collapse**. The metric assumes the output is one thing; in practice the output has two parts and only one part collapsed.

## Key Insights

### 1. **In PPO, the reward curve is necessary but not sufficient for "is this working."**

Originally I wrote: "the reward curve is the only metric that actually answers 'is this working?'" That was almost right and dangerously wrong. The reward curve answers "is the policy optimizing the RM?" but the question that actually matters is "is the policy doing the task well?" — and those are different when the RM is imperfect, which it always is.

The reward curve told the truth about optimization. Layer 3 told the truth about quality. **Without Layer 3, the verdict would have been "successful run" and the project would have shipped a worse policy than the base.**

### 2. **`std_across_prompts` is not a complete mode-collapse detector.**

Specifically, it fails to catch *head/tail mode collapse*: when outputs have a prompt-specific component and a generic component, and only the generic component collapses. Standard variance metrics average over both and stay roughly flat.

Better detection ideas for next time:
- **N-gram overlap across batch.** Compute the fraction of 4-grams in each generation that also appear in other generations from the same batch. If this rises sharply, generic-tail collapse is happening even if `std_across_prompts` is flat.
- **Position-wise variance.** Measure response variance at the *end* of the generation separately from the start. Tail collapse shows up specifically as low end-of-response variance.
- **Just read 5 random samples every 50 steps.** The cheapest detector and the most decisive. Should be wired into the training loop, not done as an afterthought.

### 3. **The "magic words" attractor is a real and predictable failure mode for this kind of RM.**

The RM was trained on (prompt, completion) pairs where high-quality completions in the training data tended to contain SD-community modifiers ("octane render", "greg rutkowski", etc.). The RM correctly learned that these correlate with quality. PPO then exploited that correlation by adding the modifiers to everything — including prompts where they make no sense.

This isn't a bug in the RM exactly; the RM is doing what it was trained to do. It's a bug in *how the RM is used in PPO*: any imperfect correlation in the RM's training data is something PPO will find and exploit. **The stronger PPO optimizes, the more brittle the RM's correlations become.** This is the canonical RLHF tension.

### 4. **PPO converges much faster than the defaults suggest — and that doesn't actually help here.**

The original script had `num_steps=10000`. The reward curve plateaued around step 200. 98% of the planned compute was past the plateau. But — and this is the subtle update from the original draft — **stopping earlier would not have fixed the hacking**, because the magic-word soup discovery probably happened in the first 100 steps. The reward curve plateau and the hacking saturation arrived together. The thing that would have fixed the hacking is a stronger KL penalty, not fewer steps.

### 5. **Environment setup is the project.**

Probably 60% of the work was getting torch / torchvision / torchaudio / transformers / trl / tokenizers / fsspec / datasets / CUDA / accelerate to all agree with each other. Once that worked, training was a couple of bugs (reward scoring not batched, accelerate config forcing CPU) and a lot of staring at W&B. If I were starting this kind of project today I'd freeze a known-good combo of versions immediately and never touch it.

### 6. **The conservative regime worked exactly as advertised — and still hacked.**

`clipfrac ≈ 0.001` and `approxkl ≈ 0.0002` were both ~50× below the published "sweet spot." Every update was tiny. The clipping mechanism — the entire reason PPO is called *Proximal* — almost never had anything to clip.

And the policy still found the magic-word exploit. **Slow optimization toward a misaligned objective is still optimization toward a misaligned objective.** Being conservative bought stability and bought time, but it didn't prevent the underlying failure. Increasing `init_kl_coef` would have helped (constrains *where* the policy can move); making each update smaller doesn't (just makes the journey to the same destination longer).

## What I'd Do Differently

In rough priority order, updated post-Layer-3:

1. **`init_kl_coef=0.5` or higher.** The single most impactful change. 0.05 was a typical PPO tutorial value, but for an RM with known magic-word correlations it was too permissive. A short leash forces the policy to stay close to base, which directly limits how much it can lean on the magic words.
2. **Build Layer 3 into the training loop.** Every N steps, sample 5 generations, compute n-gram overlap across them, and log it as a metric. If overlap rises sharply, that's the early-warning that metrics like `std_across_prompts` can't provide.
3. **Validate the RM input format before training**, not by guessing. The `p + "</s>" + r` concatenation produces `</s></s>` because `p` already ends in `</s>`. Wrong RM input format directly weakens the training signal and probably makes hacking easier (cheap exploits are easier when the RM is being fed off-distribution inputs).
4. **Train for fewer steps with stronger KL** instead of more steps with weaker KL. Reward plateaus around step 200 regardless; the question is where the plateau lands.
5. **Use LoRA (`peft_config`) instead of layer freezing.** Same regularization spirit, better capacity/stability tradeoff, more modern. Not strictly a hacking fix, but a quality-of-life improvement.
6. **Pair the RM with a length or diversity penalty.** Crude but effective: penalize the policy for n-gram overlap with other generations in the same batch, or for using the same final-N tokens too consistently. Directly targets the magic-word attractor.
7. **Pin the entire environment from day one** in a single `requirements.txt` with an `--extra-index-url` line for CUDA wheels. Skip the migration through three different torch versions.

## What Was Shipped vs. What Should Be Used

- **`tsaxena/gpt2-large-ppo-prompt-tags`** (PPO output) — uploaded as a research artifact. Should carry a model-card warning that it exhibits magic-word reward hacking and should not be used as a drop-in replacement for the SFT base. Useful as a teaching example of how PPO can fail subtly.
- **`tsaxena/gpt2-large-prompt-tags`** (SFT base) — recommended for production use. Higher diversity, more prompt-specific stylistic choices.

## What's Next

- **Re-run with `init_kl_coef=0.5`** and the same other settings. If reward only climbs to +0.3 but generations stay diverse, that's a win.
- **Implement n-gram-overlap-across-batch as a metric** and log it during training. This is the missing alarm bell from this run.
- **Try LoRA + stronger KL** as a paired change. Modern RLHF recipe.
- **Inspect `toloka/prompts_reward_model`'s training data** (if accessible) to understand exactly which patterns it scores well. Knowing the failure mode of the RM ahead of time would have made this whole result predictable.
- **Add a model card warning** to the PPO HF repo. Honest documentation > inflated metrics.

---

The single sentence I'd take away from this: **the reward curve told the truth about optimization; only reading the actual generations told the truth about quality — and they were different stories**.
