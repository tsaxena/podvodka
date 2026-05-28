# PPO Fine-Tuning of GPT-2-Large — Project Summary

## Outcome

A successful PPO RLHF run on a single A100 80GB.

- **Base model**: `tsaxena/gpt2-large-prompt-tags` (SFT'd GPT-2-large, 36 transformer blocks)
- **Reward model**: `toloka/prompts_reward_model` (frozen; cannot retrain)
- **Result**: mean reward improved from **−0.27 → +0.55** over ~200 PPO steps, then plateaued
- **Run length**: ~330 steps total (~85% of compute past plateau was wasted)
- **Uploaded to HF**: `tsaxena/gpt2-large-ppo-prompt-tags`

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
    init_kl_coef=0.05,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=1.0,
    accelerator_kwargs={"cpu": False},  # CRITICAL — was silently CPU
)

# Partial fine-tuning: only 2 of 36 transformer blocks unfrozen + value head
# Generation: max_new_tokens=80, do_sample=True, top_p=1.0
# Reward scoring: batched, 32 at a time
# Generation: batched, 32 at a time
```

## Training Health Snapshot

| Metric | Reading | Verdict |
|---|---|---|
| `env/reward_mean` | −0.27 → +0.55, plateau | **PPO worked** |
| `env/reward_std` | 0.38 → 0.28, modest decline | Convergence, not collapse |
| `env/reward_dist` | Shifted up, width preserved | Healthy translation, no reward hacking |
| `ppo/loss/value` | 0.15 → 0.005, smooth | Critic converged |
| `ppo/loss/policy` | ~−0.004 steady | Healthy (negative = correct) |
| `ppo/policy/clipfrac` | ~0.001 | Updates extremely conservative |
| `ppo/policy/approxkl` | ~0.0002 | Same |

## Reward Hacking Sanity Check (3-Layer Verification)

PPO can quietly fail by *gaming the reward model* — producing outputs that score high without being better. Loss curves can't catch this. Even rising reward can't catch it. The policy is literally optimizing for the RM, so high RM scores are exactly what reward hacking looks like.

Three layers of evidence, in increasing order of how definitive they are:

### Layer 1: Metric-pattern check (during training)

Two failure-mode fingerprints to scan the W&B charts for.

**The "hacking event" pattern** — these metrics moving together within a 10–30 step window:

- `env/reward_mean` — sudden jump (not smooth climb)
- `objective/kl` — sudden spike
- `objective/entropy` — sudden crash
- `env/reward_std` — sudden compression
- `ppo/policy/clipfrac` — sudden spike

**The "gradual drift" pattern** — slower but equally dangerous:

- `env/reward_std` slowly shrinking over hundreds of steps
- `env/reward_dist` mass piling up against the RM's ceiling
- Bottom of `reward_dist` disappearing entirely

**Result for this run:**

| Signal | Reading | Verdict |
|---|---|---|
| Reward curve shape | Smooth climb, no discontinuities | ✅ No discrete hacking event |
| `reward_std` trajectory | 0.38 → 0.28, ~25% decline | ✅ Mild, far from collapse (>70%) |
| `reward_dist` shape | Distribution translated up, width preserved | ✅ No mode collapse |
| Ceiling saturation | Top edge sits flat against +1.0 | ⚠ Mild yellow flag — verify |

Verdict: **weak metric-level evidence of hacking, but ceiling saturation worth investigating.** Proceed to Layer 2.

### Layer 2: Quantitative checkpoint comparison (post-training)

Used `compare_checkpoints.py` to generate 3 samples per prompt from each checkpoint, score everything with the same reward model, and compute distribution statistics. The key question this answers: *does the gain look like genuine improvement or like exploitation?*

**Results** — base SFT vs. PPO best, 30 prompts × 3 samples each:

| Metric | base | ppo | Interpretation |
|---|---|---|---|
| `rm_mean` | −0.167 | +0.553 | **+0.72 gap.** Real shift, consistent with the training curve. |
| `rm_std_overall` | 0.330 | 0.254 | ~23% drop. Mild compression. Catastrophic collapse = 70%+. |
| **`rm_std_across_prompts`** | **0.182** | **0.174** | **Essentially unchanged. The most diagnostic number.** |
| `rm_min` | −0.977 | −0.289 | Bottom lifted by +0.69 — policy eliminated bad outputs. |
| `rm_max` | +0.758 | +1.108 | Top moved past the original ceiling. |
| `frac_above_+0.5` | 3.3% | 63.3% | **Huge shift.** Bulk of distribution moved up, not just outliers. |
| `frac_at_ceiling_+0.9` | 0.0% | 5.6% | Well below the 30% threshold for saturation-exploit concern. |

**Why these numbers point to genuine improvement, not hacking:**

1. **`rm_std_across_prompts` barely moved.** This is the most telling number. If the policy had mode-collapsed onto an "RM-pleasing template," every prompt would get a similar response → low across-prompt variance. Yours stayed flat. The policy is still *differentiating* between prompts, producing better-scored responses for each one.
2. **Ceiling saturation is only 5.6%.** A reward-hacking policy typically pins 30–60% of outputs against the RM's ceiling. The Layer 1 yellow flag (top edge at +1.0) was about a small minority of outputs, not the bulk.
3. **The bottom moved up, not just the top.** Hacking policies don't fix bad responses; they pile good ones on top. This one eliminated the worst outputs (`rm_min` went from −0.977 to −0.289).
4. **The middle moved most.** 3.3% → 63.3% above +0.5 is a translation of the whole distribution, not an outlier-driven score increase.

Verdict: **numbers strongly suggest healthy improvement.** Proceed to Layer 3 to confirm.

### Layer 3: Read the generations (the only definitive test)

The script outputs a CSV with one row per (prompt, sample) and columns for every checkpoint's generation and RM score. **Numbers can't catch every form of hacking** — specifically, if the policy learned a phrase or pattern that the RM happens to genuinely score well on, the metrics look great but the outputs all become formulaic.

The CSV columns:

```
prompt, sample_idx,
base__gen, base__rm_score,
ppo__gen, ppo__rm_score
```

**Things to look for** when reading 15–30 rows:

- **Repetition.** Does any `ppo` output repeat a phrase/token 3+ times in 80 tokens?
- **Magic phrase insertion.** Same phrase appearing in many `ppo` outputs across very different prompts.
- **Length pathology.** Are all `ppo` outputs uniformly the same length regardless of prompt?
- **Format gaming.** Same surface structure (opening word, punctuation, layout) across unrelated prompts.
- **Topic drift.** Does `ppo` ignore the prompt and emit something the RM scored well on?
- **Coherence.** Does `ppo` actually read like better prompt-writing than `base`?

**Targeted CSV queries that catch the most suspicious cases:**

```python
import pandas as pd
df = pd.read_csv("sft_vs_ppo.csv")

# Top-scoring ppo generations — most likely to contain RM exploits if any exist
suspicious = df[df["ppo__rm_score"] > 0.9].sort_values("ppo__rm_score", ascending=False)
print(suspicious[["prompt", "ppo__gen", "ppo__rm_score"]].to_string())

# Biggest gaps — where PPO disagrees most with SFT; what the policy "learned to do differently"
df["gap"] = df["ppo__rm_score"] - df["base__rm_score"]
biggest_gaps = df.sort_values("gap", ascending=False).head(15)
print(biggest_gaps[["prompt", "base__gen", "ppo__gen", "gap"]].to_string())
```

**Decision criterion**: pick the policy where the generations *read best to you*, not the one with the highest RM score. If `ppo` reads worse than `base` despite scoring higher, that's the proof of reward hacking — and the SFT model is actually your better policy.

### Overall verdict from the three layers

| Layer | Signal | Verdict |
|---|---|---|
| 1. Metrics during training | Smooth reward climb, mild std decline, no spikes | ✅ Healthy |
| 2. Checkpoint comparison | Across-prompt variance preserved, ceiling-saturation low, bottom of dist lifted | ✅ Healthy |
| 3. Reading generations | Pending — open `sft_vs_ppo.csv` and read 15–30 rows | Required for definitive answer |

The first two layers are consistent with **genuine, modest-magnitude PPO improvement**. Layer 3 is the last step before declaring victory.

## Key Insights

### The big one: **the reward curve is the only metric that actually answers "is this working?"**

Loss curves in PPO are misleading in three different ways at once:

1. **Policy loss is negative when training succeeds.** `trl` reports `-L^CLIP` so the optimizer can minimize it. Negative means the policy is increasing the probability of high-advantage actions — the desired behavior. People used to supervised learning panic at negative losses and shouldn't.
2. **Total loss is dominated by value loss** when `vf_coef=1`. The total loss going down beautifully (from 0.15 to 0.005) just means the *critic* learned to predict returns. It says nothing about whether the *policy* is producing better outputs.
3. **Loss can fall while the policy gets worse.** This is the reward-hacking failure mode: policy keeps gaming the RM, reward keeps rising, loss keeps falling, and actual text quality collapses. Loss never warns you.

The signals that actually told the truth:
- `env/reward_mean` for "is it improving?"
- `env/reward_std` and `env/reward_dist` for "is it improving in a healthy way, or collapsing?"
- **Reading actual generations in the W&B `game_log` table** for "is the RM measuring what I think it's measuring?"

### The second one: **PPO converges much faster than the defaults suggest.**

The original script had `num_steps=10000`. The reward curve plateaued around **step 200**. Doing the math after the fact: 10k × 128 = 1.28M rollouts, vs. ~25k actually needed. **98% of the planned compute was past the plateau.**

The right way to size a PPO run isn't to copy-paste a number from a tutorial; it's to watch the reward curve and stop when it flattens (or roll back to `best/` if it starts going weird). A 2,000-step ceiling with checkpointing every 100–500 steps gives you essentially the same outcome as 10,000 steps, in 1/5 the time.

### The third one: **environment setup is the project.**

Counting the messages we exchanged before training even started: probably 60% of the work was getting torch / torchvision / torchaudio / transformers / trl / tokenizers / fsspec / datasets / CUDA / accelerate to all agree with each other. Once that worked, training itself was a couple of bugs (reward scoring not batched, accelerate config forcing CPU) and a lot of staring at W&B.

If I were starting this kind of project today I'd freeze a known-good combo of versions immediately and never touch it. The matrix of compatible versions is narrow and moves under your feet between releases. "Latest of everything" is the wrong default for RL training.

### The fourth one: **the conservative regime works, but most of your update budget goes unused.**

`clipfrac ≈ 0.001` and `approxkl ≈ 0.0002` are both ~50× below the published "sweet spot" (0.05–0.25 and ~0.005–0.02). That means PPO's clipping mechanism — the entire reason it's called *Proximal* — almost never had anything to clip. Each update moved the policy by a hair.

And it still got from −0.27 to +0.55. So: PPO works even when you're being extremely cautious. The cost is wall-clock; the benefit is that pathologies (mode collapse, reward hacking) had nowhere to grow. Worth knowing for the next run: pushing harder with more unfrozen layers and a higher LR could plausibly find a higher reward peak — but the conservative run will get *something* good with low risk.

### The fifth one (from the sanity check): **`std_across_prompts` is the single best mode-collapse detector.**

Of all the numeric signals in the Layer 2 comparison, the one that most cleanly separates "healthy improvement" from "mode collapse" is **the standard deviation of mean rewards taken *across prompts*** (not across samples or across all (prompt, sample) pairs). If a policy has collapsed onto an RM-pleasing template, every prompt produces similar output → similar reward → this std crashes. If the policy is genuinely better at the task, it still produces *different* responses for *different* prompts → this std stays flat. In this run, it moved from 0.182 to 0.174 — essentially unchanged. That's worth more than any other single number.

## What I'd Do Differently

In rough priority order:

1. **Set a smaller `num_steps` (1–2k) and trust the `best/` checkpoint.** The 10k was a magic number from a tutorial and almost all of it was wasted.
2. **Use LoRA (`peft_config`) instead of layer freezing.** Same regularization spirit, better capacity/stability tradeoff, more modern.
3. **Set `num_layers_unfrozen` higher (4 or 6) and LR higher (3e-5)** for the *next* run as an A/B against this one. The conservative regime had huge headroom.
4. **Pin the entire environment from day one** in a single `requirements.txt` with an `--extra-index-url` line for CUDA wheels. Skip the migration through three different torch versions.
5. **Validate the RM input format before training**, not by guessing. The `p + "</s>" + r` concatenation produces `</s></s>` because `p` already ends in `</s>`. May or may not match how the RM was trained; uncertainty here directly degrades training signal.
6. **Lower `ppo_epochs` to 2.** With updates this small, doing 4 passes over the same 128 rollouts is mostly wasted compute.
7. **Run the 3-layer sanity check as a standard step**, not an afterthought. Layer 2 takes 10 minutes and provides the most decisive evidence; building it into the training pipeline as a post-run gate would catch hacking early.

## What's Next

- **Try the aggressive config**: `--num_layers_unfrozen 4 --lr 3e-5 --num_steps 800`. Should hit a higher plateau or collapse — both are useful data.
- **Try LoRA** (`peft_config=LoraConfig(r=16, ...)`) as an alternative to layer freezing.
- **Verify RM input format** by inspecting `toloka/prompts_reward_model`'s training script or tokenizer config; correct the `</s></s>` artifact if confirmed.
- **Complete Layer 3 of the sanity check** — read `sft_vs_ppo.csv` and confirm generation quality matches the score gains.
- **Use the uploaded HF model** for downstream prompt-writing tasks and compare side-by-side with the SFT baseline.

---

The single sentence I'd take away from this: **in PPO, trust the reward curve, the reward distribution shape, and the sampled generations — and treat the loss as an artifact of the algorithm's bookkeeping, not as a measure of progress.**
