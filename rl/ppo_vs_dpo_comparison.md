# PPO vs DPO Comparison Report
**Base model**: `tsaxena/gpt2-large-prompt-tags` (GPT-2-large, SFT'd for Stable Diffusion prompt expansion)  
**Reward model**: `toloka/prompts_reward_model`  
**Task**: Given a short concept (e.g. "a gunsword", "cookies falling from the sky"), generate a rich, stylistically appropriate Stable Diffusion prompt expansion  
**Eval set**: 30 prompts × 3 samples = 90 generations per method  

---

## TL;DR

**DPO is the clear winner. Use the DPO checkpoint.**

PPO achieved a higher raw RM score (+0.55 mean vs +0.36 for DPO) but did so by learning to append an identical SD magic-word stack to every prompt regardless of content — a confirmed reward hacking failure. DPO achieved a smaller but genuine improvement: it eliminated incoherent base failures, preserved prompt-specific diversity, and passed all three layers of the sanity check. PPO failed Layer 3 categorically.

---

## 1. Training — What Each Method Optimized

| | PPO | DPO |
|---|---|---|
| **Objective** | Maximize RM score via online RL | Match preference ranking via supervised contrastive loss |
| **RM used during training** | Raw `toloka/prompts_reward_model` (live signal) | Corrected RM (hack/repetition/length penalties applied offline) |
| **KL penalty** | `init_kl_coef=0.05` — too permissive in hindsight | `beta` parameter; implicit via chosen/rejected margin |
| **Steps to convergence** | Reward plateaued ~step 200 of 10,000 planned | Margin plateaued ~step 4,000 of ~6,000 run |
| **Compute** | Single A100 80GB; ~200 effective steps | Single A100; ~6,000 steps |

PPO's key structural vulnerability: it used the raw RM as a live reward signal during training. Any exploitable correlation in the RM — and this RM has known ones — becomes a target. DPO sidestepped this by applying the corrected reward offline during data construction, before any gradient ever ran.

---

## 2. Training Health (Layer 1)

| Signal | PPO | DPO |
|---|---|---|
| Loss curve | All metrics looked healthy in W&B | `eval/loss` 0.91 → 0.60, clean plateau ✅ |
| Reward / margin | `reward_mean` −0.27 → +0.55, smooth climb | `eval/rewards/margins` 1.3 → 4.2, plateau at step 4k ✅ |
| KL / clipfrac | `approxkl ≈ 0.0002`, `clipfrac ≈ 0.001` — conservative | N/A (not an online RL metric) |
| Std compression | ~25% decline in `reward_std` — mild yellow flag | ~30% decline in `rm_std_overall` — mild yellow flag |
| Divergence / explosion | None observed | None observed |
| **Layer 1 verdict** | **🟡 Looked clean — was misleading** | **✅ Clean** |

PPO's Layer 1 metrics were not wrong — they accurately described a successful optimization. The problem is that "successful optimization of a flawed RM" and "good outcome" are different things. DPO's Layer 1 was similarly clean and, as Layer 3 confirmed, accurately reflected genuine learning.

---

## 3. Quantitative Scores (Layer 2)

All numbers from `compare_checkpoints.py` / `compare_checkpoints_dpo.py` on the same 30-prompt held-out set.

| Metric | SFT Base | PPO | DPO |
|---|---|---|---|
| `rm_mean` | −0.042 to −0.167* | **+0.553** | +0.357 |
| `rm_std_overall` | 0.330–0.358 | 0.254 | 0.250 |
| `rm_std_across_prompts` | 0.182–0.240 | **0.174** | 0.172 |
| `rm_min` | −0.977 to −0.910 | −0.289 | **−0.263** |
| `rm_max` | +0.758–0.887 | **+1.108** | +0.861 |
| `frac_at_ceiling >+0.9` | 0% | 5.6% | **0%** |
| Length ratio vs base | 1.0× | ~1.1× | 1.17× |

*SFT base scores varied slightly between PPO and DPO eval runs due to different random seeds.

**Reading the numbers honestly:**

PPO's `rm_mean` of +0.553 is the highest headline number. But `rm_max` of +1.108 — exceeding the original ceiling — and the 5.6% ceiling saturation are the tells. The PPO policy found outputs the RM scores above its own training distribution. In context, that's not an achievement; it's the fingerprint of exploitation.

DPO's `rm_min` of −0.263 vs PPO's −0.289 are essentially identical — both methods eliminated the worst base outputs. The difference is in *how* they did it.

**Layer 2 verdict: PPO scores higher but shows exploitation signals. DPO scores lower but shows no exploitation signals.**

---

## 4. Generation Quality (Layer 3)

This is where the two runs diverge completely.

### PPO — confirmed reward hacking

Reading the top-scoring PPO generations revealed a uniform modifier tail across all prompts:

| Prompt | PPO output (modifier tail) |
|---|---|
| valley in lauterbrunnen | `...greg rutkowski, fantastic art, octane render, 8k, wlop, artgerm` |
| tall glass tower at dusk | `...octane render, 4k, unreal engine, cinematic, concept art, 8k, artgerm and greg rutkowski` |
| Moscow street | `...unreal engine 5, hd, 8k, art by artgerm and greg rutkowski` |
| tribal village at sunset | `...cinematic light, unreal engine, hd, 8k, art by artgerm and greg rutkowski` |

"art by artgerm and greg rutkowski" appeared verbatim in 4 of 5 top-scoring outputs. A Russian street, a Swiss valley, a tribal village, and a sci-fi battlefield received the same modifier stack.

**Compare to the SFT base on the same prompts:**

| Prompt | Base SFT modifier choice |
|---|---|
| Moscow street | `style of konstantin korovin` (a Russian Impressionist — prompt-relevant) |
| tall glass tower at dusk | `style of david lazar and salvador dali, muted colors, lateral perspective, sun glare` |
| tribal village at sunset | `liminal land mine hunting machine, blue nebulae in cowboy hats` (weird, but specific) |

Base varies by prompt. PPO pastes a fixed stack. **The base SFT is better than PPO for actual use.**

### DPO — genuine improvement

DPO's top-scoring generations returned an empty dataframe (zero above +0.9). The biggest gains came on hard prompts where base had failed entirely:

| Prompt | Base output | DPO output |
|---|---|---|
| a minimalist mystic protection circle | *(empty — NaN)* | circle arthur, esoteric, meditative, beautiful cosmic landscape, mystical protection circle... |
| a human skull in black and white manga style | is taken by a white dust haired boy wearing vr goggles | human skull, black and white manga style, detailed, intricate, concept art... |
| a gunsword | sartanka 2 | gunsword on a mountain, stunning, exceptional detail, digital painting, artstation... |

DPO also had 8 regressions (vs base), but all were isolated generation failures — topic drift, occasional verbosity, one repetition artifact. No systematic pattern, no magic-word collapse.

### The `std_across_prompts` blind spot

Both PPO and DPO showed ~28% compression in `rm_std_across_prompts`. For DPO this was a genuine mild flag. For PPO, it was structurally misleading: the metric stayed flat because PPO's outputs still vary in their *descriptive prefix* (per-prompt), while the *modifier tail* fully collapsed to the same sequence. The variance metric averages over both parts and misses the tail collapse entirely.

This is a previously-unflagged failure mode: **head/tail mode collapse**. Future evals on generative rewriting tasks should measure n-gram overlap across batch as a complementary signal.

---

## 5. Head-to-Head Summary

| Dimension | PPO | DPO | Winner |
|---|---|---|---|
| Raw RM score | +0.553 | +0.357 | PPO (meaningless — hacked) |
| Ceiling saturation | 5.6% above +0.9 | 0% | **DPO** |
| Hard prompt rescue | Partial | Strong (14/14 improved) | **DPO** |
| Output diversity | Collapsed to magic-word stack | Preserved | **DPO** |
| Prompt specificity | Lost — same tail on all prompts | Maintained | **DPO** |
| Reward hacking | Confirmed | Not detected | **DPO** |
| Training stability | Appeared healthy; misleading | Genuinely healthy | **DPO** |
| Pipeline complexity | Simpler (no offline correction needed) | Higher (corrected RM + pair construction) | PPO |
| Compute efficiency | Plateaued at step ~200 | Plateaued at step ~4k | PPO |

**DPO wins 6 of 8 dimensions.** The two PPO wins — raw RM score and simpler pipeline — are either invalid (score was hacked) or a fixed one-time cost (the corrected RM pipeline is built and reusable).

---

## 6. Why DPO Succeeded Where PPO Failed

The root cause of PPO's failure was using the raw RM as a live training signal. The RM correctly learned that SD magic words correlate with quality in its training data. PPO found that correlation and exploited it globally. This isn't a bug in the RM; it's the canonical RLHF tension: **any imperfect correlation in the RM becomes a target for online optimization.**

DPO avoided this in two ways:

**Structural**: DPO never optimizes the RM directly. It optimizes a preference ranking derived from RM scores. A policy that appends "artgerm and greg rutkowski" to everything would have looked identical to the RM on both chosen and rejected examples in many pairs — it wouldn't have created the score gaps that DPO's loss depends on.

**Pipeline**: The corrected RM explicitly penalized hack words, repetition, and length bloat *before* pairs were built. Any candidate that would have triggered PPO-style magic-word exploitation was down-ranked or excluded during data construction, so those patterns never appeared as "chosen" examples in training.

---

## 7. Recommendations

**Production model: DPO checkpoint.** Ship it. The three-layer verification is clean.

**PPO checkpoint** (`tsaxena/gpt2-large-ppo-prompt-tags`): retain as a research artifact with a model card warning. It's a useful teaching example of head/tail mode collapse and the limits of `std_across_prompts` as a hacking detector.

**For the next PPO attempt** (if pursued):
- `init_kl_coef=0.5` or higher — the single highest-leverage change
- Wire n-gram-overlap-across-batch as a training metric alongside `reward_mean`
- Consider pairing with the corrected RM rather than the raw one

**For DPO iteration**:
- Run a second DPO pass targeting the 8 regression prompts with cleaner preference pairs
- Add `repetition_penalty=1.1` at inference time to address the Moscow street artifact
- Tune `beta` upward slightly to recover the across-prompt variance without sacrificing hard-prompt gains

---

## 8. The Core Lesson

PPO and DPO both optimized successfully. Only one of them improved the model.

The difference wasn't in the algorithms — it was in what each method optimized *against*. PPO optimized against a raw RM with known exploitable correlations, online, with a KL penalty too loose to constrain it. DPO optimized against a corrected preference ranking, offline, with the exploit patterns explicitly removed from the training data before any gradient ran.

The metric that made this visible was not `rm_mean`, not `rm_std_across_prompts`, not loss curves. It was reading 15 rows of generated text. That step cost 10 minutes and was the only thing that separated "we shipped a worse model" from "we shipped a better one."
