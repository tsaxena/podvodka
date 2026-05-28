# RL Training Scripts

Two scripts are provided: `run.py` (PPO) and `run_grpo.py` (GRPO). See per-algorithm sections below.

---

# PPO Setup Summary

## Environment

- **Hardware**: RunPod, 1× A100 80GB, driver supporting CUDA 12.8
- **Stack** (after a lot of fighting):
    - torch 2.4.1 + cu124, torchvision 0.19.1, torchaudio 2.4.1
    - transformers 4.45.2
    - trl 0.11.4 (pinned to keep the classic PPO API)
    - tokenizers 0.20.x, accelerate, peft, datasets 4.8.4, fsspec ≤ 2026.2.0
- **Cache locations**: `HF_HOME`, `PIP_CACHE_DIR`, `TMPDIR` all redirected to `/workspace` (container root is only 20 GB on RunPod; `/workspace` has terabytes)

## Models and Data

- **Policy / ref model**: `tsaxena/gpt2-large-prompt-tags` (GPT-2-large, 36 transformer blocks), wrapped with `AutoModelForCausalLMWithValueHead`
- **Reward model**: `toloka/prompts_reward_model`, used as a `text-classification` pipeline; cannot be retrained
- **Data**: train/val CSVs of prompt strings with `</s>` separators

## PPO Configuration

Key hyperparameters:

- `lr=1.4e-5`, AdamW (β=0.9/0.95, wd=1e-6)
- `batch_size=128` rollouts, `mini_batch_size=128`, `ppo_epochs=4`
- `init_kl_coef=0.05`, `target=6` (adaptive KL)
- `cliprange=0.2`, `cliprange_value=0.2`, `vf_coef=1`
- Generation: `max_new_tokens=80`, `do_sample=True`, `top_p=1.0`, `top_k=0`
- Cosine LR decay to 10% of initial over 10,000 steps

## Training Strategy

- **Partial fine-tuning**: only the top 2 of 36 transformer blocks unfrozen, plus the value head. Acts as a regularizer (PPO stability), reduces optimizer memory, and concentrates noisy RL gradients on the most task-relevant params. This is the trlX-style recipe; LoRA would be the modern alternative.
- **Frozen reference model** + **KL penalty** to keep the policy near base, complementing the layer freezing.

## Issues Found and Fixed

| Problem | Fix |
|---|---|
| `cliprange_reward` not in trl | It's a trlX parameter, not HF trl — removed (clip rewards manually if needed) |
| `PPOTrainer` API mismatch | Pinned `trl==0.11.4`, the last version with the classic API |
| Disk full (container root, 20 GB) | Moved HF/pip/tmp caches to `/workspace` |
| torch/torchvision/torchaudio mismatch | Aligned all three to torch 2.4.1 + cu124 |
| transformers too new for torch 2.4 | Downgraded to transformers 4.45.2 |
| fsspec/datasets conflict | Unpinned fsspec or pinned to ≤2026.2.0 |
| **Training silently on CPU** | Added `accelerator_kwargs={"cpu": False}` to `PPOConfig` + device assertion |
| 50s/step (reward scoring on every sample) | Batched reward pipeline calls (`score_batch`, batch_size=32) |
| Slow generation | Raised `ppo_trainer.generate(batch_size=32)` from default 4 |
| LR schedule was a no-op | Fixed `eta_min = lr * 0.1` |

## Reward Model Sanity Check (Done Pre-Training)

Built a `rm_sanity.py` script that scores ~20 good/bad pairs and verifies:

- Mean gap (good − bad) is clearly positive
- Pairwise win rate > 90%
- No NaN/Inf, reasonable score range
- Result: "Looks reasonable — RM separates good from bad" ✅

## W&B Logging

- Enabled via `log_with="wandb"` in PPOConfig
- CLI args auto-logged to `wandb.config`
- Heavy query/response text table gated to every N steps; scalars logged every step

## Training Health (after ~75 steps)

| Metric | Reading | Interpretation |
|---|---|---|
| `env/reward_mean` | **−0.27 → +0.10** | **PPO is working** — the core signal |
| `env/reward_std` | ~0.34 → 0.38, stable | Healthy exploration, no mode collapse |
| `env/reward_dist` | Dark mass shifting up, shape preserved | Distribution improving without collapsing |
| `ppo/loss/value` | 0.13 → 0.025 | Critic fitting well |
| `ppo/loss/total` | Dominated by value loss (`vf_coef=1`) | Falling smoothly |
| `ppo/loss/policy` | ~−0.004 steady | Healthy and small (negative is correct in PPO — see below) |
| `ppo/val/var_explained` | −3.5 → −0.8 | Critic still catches up from random init; trajectory is right |
| `ppo/policy/clipfrac` | ~0.001 | Clip almost never binds — updates are very small |
| `ppo/policy/approxkl` | ~0.0001 | Per-update KL tiny — policy crawls per step |

## Key Insights from the Run

1. **Negative `policy/loss` is correct.** trl reports `-L^CLIP` so the optimizer can minimize it. Negative = the policy is increasing the probability of high-advantage actions. The shape (sharp dip → flat) is steady-state PPO.

2. **The critic lags the policy.** Value head was randomly initialized (the `no v_head weight is found` warning), so `var_explained` starts very negative. It's climbing toward 0 — keep watching.

3. **Updates are extremely conservative.** `clipfrac ≈ 0.001` and `approxkl ≈ 1e-4` are an order of magnitude below the typical PPO sweet spot (clipfrac 0.05–0.25, approxkl ~0.005–0.02). That's the natural consequence of: only 2 blocks unfrozen, lr=1.4e-5, ppo_epochs=4. **Safe but slow.**

4. **Bottleneck migrated.** Was reward scoring (50s/step → ~18s/step after batching). Now the dominant cost is autoregressive generation (80 tokens × 128 sequences). Levers if more speed needed: raise `--gen_batch_size` to 64+, bf16 the policy, shorten `max_new_tokens`.

5. **Loss alone is a poor health signal in PPO.** Reward, KL, clipfrac, and entropy together tell the truth. Loss going down can coexist with policy collapse.

## What to Watch as Training Continues

- `env/reward_mean` — should keep climbing, then plateau
- `objective/kl` (vs reference, not approxkl) — gradual rise OK, spikes mean drift; the adaptive coef should handle it if `target=6` is right
- `objective/entropy` — slow decline OK, sudden crash = mode collapse
- **The actual generated text** in the W&B table — reward going up doesn't guarantee text quality; reward hacking shows up here before it shows up in metrics
- `ppo/val/var_explained` crossing 0 — the moment the critic becomes genuinely useful

## Headroom If Reward Plateaus

In order of likely impact:

1. `--num_layers_unfrozen 4` or `6` (more capacity)
2. Raise LR to 5e-5 (clipfrac and approxkl have huge headroom)
3. Lower `ppo_epochs` to 2 (each batch is barely being squeezed anyway)
4. Switch to bf16 + larger `gen_batch_size` for wall-clock speed
5. Consider a LoRA `peft_config` instead of layer freezing — modern equivalent, better capacity/stability tradeoff

## Open Concerns

- **Reward text uses `p + "</s>" + r`** but `p` already ends in `</s>`, producing `</s></s>` — may or may not match how `toloka/prompts_reward_model` was trained. Worth verifying once.
- **Cosine schedule** decays over 10k steps; if you stop earlier, you got essentially no decay. Adjust `T_max` to match actual run length if relevant.
- **`ppo_epochs=4` over the same 128 rollouts**, while updates are so small, is doing a lot of work for not much movement — likely overkill at current LR.

Overall: a textbook-healthy PPO run that just needs to keep cooking. The fundamentals (environment, devices, RM behavior, reward signal reaching the policy) are all confirmed working.

---

# GRPO Design Decisions (`run_grpo.py`)

## Why GRPO vs PPO

| Dimension | PPO | GRPO |
|---|---|---|
| Critic | Separate value head (extra params, warm-up lag) | None — advantage from group statistics |
| Memory | Model + ref model + value head | Model + ref model only |
| Variance | Reduced by learned baseline | Reduced by within-group normalisation |
| Implementation surface | Large (critic loss, GAE, clipping on both policy and value) | Smaller (single clipped surrogate + group z-score) |

GRPO is the natural choice when you want to avoid the critic cold-start problem (the `var_explained` going deeply negative at the start of PPO runs) and when GPU memory is tight enough that eliminating the value head matters.

## Model

`AutoModelForCausalLM` instead of `AutoModelForCausalLMWithValueHead`. No value head is initialised, so there is no noisy random-init phase and no `vf_coef` to tune. The same partial-freeze strategy is kept (top `--num_layers_unfrozen` transformer blocks + `lm_head`).

## Advantage Estimation

For each prompt, G completions are sampled (`--num_generations`, default 8). The advantage for completion _i_ in the group is:

```
A_i = (r_i − mean(r_1..G)) / std(r_1..G)
```

This is computed inside `GRPOTrainer`; no external baseline model is needed.

## Hyperparameter Mapping from PPO

| PPO arg | GRPO arg | Notes |
|---|---|---|
| `--init_kl_coef 0.05` | `--beta 0.05` | Same KL penalty concept |
| `--num_rollouts 128` | `--batch_size 16 --num_generations 8` | 16 × 8 = 128 total completions/step |
| `--chunk_size 128` | N/A | GRPOTrainer handles mini-batching internally |
| `--ppo_epochs 4` | `--grpo_epochs 1` | GRPO is typically run with 1 inner epoch; the group sampling itself diversifies gradient signal |
| `--vf_coef 1` | N/A | No value head |
| `--cliprange 0.2` | `--cliprange 0.2` | Same ratio clip (ε) |

## Trainer

Uses `GRPOTrainer` from `trl >= 0.12.0`. The PPO script is pinned to `trl==0.11.4` (classic API); GRPO requires a separate environment or an upgrade. The two scripts are intentionally independent — no shared state.

W&B is initialised explicitly with `wandb.init()` before constructing `GRPOTrainer` so that `entity`, `tags`, and `config` all attach to the same run. HF's `report_to="wandb"` path doesn't expose those fields.

## Checkpointing

Periodic checkpoints are delegated to HF Trainer's native `save_steps` / `save_total_limit`. Best-reward checkpointing is handled by `BestRewardCheckpointer`, a `TrainerCallback` that reads `rewards/mean` from the logged metrics and saves `best/` whenever a new high is reached (after `--save_best_after` warmup steps). This mirrors the `best/` logic in `run.py`.

## Reward Function

Signature matches the dataset column name: `reward_fn(completions, prompt=..., **kwargs)`. GRPOTrainer repeats each prompt G times before calling the reward function, so `completions` and `prompt` are always the same length. The reward text format (`p + "</s>" + c`) deliberately mirrors `run.py` so reward distributions are directly comparable.

## Trade-offs and Open Questions

- **Group size G=8 with batch_size=16** gives 128 completions/step matching the PPO rollout count, but each prompt now has 8 on-policy samples instead of 1. This increases diversity of the gradient signal at the cost of running the reward model on 8× as many texts per prompt.
- **`grpo_epochs=1`** is the standard choice. Increasing it reuses the same rollouts for multiple gradient steps (like `ppo_epochs` in PPO) but introduces off-policy error since GRPO's advantage is computed once at rollout time.
- **No adaptive KL**: PPO uses an adaptive KL controller targeting `kl=6`. GRPO uses a fixed `beta`; if the policy drifts far from reference, `beta` may need manual tuning.
- **Reward text `</s></s>` double-separator**: inherited from `run.py`. Both scripts have the same potential mismatch with how `toloka/prompts_reward_model` was trained. Verify once before drawing reward-scale comparisons between the two runs.