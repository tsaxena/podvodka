# Step 1: Supervised Fine-Tuning

## Environment Setup

Install system dependencies:
```bash
sudo apt update && sudo apt install -y \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    gcc \
    python3-dev
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

# GPT-2 Large Continued Pretraining Sweep — Decision Summary

## Task Setup

Continued pretraining (not fine-tuning, not from-scratch) of GPT-2 Large on a domain-specific corpus of prompt-completion pairs.

- **Dataset**: ~1.08M words / ~1.7M tokens / ~43k examples
- **After chunking**: 1,723 training blocks of 1024 tokens (concatenated stream, remainder dropped)
- **Hardware**: Single A100 80GB

## Search Strategy

- **Bayesian optimization** with **hyperband early-termination** (`eta=3`, `min_iter=20`). Bayes builds a surrogate model after ~10 random-init runs to steer toward promising regions; hyperband kills bad runs at ~20 and ~60 steps so compute concentrates on good ones.
- **20 total runs** (`run_cap: 20`). Below 15 is essentially random search; above 30 hits diminishing returns on this search space.
- **Metric**: `best_eval_loss` (minimize). Loss instead of perplexity because they're monotonically related, and loss is logged natively by the Trainer at every eval step while perplexity isn't.

## Hyperparameters Swept

| Parameter | Range | Reasoning |
|---|---|---|
| `learning_rate` | log-uniform 5e-6 to 5e-5 | Continued pretraining sweet spot — high enough to shift the model toward the domain, low enough to avoid catastrophic forgetting. Below the original GPT-2 pretraining LR (~1.5e-4) but above typical fine-tuning LRs (~1e-5). |
| `warmup_ratio` | uniform 0.01 to 0.05 | Short warmup is sufficient for continued pretraining; the model is already initialized to reasonable values. |
| `lr_scheduler_type` | {cosine, linear} | Two reasonable defaults; bayes can pick. |
| `adam_beta2` | {0.95, 0.98} | 0.95 helps stability on noisy gradients; 0.98 is more standard. Beta1 fixed at 0.9. |
| `per_device_train_batch_size` | {4, 8} | Hardware-bounded by A100 memory with GPT-2 Large + grad checkpointing. |
| `gradient_accumulation_steps` | {2, 4, 8} | Tuned down from initial [16, 32, 64] after discovering only 1,723 training blocks → effective batch size needs to be modest to give ~80–600 optimizer steps per run rather than 9. |
| `seed` | {42, 1337} | Two seeds to expose run-to-run variance from initialization. |

## Hyperparameters Fixed (Not Swept)

- `weight_decay`: 0.01 (standard for LM training)
- `max_grad_norm`: 1.0
- `adam_epsilon`: 1e-8
- `block_size`: 1024 (GPT-2's max context)
- `num_train_epochs`: 3 (small dataset, more epochs risks overfitting)
- `optim`: adamw_torch_fused (fast on Ampere)
- `fp16`: true (GPT-2 era model, more stable than bf16 here)
- `gradient_checkpointing`: true (memory savings allow batch 8)

## Sweep Mechanics

- **`save_strategy=no`** during sweep — checkpoints aren't needed for HP selection, just disk overhead. Save once with the winning HPs in a final dedicated run.
- **Boolean flags hardcoded in `command:`** as `--flag` (no value) rather than via `parameters: value: true`, to avoid HF argparse misinterpreting `--fp16=true` as a string and falling back to CPU autocast.
- **`output_dir` and `--save_strategy=no` hardcoded in `command:`** rather than `parameters:`, to work around YAML quoting bugs and ensure required args always pass through.

## Code Changes to Support the Sweep

- Removed manual `wandb.init()` at module top — it created a parallel wandb run with no sweep config attached, breaking metric logging.
- Uncommented the `--overwrite_output_dir` check in `detect_last_checkpoint` so each run starts fresh instead of "resuming" a previous run's completed state in 2 seconds.
- Added `WandbBestEvalCallback` (a `TrainerCallback`) that writes `best_eval_loss` to `wandb.run.summary` after each eval. Without this, hyperband-terminated runs have no summary value, leaving sweep panels empty.

## Expected Scale

- ~80–650 optimizer steps per run depending on which `(per_device_batch, grad_accum)` combo bayes samples
- ~5–15 min per full run on A100
- Hyperband kills ~50–60% of runs early
- **Total wall time: ~1.5–3 hours for 20 runs**

## Results (after 10 runs)

- `best_eval_loss` range: ~2.25 to ~2.55
- Best run: **2.25** (vs. pretrained baseline of ~3.77)
- ~40% absolute drop in eval loss; perplexity from ~43 → ~9.5

## Final Training (Post-Sweep)

Sweep ran with `--save_strategy=no`, so the winning model isn't saved. Do one final dedicated run with the winning HPs plus:

- `--save_strategy steps --save_steps 20`
- `--load_best_model_at_end True`
- `--metric_for_best_model eval_loss`
- `--num_train_epochs 10` (or more) with `EarlyStoppingCallback(early_stopping_patience=3)`

Early stopping lets the validation loss decide when to stop instead of guessing an epoch count.

## Open Questions Left Unaddressed

- Whether the LR range needs adjusting if bayes keeps preferring a boundary (would suggest extending the range).
- Whether to add `data_mixing_ratio` to sweep over domain/general-corpus blending — usually higher-impact than HP tuning, but out of scope here.