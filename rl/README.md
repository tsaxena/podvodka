# RL Fine-Tuning

Three training methods are implemented. All scripts require CUDA.

---

## Environment

PPO and GRPO use incompatible `trl` versions and must run in separate environments.

```bash
# PPO — trl 0.12 rewrote the PPOTrainer API; pin to 0.11.4
pip install trl==0.11.4 transformers==4.45.2 datasets fsspec==2023.10.0
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# DPO — same environment as PPO
# (same pin; DPOTrainer API is stable across 0.11.x)

# GRPO — GRPOTrainer shipped in 0.12; requires a newer trl
pip install "trl>=0.12.0" transformers datasets
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```

Redirect the HuggingFace cache if your container root is small:

```bash
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub
```

Force single-GPU to avoid accelerate trying distributed init:

```bash
accelerate config  # choose "No distributed training" / "GPU 0" when prompted
```

---

## PPO

Online RL with a value head. Fast to set up; prone to reward hacking when the
reward model is imperfect. See `ppo_results_report.md` for what went wrong.

```bash
python run_ppo.py \
    --train_path  /workspace/podvodka/data/train_strings.csv \
    --val_path    /workspace/podvodka/data/val_strings.csv \
    --output_path /workspace/podvodka/models/gpt2-large-ppo-prompt-tags \
    --num_steps 2000 \
    --lr 1.4e-5 \
    --init_kl_coef 0.05 \
    --num_layers_unfrozen 2 \
    --save_every 500 \
    --wandb_project podvodka-rl \
    --wandb_run_name ppo-run-1
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--init_kl_coef` | 0.05 | Raise to 0.3–0.5 to suppress reward hacking |
| `--num_layers_unfrozen` | 2 | Number of transformer blocks to unfreeze (from the top) |
| `--num_steps` | 2000 | PPO steps; plateau was reached around 400 in the first run |
| `--save_every` | 500 | 0 disables periodic checkpoints |
| `--keep_last_n` | 3 | Periodic checkpoints to retain; `best/` and `final/` always kept |

---

## DPO

Offline preference learning. Builds a chosen/rejected dataset from the SFT
model first, then trains with DPO loss. No RL loop, so reward hacking manifests
differently (length inflation, hedge insertion). See `dpo_results_report.md`.

**Phase 1 + Phase 2 combined (build dataset then train):**

```bash
python run_dpo.py \
    --build_dataset \
    --train_path    /workspace/podvodka/data/train_strings.csv \
    --dataset_path  /workspace/podvodka/data/preferences.jsonl \
    --num_prompts 1000 \
    --candidates_per_prompt 8 \
    --output_path /workspace/podvodka/models/gpt2-large-dpo-prompt-tags \
    --epochs 2 \
    --lr 5e-6 \
    --beta 0.1 \
    --wandb_run_name dpo-run-1
```

**Phase 2 only (reuse an existing preference dataset):**

```bash
python run_dpo.py \
    --dataset_path /workspace/podvodka/data/preferences.jsonl \
    --output_path  /workspace/podvodka/models/gpt2-large-dpo-prompt-tags \
    --epochs 2 \
    --lr 5e-6 \
    --beta 0.1 \
    --wandb_run_name dpo-run-1
```

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--candidates_per_prompt` | 8 | Completions generated per prompt for pair construction |
| `--min_score_gap` | 0.3 | Skip pairs where chosen−rejected gap is below this |
| `--beta` | 0.1 | DPO temperature; higher = stronger pull toward reference |
| `--lr` | 5e-6 | DPO is LR-sensitive; stay in 1e-6 to 5e-6 |
| `--epochs` | 2 | Training epochs over the preference dataset |
| `--no_wandb` | off | Pass to disable W&B logging |

---

## GRPO

Critic-free group relative policy optimisation. No value head; advantage is
computed from the spread of rewards within each group of G completions. Requires
`trl >= 0.12.0`.

```bash
python run_grpo.py \
    --train_path  /workspace/podvodka/data/train_strings.csv \
    --output_path /workspace/podvodka/models/gpt2-large-grpo-prompt-tags \
    --num_steps 10000 \
    --lr 1.4e-5 \
    --beta 0.05 \
    --num_generations 8 \
    --batch_size 16 \
    --num_layers_unfrozen 2 \
    --save_every 500 \
    --wandb_project podvodka-rl \
    --wandb_run_name grpo-run-1
```

Effective completions per gradient step = `batch_size × num_generations`
(16 × 8 = 128 by default).

Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--num_generations` | 8 | Group size G; larger = more stable advantage estimates |
| `--batch_size` | 16 | Prompts per step (not completions) |
| `--beta` | 0.05 | KL penalty; analogous to `init_kl_coef` in PPO |
| `--cliprange` | 0.2 | Ratio clip ε; same role as PPO's `cliprange` |
| `--grpo_epochs` | 1 | Inner epochs over each rollout batch |
| `--num_layers_unfrozen` | 2 | Transformer blocks unfrozen from the top |
| `--save_every` | 500 | 0 disables periodic saves |
| `--keep_last_n` | 3 | Periodic checkpoints to retain |
| `--no_wandb` | off | Pass to disable W&B logging |

---

## Checkpoints

All three scripts write to `--output_path` with the same layout:

```
output_path/
  step-000500/   ← periodic (rotated, keep_last_n)
  step-001000/
  best/          ← highest mean reward seen after save_best_after steps
  final/         ← written when training completes normally
  interrupted/   ← written on KeyboardInterrupt
```

---

## Sanity checks after training

```bash
# Check a PPO checkpoint
python sanity_check_ppo.py \
    --model_path /workspace/podvodka/models/gpt2-large-ppo-prompt-tags/best \
    --val_path   /workspace/podvodka/data/val_strings.csv

# Compare PPO checkpoints (detect reward hacking progression)
python compare_checkpoints.py \
    --checkpoint_dirs \
        /workspace/podvodka/models/gpt2-large-ppo-prompt-tags/step-000500 \
        /workspace/podvodka/models/gpt2-large-ppo-prompt-tags/step-001000 \
        /workspace/podvodka/models/gpt2-large-ppo-prompt-tags/best \
    --val_path /workspace/podvodka/data/val_strings.csv

# Compare DPO checkpoint vs SFT base
python compare_checkpoints_dpo.py \
    --dpo_path  /workspace/podvodka/models/gpt2-large-dpo-prompt-tags \
    --base_path tsaxena/gpt2-large-prompt-tags \
    --val_path  /workspace/podvodka/data/val_strings.csv
```



```
python run_grpo.py \
  --beta 0.05 \
  --log_completions \
  --save_best_min_interval 100 \
  --local_scratch_dir /tmp/grpo_scratch \
  --train_path /workspace/podvodka/data/train_strings.csv \
  --val_path /workspace/podvodka/data/val_strings.csv \
  --output_path /workspace/podvodka/models/gpt2-large-grpo-prompt-writing \
  --reward_model_path toloka/prompts_reward_model \
  --base_model_path tsaxena/gpt2-large-prompt-tags \
  --wandb_project podvodka-rl \
  --wandb_run_name grpo-beta0.05-guarded-v2
```