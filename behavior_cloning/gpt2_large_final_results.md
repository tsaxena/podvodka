# GPT-2 Large Continued Pretraining — Final Results

## Task

Continued pretraining of GPT-2 Large on a prompt-completion corpus that maps natural-language image descriptions to text-to-image tag strings. Each training example follows the format:

```
[BOS] <natural language description> = <tag1, tag2, ...> <|endoftext|>
```

The model learns to convert free-text image descriptions into structured tag vocabularies (e.g., for Stable Diffusion / NovelAI style prompt engineering).

## Dataset

- **Training file**: `train_strings.txt` — 1.08M words, ~1.7M tokens, ~43k examples
- **After chunking** into 1024-token blocks: 1,723 training blocks
- **Validation set**: 430 blocks

## Winning Hyperparameters (from sweep)

| Parameter | Value |
|---|---|
| `learning_rate` | ~4.5e-5 (upper end of 5e-6 to 5e-5 swept range) |
| `lr_scheduler_type` | linear |
| `warmup_ratio` | ~0.03 |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 2 |
| Effective batch size | 16 |
| `adam_beta1` | 0.9 |
| `adam_beta2` | 0.98 |
| `adam_epsilon` | 1e-8 |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| `optim` | adamw_torch_fused |
| `fp16` | true |
| `gradient_checkpointing` | true |

## Final Training Run

8 epochs, ~860 optimizer steps, with `load_best_model_at_end=True` and `metric_for_best_model=eval_loss`. Best checkpoint saved.

### Metrics

| Metric | Value |
|---|---|
| `eval_loss` | **2.2324** |
| `perplexity` | **9.32** |
| `eval_accuracy` (next-token) | **0.5972** |
| `eval_samples` | 430 |
| `eval_runtime` | 12.3 sec |
| `epochs trained` | 8.0 |

### Comparison to Baselines

| Stage | eval_loss | perplexity | accuracy |
|---|---|---|---|
| Pretrained GPT-2 Large (no training) | ~3.77 | ~43.4 | — |
| Best sweep run (3 epochs, no save) | 2.25 | ~9.5 | ~0.52 |
| **Final run (8 epochs, saved)** | **2.23** | **9.32** | **0.597** |

**~4.5x perplexity improvement** over the pretrained baseline. The model has genuinely adapted to the prompt-tag distribution.

### Training Curve

The `eval/accuracy` curve showed classic logarithmic improvement:

- Steep gains in first ~200 steps (53% → 58%)
- Gradual climb 200–500 (58% → 59.5%)
- Plateau 500–860 (~59.7–59.8%, within eval noise)

Model converged cleanly. No overfitting, no instability. 8 epochs was slightly more than needed — gains past step 500 were marginal. Early stopping would have triggered around epoch 5–6.

## Qualitative Behavior

### Example 1: Anime character portrait

**Input:**
```
a character portrait of a woman with long silver hair and a black dress
```

**Output:**
```
a character portrait of a woman with long silver hair and a black dress = character portrait, woman, long silver hair, black dress, anime, by masamune shirow, by katsuhiro otomo<|endoftext|>
```

Model correctly extracted literal tags ("character portrait", "woman", "long silver hair", "black dress") and inferred stylistically appropriate aesthetic tags (anime artists Masamune Shirow and Katsuhiro Otomo).

### Example 2: Sparse photo prompt

**Input:**
```
beautiful photo of a cat
```

**Output:**
```
beautiful photo of a cat wearing a yellow hat and yellow dress, realistic, 4k, hd<|endoftext|>
```

Model embellished the sparse description with plausible details ("wearing a yellow hat and yellow dress") and selected photorealism style tags ("realistic, 4k, hd") instead of anime style — demonstrating conditional style inference based on medium ("photo" vs "portrait").

### Observed Behavior Patterns

1. **Style cluster inference**: model correctly maps subject + medium to appropriate aesthetic vocabulary (photo → photorealism, portrait → illustration/anime)
2. **Generative completion**: sparse inputs trigger description embellishment rather than literal tag extraction
3. **Clean stop tokens**: `<|endoftext|>` emitted at natural boundaries
4. **Artist association**: model learned named-artist style anchors from training data

## Sweep Decisions (Summary)

- **Bayesian optimization** with **hyperband early-termination** (eta=3, min_iter=20)
- **20 total runs**, `best_eval_loss` metric
- Search ranges tuned for continued pretraining (not fine-tuning, not from-scratch)
- Step-count math: with 1,723 training blocks, grad_accum tuned down to [2, 4, 8] to give 80–600 optimizer steps per run rather than 9
- Key code fixes during setup:
  - Removed manual `wandb.init()` at module top
  - Restored `--overwrite_output_dir` check to prevent ghost-resume from prior checkpoints
  - Added `WandbBestEvalCallback` to write `best_eval_loss` to run summary so hyperband-terminated runs populate sweep panels
  - Hardcoded boolean flags in `command:` section to avoid `--fp16=true` falling back to CPU autocast

## Next Steps / Open Questions

- **Generation controls**: lower temperature or include `=` delimiter to suppress description embellishment when literal tag extraction is desired
- **Held-out test set**: run final eval on a corpus not seen during sweep/training for an unbiased read
- **Data improvements likely higher leverage than further HP tuning**: deduplication, filtering, or more examples would shift the plateau higher; HPs are already near optimal
- **Stress-test prompts**: out-of-distribution subjects, format adherence with explicit `=`, style override cues, length boundaries
