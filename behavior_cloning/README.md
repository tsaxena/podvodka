# Step 1: Supervised Fine-Tuning

We fine-tune a `gpt2-large` in the following setting:

1. We constuct a dataset containig strings `image description</s>prompt<|endoftext|>`
2. We use a standard LM fine-tuning pipeline from the HuggingFace Transformers examples.

You can find the modified version of the fine-tuning script in the `run_clm.py` file.

For hyperparameter search we use W&B Sweep to find the best values of `learing_rate` and `weight_decay`. The Sweep's config is written in the `sweep.yml` file.

Finally, to reproduce the training with the best params, run
```bash
sh run_training.sh
```

We used a single NVIDIA A100 80 GB GPU, the full training takes roughly 90 mins.

## Fine-tuning Qwen2.5-7B

To fine-tune `Qwen/Qwen2.5-7B` instead of `gpt2-large`, run

```bash
sh run_training_qwen.sh
```

This uses the same `run_clm.py` script with the following key differences:
- Model: `Qwen/Qwen2.5-7B`
- dtype: `bfloat16`
- Gradient checkpointing enabled
- Output dir: `qwen2.5-7b-finetuned`
- Logs dir: `qwen2.5-7b-finetuned-log`
- Training logged to W&B (`--report_to wandb`)
