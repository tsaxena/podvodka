# Eval

DSG (Davidsonian Scene Graph) + HPSv2 evaluation for text-to-image fidelity and human preference scoring.

## Setup

```bash
export OPENROUTER_API_KEY=sk-or-...
pip install hpsv2
```

## Run DSG on DPO model

Enrich the prompt with the DPO-fine-tuned model (`tsaxena/gpt2-large-dpo-corrected`) before image generation:

```bash
python eval/dsg.py --prompt "a red car next to a blue bicycle" \
    --enrich-method dpo \
    --enrich-device cpu \
    --enrich-max-tokens 80
```

To evaluate a DPO checkpoint over the 200 DrawBench prompts:

```bash
python eval/dsg.py --checkpoint-path /path/to/dpo-checkpoint \
    --step-label dpo_v1 \
    --num-samples 2 \
    --output dsg_hps_results.json
```

## Other enrichment methods

| `--enrich-method` | Model |
|---|---|
| `base` | `tsaxena/gpt2-large-prompt-tags` (SFT baseline) |
| `dpo` | `tsaxena/gpt2-large-dpo-corrected` (DPO-optimised) |
| `ppo` | `tsaxena/gpt2-large-ppo-prompt-tags` (PPO-optimised) |

See `dsg.py` for full usage.
