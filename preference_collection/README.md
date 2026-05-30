# Step 4 — DPO Preference Data Collection

> **Status:** 8,000 judgements collected (1,000 prompts × 8 candidates each).
> Multi-pair extraction complete; dataset ready for `train_dpo.py`.

---

## Why this step exists

The original PPO run and the first DPO run both used `toloka/prompts_reward_model`
as the judge. That RM was trained on aggregated human preferences and has a known
bias toward "professional-sounding" modifier stacks — artgerm, greg rutkowski,
octane render, 8k, trending on artstation — regardless of whether those modifiers
fit the base concept. Both trained models inherited that bias.

This step replaces the RM with an OpenRouter reasoning model as the judge. A
reasoning model can evaluate whether a style choice is *specific to the prompt*
versus generic magic-word stuffing, which requires the kind of multi-step
contextual inference that RM classifiers cannot do.

---

## Pipeline

```
train_strings.csv
    │
    ▼
[Phase 1] generate_candidates()
    │  SFT model: tsaxena/gpt2-large-prompt-tags
    │  8 completions per prompt, top-p sampling
    │  Output: sft_candidates.jsonl  (cache; skip with --skip_generation)
    │
    ▼
[Phase 2] score_all()  ── async, resumable ──────────────────────────────────┐
    │  Judge: OpenRouter reasoning model (default: deepseek/deepseek-r1)      │
    │  8 objectives scored 1–10 per (prompt, candidate)                       │
    │  Checkpoint: openrouter_scores_mo.jsonl  (auto-resumed on restart)      │
    │  8,000 API calls total for a 1k×8 run                                   │
    └────────────────────────────────────────────────────────────────────────┘
    │
    ▼
[Phase 3] build_pairs()
    │  Up to --max_pairs_per_prompt pairs per prompt (default 4)
    │  Round-robin allocation across prompts to --target_pairs (default 3000)
    │  Guardrails: per-objective minimums + hard failure mode blocklist
    │  Output: preferences_openrouter_mo.jsonl  (DPO-ready JSONL)
```

---

## Judging objectives

The judge scores 8 objectives per expansion (1–10 integers):

| Objective | What it catches |
|---|---|
| `fidelity` | Contradictions or subject drift from the base concept |
| `visual_specificity` | Vague or purely restatement expansions |
| `style_fit` | Random artist names / genres that don't match the subject |
| `composition_lighting` | Generic or absent camera / light / mood details |
| `non_genericness` | Reusable modifier tails that could apply to any prompt |
| `coherence` | Internal contradictions or repetition |
| `brevity_control` | Overlong modifier soup |
| `anti_magic_word_score` | artgerm / greg rutkowski / octane render stuffing |

Hard scoring rules are baked into the judge prompt:

- `fidelity ≤ 4` → `overall ≤ 5`
- `anti_magic_word_score ≤ 4` → `overall ≤ 5`
- Prompt restatement → `overall ≤ 6`
- Off-topic or contradiction → `overall ≤ 4`

Overall is a weighted average (fidelity 25%, visual_specificity 20%,
style_fit 15%, composition_lighting 15%, non_genericness 15%, coherence 10%)
before the hard caps are applied.

---

## Multi-pair extraction

Earlier versions of the script produced one pair per prompt (best vs. worst).
With 8 candidates per prompt there are up to C(8,2) = 28 valid ordered pairs;
the current script extracts up to `--max_pairs_per_prompt` high-confidence ones.

**Pair selection algorithm:**

1. Rank candidates highest → lowest by `overall`.
2. Walk (chosen tier, rejected tier) combinations in score order.
3. Accept the first valid `rejected` for each `chosen` that passes:
   - `score_gap ≥ --min_score_gap` (default 2)
   - chosen passes per-objective minimum thresholds
   - chosen has no hard failure modes (magic_words, off_topic, etc.)
   - chosen beats rejected on all four core objectives
     (fidelity, style_fit, non_genericness, anti_magic_word_score)
4. Stop at `--max_pairs_per_prompt` accepted pairs per prompt.

**Round-robin budget allocation** then picks one pair per prompt per pass
(highest-quality pair first) until `--target_pairs` is reached. This keeps
prompt diversity: a prompt with 4 valid pairs does not crowd out prompts with
only 1.

**Pair quality labels** in the output:

| Label | Meaning |
|---|---|
| `best_vs_worst` | Highest vs. lowest ranked candidate |
| `high_contrast` | Score gap ≥ 4 |
| `medium_contrast` | Score gap ≥ 2 |

---

## Output schema

Each line of `preferences_openrouter_mo.jsonl` is a JSON object:

```jsonc
{
  // DPO training fields (consumed by train_dpo.py)
  "prompt":   "a portrait of an astronaut on mars</s>",
  "chosen":   "<expansion text>",
  "rejected": "<expansion text>",

  // Legacy scalar fields (for quick grep / pandas filtering)
  "chosen_score":  8,
  "rejected_score": 4,
  "score_gap":      4,

  // Audit metadata
  "chosen_candidate_id":       0,
  "rejected_candidate_id":     5,
  "chosen_rank_by_overall":    0,   // 0 = best in this prompt's pool
  "rejected_rank_by_overall":  6,
  "pair_quality":              "best_vs_worst",

  // Full objective breakdowns
  "chosen_scores": {
    "overall": 8, "fidelity": 9, "visual_specificity": 8,
    "style_fit": 7, "composition_lighting": 8, "non_genericness": 7,
    "coherence": 9, "brevity_control": 8, "anti_magic_word_score": 9,
    "failure_modes": ["none"], "reason": "..."
  },
  "rejected_scores": { ... }
}
```

---

## Usage

**Re-run pair extraction only** (scores already collected, no new API calls):

```bash
python build_dpo_dataset_openrouter_multiobjective_updated.py \
    --num_prompts 1000 \
    --candidates_per_prompt 8 \
    --skip_generation \
    --target_pairs 3000 \
    --max_pairs_per_prompt 4 \
    --output_path /workspace/podvodka/data/preferences_openrouter_mo.jsonl
```

**Full run** (generate → score → pair):

```bash
export OPENROUTER_API_KEY=sk-or-...

python build_dpo_dataset_openrouter_multiobjective_updated.py \
    --num_prompts 1000 \
    --candidates_per_prompt 8 \
    --judge_model deepseek/deepseek-r1 \
    --max_concurrency 10 \
    --target_pairs 3000 \
    --max_pairs_per_prompt 4 \
    --audit_path /workspace/podvodka/data/audit_sample.jsonl \
    --output_path /workspace/podvodka/data/preferences_openrouter_mo.jsonl
```

**Key flags:**

| Flag | Default | Notes |
|---|---|---|
| `--judge_model` | `deepseek/deepseek-r1` | Any OpenRouter model slug; reasoning models recommended |
| `--candidates_per_prompt` | `8` | Must match the cache; validated on load |
| `--target_pairs` | `3000` | Hard cap; round-robin keeps diversity |
| `--max_pairs_per_prompt` | `4` | Max pairs extracted from one prompt |
| `--min_score_gap` | `2` | Raise to 3 for cleaner but fewer pairs |
| `--skip_generation` | off | Reuse `sft_candidates.jsonl`; safe after first run |
| `--audit_path` | None | Human-readable sample of first N pairs |

---

## Guardrail thresholds (defaults)

| Threshold | Default | Purpose |
|---|---|---|
| `--min_chosen_fidelity` | 6 | Reject expansions that drift from the concept |
| `--min_chosen_style_fit` | 5 | Reject random-artist stuffing in chosen |
| `--min_chosen_non_genericness` | 6 | Reject reusable modifier tails |
| `--min_chosen_anti_magic_word_score` | 7 | Core anti-hacking gate |
| `--min_chosen_coherence` | 6 | Reject incoherent/repetitive expansions |
| `--hard_failure_modes` | `magic_words, irrelevant_artist, off_topic, contradiction, incoherent` | Instant disqualification |

---

## Resumability

Scoring is the expensive phase (~8,000 API calls). The script checkpoints every
completed score to `openrouter_scores_mo.jsonl` and skips already-scored
`(prompt_id, candidate_id)` pairs on restart. Kill and re-run freely.

Pair extraction reads from the checkpoint and is instant — no API calls.

---

## Sanity checks before training

1. **Score distribution.** Run with `--audit_path` and inspect. Mean overall
   in 4–7 is healthy. Mean > 8.5 means the judge is over-rating; mean < 3
   means it is under-rating. A bottom- or top-heavy distribution gives DPO
   little contrast to learn from.

2. **Read 20–30 pairs from the audit file.** Check that chosen samples are
   genuinely better, not just longer or more verbose. Look for pairs where the
   rejected sample contains obvious magic-word stuffing and the chosen sample
   does not.

3. **Pair quality breakdown.** The script prints a breakdown of
   `best_vs_worst` / `high_contrast` / `medium_contrast` pairs. A healthy
   dataset has a mix; if everything is `medium_contrast` (gap = 2–3) the
   labeler may not be discriminating enough.

---

## Next step

```bash
python train_dpo.py \
    --dataset_path /workspace/podvodka/data/preferences_openrouter_mo.jsonl \
    --output_path /workspace/podvodka/models/gpt2-large-dpo-openrouter
```


# Original RM Preference Collection

This folder contains scripts for the preference collection for the reward model training.

1. Write prompts for training image descriptions using `write_prompts.py`
2. Generate images for prompts using `gen_images.py`
3. Upload images to S3 using `upload_images.py`
4. Create tasks for Toloka using `create_tasks.py`
5. Create a Toloka project using the interface from `interface.json` and instructions from `instructions.html`
6. Create a pool and upload golden tasks from `golden_tasks.tsv` and the created tasks



# LLM Judge Data Collection
```
python build_dpo_dataset_openrouter_multiobjective.py \
  --skip_generation \
  --num_prompts 1000 \
  --candidates_per_prompt 8 \
  --score_checkpoint /workspace/podvodka/data/openrouter_scores_mo.jsonl \
  --candidates_cache /workspace/podvodka/data/sft_candidates.jsonl \
  --output_path /workspace/podvodka/data/preferences_openrouter_mo_multi_pairs.jsonl \
  --audit_path /workspace/podvodka/data/preferences_openrouter_mo_multi_pairs_audit.txt \
  --target_pairs 3000 \
  --max_pairs_per_prompt 4 \
  --min_score_gap 1 \
  --min_chosen_fidelity 5 \
  --min_chosen_style_fit 4 \
  --min_chosen_non_genericness 5 \
  --min_chosen_anti_magic_word_score 6 \
  --min_chosen_coherence 5
  ```
