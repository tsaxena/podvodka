"""
Build a DPO preference dataset using OpenRouter (with a reasoning model) as the labeler.

Why a reasoning model?
  Judging "is this expansion appropriate for the base concept?" is genuinely a
  multi-step inference: the judge has to consider what the prompt depicts, what
  artistic/stylistic choices fit, and whether the proposed modifiers actually
  match — versus just being generic magic-word stuffing. Reasoning models do
  this kind of analysis substantially better than non-reasoning models, and on
  OpenRouter, DeepSeek R1 is cheap enough to use at scale (~$2-4 for 8000 calls).

Pipeline:
  1. Generate K candidate completions per prompt using the SFT model (HF).
  2. Score every (prompt, candidate) by asking an OpenRouter reasoning model.
  3. Pair the highest- and lowest-scoring candidates per prompt → DPO triples.
  4. Save to JSONL that train_dpo.py can consume.

Auth:
  Set OPENROUTER_API_KEY in your environment (sign up at https://openrouter.ai/),
  or pass --api_key on the command line.

Cost (rough estimates, for 1000 prompts × 8 candidates = 8000 calls):
  - deepseek/deepseek-r1        : ~$2-4   (recommended; cheap, strong reasoning)
  - openai/gpt-oss-120b          : ~$3-5   (open-weight reasoning model)
  - openai/o3-mini               : ~$15-30 (premium, fast)
  - anthropic/claude-sonnet-4.5  : ~$15-30 (with extended thinking)
  - openai/o1                    : ~$60+   (overkill for this task)

Resumability:
  Scores are appended to --score_checkpoint as they arrive. Interrupted runs
  resume from the last saved score.

Example
-------
python build_dpo_dataset_openrouter.py \\
    --num_prompts 1000 \\
    --candidates_per_prompt 8 \\
    --judge_model deepseek/deepseek-r1 \\
    --output_path /workspace/podvodka/data/preferences_openrouter.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import torch
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# JUDGE PROMPT
# ============================================================
# Explicitly calls out the failure modes observed in PPO and DPO runs so the
# judge actively penalizes them. Without these specifics, judges tend to
# over-reward "professional-looking" outputs — which is the original problem.

JUDGE_SYSTEM = (
    "You are an expert at evaluating stable-diffusion-style prompt expansions. "
    "You penalize generic 'magic word' stuffing and reward prompt-specific creativity. "
    "Reason carefully about whether each modifier genuinely fits the base concept "
    "before assigning a score."
)

JUDGE_TEMPLATE = """Rate the following stable-diffusion prompt expansion on a scale of 1 to 10.

A GOOD expansion (7-10):
- Adds vivid, specific visual details that fit the base concept
- Makes artistic or stylistic choices that genuinely match the subject
  (e.g., a Russian Impressionist painter for a Moscow scene, not random famous names)
- Uses diverse modifiers covering several dimensions (subject detail, composition,
  lighting, medium, mood, color) rather than stacking the same kind of word
- Is creative and varied without being incoherent

A MEDIOCRE expansion (4-6):
- Adds some detail but is generic
- Uses well-known SD modifiers in moderation and roughly fitting the subject
- Mostly just rephrases the prompt with some adjectives

A BAD expansion (1-3):
- Just repeats or restates the base concept without adding meaningful content
- Stacks generic "quality" tokens like "8k, octane render, unreal engine, \
trending on artstation, masterpiece, ultra detailed" without prompt-specific content
- Names famous artists ("art by artgerm and greg rutkowski", "wlop", etc.) \
that don't fit the subject — e.g., fantasy painters for a city street, or the \
same artists regardless of what's being depicted
- Is incoherent, repetitive, or off-topic
- Adds details that contradict or ignore the base concept

Base concept: {prompt}

Generated expansion: {expansion}

After your reasoning, respond with a JSON object on its own line:
{{"score": <integer 1-10>, "reason": "<one short sentence>"}}
"""


# ============================================================
# Data structures
# ============================================================

@dataclass
class ScoredItem:
    prompt_id: int
    candidate_id: int
    prompt: str
    expansion: str
    score: Optional[int]
    reason: str


# ============================================================
# Phase 1: generate candidates with the SFT model
# ============================================================

@torch.no_grad()
def generate_candidates(
    sft_model_path: str,
    prompts: List[str],
    k: int,
    max_new_tokens: int,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float16,
) -> List[List[str]]:
    print(f"[gen] loading SFT model from {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    out: List[List[str]] = []
    for prompt in tqdm(prompts, desc="generating candidates"):
        ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).input_ids.to(device)
        generated = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = [
            tokenizer.decode(g[ids.shape[1]:], skip_special_tokens=True).strip()
            for g in generated
        ]
        out.append(completions)

    del model
    torch.cuda.empty_cache()
    return out


# ============================================================
# Phase 2: score with OpenRouter (async, resumable)
# ============================================================

def parse_score_response(text: str) -> Tuple[Optional[int], str]:
    """Extract score and reason from a model response.
    Reasoning models often emit reasoning before the JSON; we hunt for the
    last JSON-looking blob in the response."""
    if not text:
        return None, "empty_response"
    text = text.strip()

    # Strip markdown fences anywhere in the response
    text_clean = re.sub(r"```(?:json)?\s*", "", text)
    text_clean = re.sub(r"```", "", text_clean)

    # Look for JSON object — prefer the LAST one (in case reasoning has intermediate JSON)
    json_matches = list(re.finditer(r"\{[^{}]*\}", text_clean))
    for match in reversed(json_matches):
        try:
            data = json.loads(match.group(0))
            if "score" in data:
                score = int(data["score"])
                if 1 <= score <= 10:
                    return score, str(data.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Fallback regex
    m = re.search(r'"?score"?\s*[:=]\s*(\d+)', text_clean)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 10:
            reason_m = re.search(r'"?reason"?\s*[:=]\s*"([^"]*)"', text_clean)
            return score, reason_m.group(1) if reason_m else ""

    return None, f"parse_failed: {text[:150]}"


async def score_one(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt_id: int,
    candidate_id: int,
    prompt: str,
    expansion: str,
    max_tokens: int,
    extra_body: Optional[dict] = None,
    max_retries: int = 4,
) -> ScoredItem:
    user_msg = JUDGE_TEMPLATE.format(prompt=prompt, expansion=expansion)

    async with semaphore:
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                if extra_body:
                    kwargs["extra_body"] = extra_body

                resp = await client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content
                score, reason = parse_score_response(text or "")
                return ScoredItem(prompt_id, candidate_id, prompt, expansion,
                                  score, reason)

            except (RateLimitError, APIConnectionError):
                await asyncio.sleep(2 ** attempt)
            except APIStatusError as e:
                # Retry on 5xx, give up on 4xx
                if e.status_code >= 500 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return ScoredItem(prompt_id, candidate_id, prompt, expansion,
                                      None, f"api_error_{e.status_code}: {e}")
            except Exception as e:
                return ScoredItem(prompt_id, candidate_id, prompt, expansion,
                                  None, f"error: {type(e).__name__}: {e}")

        return ScoredItem(prompt_id, candidate_id, prompt, expansion,
                          None, "max_retries_exceeded")


def load_existing_scores(path: Path) -> Dict[Tuple[int, int], ScoredItem]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[(d["prompt_id"], d["candidate_id"])] = ScoredItem(**d)
            except (json.JSONDecodeError, KeyError):
                continue
    return out


async def score_all(
    prompts: List[str],
    candidates: List[List[str]],
    model: str,
    api_key: Optional[str],
    base_url: str,
    max_concurrency: int,
    max_tokens: int,
    extra_body: Optional[dict],
    extra_headers: Optional[dict],
    score_checkpoint: Path,
) -> List[ScoredItem]:
    all_work: List[Tuple[int, int, str, str]] = []
    for pi, (p, comps) in enumerate(zip(prompts, candidates)):
        for ci, c in enumerate(comps):
            all_work.append((pi, ci, p, c))

    existing = load_existing_scores(score_checkpoint)
    todo = [w for w in all_work if (w[0], w[1]) not in existing]
    print(f"[score] {len(all_work)} total, "
          f"{len(existing)} already done, {len(todo)} to score")

    if not todo:
        return list(existing.values())

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=extra_headers or {},
    )
    semaphore = asyncio.Semaphore(max_concurrency)

    score_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(score_checkpoint, "a")

    tasks = [
        asyncio.create_task(score_one(
            client, semaphore, model, pi, ci, p, c,
            max_tokens=max_tokens, extra_body=extra_body,
        ))
        for pi, ci, p, c in todo
    ]

    pbar = tqdm(total=len(tasks), desc=f"scoring with {model}")
    fresh_results: List[ScoredItem] = []
    failures = 0
    for coro in asyncio.as_completed(tasks):
        item = await coro
        fresh_results.append(item)
        f_out.write(json.dumps(asdict(item)) + "\n")
        f_out.flush()
        if item.score is None:
            failures += 1
        pbar.update(1)
        pbar.set_postfix(failures=failures)

    pbar.close()
    f_out.close()
    print(f"[score] done. {failures} failures out of {len(todo)}")
    return list(existing.values()) + fresh_results


# ============================================================
# Phase 3: build DPO pairs
# ============================================================

def build_pairs(
    prompts: List[str],
    candidates: List[List[str]],
    scored: List[ScoredItem],
    min_score_gap: int,
) -> List[Dict]:
    scores_by_key: Dict[Tuple[int, int], int] = {}
    for s in scored:
        if s.score is not None:
            scores_by_key[(s.prompt_id, s.candidate_id)] = s.score

    pairs = []
    skipped_no_scores = 0
    skipped_small_gap = 0

    for pi, (prompt, comps) in enumerate(zip(prompts, candidates)):
        scored_cands = [
            (ci, scores_by_key.get((pi, ci))) for ci in range(len(comps))
        ]
        scored_cands = [(ci, s) for ci, s in scored_cands if s is not None]
        if len(scored_cands) < 2:
            skipped_no_scores += 1
            continue

        best_ci, best_s = max(scored_cands, key=lambda x: x[1])
        worst_ci, worst_s = min(scored_cands, key=lambda x: x[1])
        gap = best_s - worst_s

        if gap < min_score_gap:
            skipped_small_gap += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": comps[best_ci],
            "rejected": comps[worst_ci],
            "chosen_score": best_s,
            "rejected_score": worst_s,
            "score_gap": gap,
        })

    print(f"[pairs] built {len(pairs)} pairs")
    print(f"[pairs]   skipped {skipped_no_scores} prompts with < 2 valid scores")
    print(f"[pairs]   skipped {skipped_small_gap} prompts with gap < {min_score_gap}")
    return pairs


def write_pairs(pairs: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[pairs] wrote {len(pairs)} pairs to {path}")


def print_score_distribution(scored: List[ScoredItem]) -> None:
    valid = [s for s in scored if s.score is not None]
    if not valid:
        print("[diag] no valid scores — something is very wrong")
        return

    counts = [0] * 11
    for s in valid:
        counts[s.score] += 1
    print("\n[diag] score distribution:")
    max_c = max(counts) if max(counts) > 0 else 1
    for i in range(1, 11):
        bar = "█" * (counts[i] * 40 // max_c)
        print(f"  {i:2d}: {counts[i]:5d}  {bar}")
    mean = sum(s.score for s in valid) / len(valid)
    print(f"[diag] mean score: {mean:.2f}, n={len(valid)}")

    if mean > 8.5:
        print("[diag] WARNING: judge is over-rating — tighten the prompt.")
    if mean < 3:
        print("[diag] WARNING: judge is under-rating — check prompt fairness.")
    bottom_heavy = (counts[1] + counts[2] + counts[3]) > 0.7 * len(valid)
    top_heavy = (counts[8] + counts[9] + counts[10]) > 0.7 * len(valid)
    if bottom_heavy or top_heavy:
        print("[diag] WARNING: score distribution is one-sided — "
              "DPO learns from contrast.")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()

    # Models / data
    p.add_argument("--sft_model", default="tsaxena/gpt2-large-prompt-tags")
    p.add_argument("--train_path",
                   default="/workspace/podvodka/data/train_strings.csv")
    p.add_argument("--num_prompts", type=int, default=1000)
    p.add_argument("--candidates_per_prompt", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=80)

    # OpenRouter
    p.add_argument("--judge_model", default="deepseek/deepseek-r1",
                   help="OpenRouter model slug. Reasoning models recommended. "
                        "Examples: deepseek/deepseek-r1, openai/o3-mini, "
                        "anthropic/claude-sonnet-4.5, openai/gpt-oss-120b")
    p.add_argument("--api_key", default=None,
                   help="Defaults to OPENROUTER_API_KEY env var.")
    p.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    p.add_argument("--max_concurrency", type=int, default=10)
    p.add_argument("--max_tokens", type=int, default=2000,
                   help="Max output tokens. Reasoning models need more headroom "
                        "for hidden reasoning + the final JSON.")
    p.add_argument("--reasoning_effort", default=None,
                   choices=[None, "low", "medium", "high"],
                   help="For OpenAI o-series via OpenRouter. Ignored by other models.")

    # Optional OpenRouter analytics headers
    p.add_argument("--site_url", default=None,
                   help="Optional: HTTP-Referer header for OpenRouter analytics.")
    p.add_argument("--app_name", default="dpo-dataset-builder",
                   help="Optional: X-Title header for OpenRouter analytics.")

    # Output
    p.add_argument("--output_path",
                   default="/workspace/podvodka/data/preferences_openrouter.jsonl")
    p.add_argument("--score_checkpoint",
                   default="/workspace/podvodka/data/openrouter_scores.jsonl")
    p.add_argument("--candidates_cache",
                   default="/workspace/podvodka/data/sft_candidates.jsonl")

    # Pairing
    p.add_argument("--min_score_gap", type=int, default=2)
    p.add_argument("--skip_generation", action="store_true")

    args = p.parse_args()

    # ---- Load prompts ----
    train_df = pd.read_csv(args.train_path)
    prompts = [l.split("</s>")[0] + "</s>" for l in train_df["text"]]
    prompts = prompts[:args.num_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.train_path}")

    # ---- Phase 1: candidates ----
    cand_path = Path(args.candidates_cache)
    candidates = None
    if args.skip_generation or cand_path.exists():
        print(f"[gen] loading cached candidates from {cand_path}")
        candidates = []
        with open(cand_path) as f:
            for line in f:
                candidates.append(json.loads(line)["candidates"])
        if len(candidates) < len(prompts):
            print(f"[gen] cache has {len(candidates)}, need {len(prompts)} — regenerating")
            candidates = None

    if candidates is None:
        candidates = generate_candidates(
            args.sft_model, prompts,
            k=args.candidates_per_prompt,
            max_new_tokens=args.max_new_tokens,
        )
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cand_path, "w") as f:
            for p_text, comps in zip(prompts, candidates):
                f.write(json.dumps({"prompt": p_text, "candidates": comps}) + "\n")
        print(f"[gen] wrote candidates to {cand_path}")

    # ---- Build OpenRouter request extras ----
    extra_body = None
    if args.reasoning_effort is not None:
        # OpenRouter standardized reasoning parameter
        extra_body = {"reasoning": {"effort": args.reasoning_effort}}

    extra_headers = {}
    if args.site_url:
        extra_headers["HTTP-Referer"] = args.site_url
    if args.app_name:
        extra_headers["X-Title"] = args.app_name

    total_calls = sum(len(c) for c in candidates)
    print(f"\nAbout to call OpenRouter {total_calls} times "
          f"({len(prompts)} prompts × {args.candidates_per_prompt} candidates).")
    print(f"Model: {args.judge_model}")
    print(f"Concurrency: {args.max_concurrency}")
    print(f"Max output tokens: {args.max_tokens}")
    if extra_body:
        print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Existing checkpoint: {args.score_checkpoint}")
    print()

    # ---- Phase 2: score ----
    t0 = time.time()
    scored = asyncio.run(score_all(
        prompts, candidates,
        model=args.judge_model,
        api_key=args.api_key or os.environ.get("OPENROUTER_API_KEY"),
        base_url=args.base_url,
        max_concurrency=args.max_concurrency,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        extra_headers=extra_headers,
        score_checkpoint=Path(args.score_checkpoint),
    ))
    elapsed = time.time() - t0
    print(f"[score] phase took {elapsed:.1f}s "
          f"({elapsed / max(total_calls, 1):.2f}s/call average)")

    print_score_distribution(scored)

    # ---- Phase 3: pairs ----
    pairs = build_pairs(prompts, candidates, scored, args.min_score_gap)
    write_pairs(pairs, Path(args.output_path))

    print(f"\nNext step:")
    print(f"  python train_dpo.py --dataset_path {args.output_path}")


if __name__ == "__main__":
    main()