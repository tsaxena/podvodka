"""
Build a DPO preference dataset using OpenRouter (with a reasoning model) as the labeler.

This version uses multi-objective judging instead of a single opaque score.
The judge returns separate scores for fidelity, visual specificity, style fit,
composition/lighting, non-genericness, coherence, brevity control, and
anti-magic-word behavior. The script still emits normal DPO triples
(prompt/chosen/rejected), so it remains compatible with train_dpo.py, but it
also stores the objective-level scores for auditing and filtering.

Why a reasoning model?
  Judging "is this expansion appropriate for the base concept?" is genuinely a
  multi-step inference: the judge has to consider what the prompt depicts, what
  artistic/stylistic choices fit, and whether the proposed modifiers actually
  match — versus just being generic magic-word stuffing. Reasoning models do
  this kind of analysis substantially better than non-reasoning models.

Pipeline:
  1. Generate K candidate completions per prompt using the SFT model (HF).
  2. Score every (prompt, candidate) by asking an OpenRouter reasoning model.
  3. Pair the highest- and lowest-scoring candidates per prompt, subject to
     objective-level guardrails.
  4. Save JSONL that train_dpo.py can consume, with extra score metadata.

Auth:
  Set OPENROUTER_API_KEY in your environment, or pass --api_key.

Example
-------
python build_dpo_dataset_openrouter_multiobjective.py \
    --num_prompts 1000 \
    --candidates_per_prompt 8 \
    --judge_model deepseek/deepseek-r1 \
    --output_path /workspace/podvodka/data/preferences_openrouter_mo.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, RateLimitError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Judge schema / constants
# ============================================================

OBJECTIVE_FIELDS = [
    "fidelity",
    "visual_specificity",
    "style_fit",
    "composition_lighting",
    "non_genericness",
    "coherence",
    "brevity_control",
    "anti_magic_word_score",
]

# These are failure modes we explicitly want to track because they map to the
# observed PPO/DPO reward-hacking patterns.
ALLOWED_FAILURE_MODES = {
    "none",
    "magic_words",
    "irrelevant_artist",
    "restatement",
    "off_topic",
    "contradiction",
    "too_generic",
    "too_verbose",
    "incoherent",
}

DEFAULT_HARD_FAILURE_MODES = [
    "magic_words",
    "irrelevant_artist",
    "off_topic",
    "contradiction",
    "incoherent",
]


# ============================================================
# JUDGE PROMPT
# ============================================================
# Explicitly calls out the failure modes observed in PPO and DPO runs so the
# judge actively penalizes them. Without these specifics, judges tend to
# over-reward "professional-looking" outputs — which is the original problem.

JUDGE_SYSTEM = (
    "You are an expert at evaluating stable-diffusion-style prompt expansions. "
    "You reward prompt-specific visual usefulness and penalize generic magic-word "
    "stuffing. Judge the expansion for this specific base concept, not as a "
    "generic aesthetic phrase. Return machine-parseable JSON only."
)

JUDGE_TEMPLATE = """Evaluate the following stable-diffusion prompt expansion.

Base concept:
{prompt}

Generated expansion:
{expansion}

Score each criterion from 1 to 10:

1. fidelity:
   Does the expansion preserve the base concept without contradicting it?
   Low score if it changes the subject, adds incompatible details, or ignores the concept.

2. visual_specificity:
   Does it add concrete, imageable details beyond restating the prompt?
   Low score if it is vague, generic, or only repeats the base concept.

3. style_fit:
   Do artistic/stylistic choices fit the subject and context?
   Low score for random famous artists, random genres, or styles that do not fit.

4. composition_lighting:
   Does it add useful composition, camera, lighting, color, mood, or medium details?
   Low score if these are absent or generic boilerplate.

5. non_genericness:
   Is it specific to this prompt rather than a reusable modifier tail?
   Low score for generic stacks like "8k, artstation, octane render, unreal engine,
   masterpiece, ultra detailed, cinematic lighting" when they are not justified.

6. coherence:
   Is the prompt coherent, non-repetitive, and internally consistent?

7. brevity_control:
   Is it concise enough to be usable, without overlong modifier soup?
   A good score does not require being short; it requires avoiding bloat.

8. anti_magic_word_score:
   High score if it avoids reward-hacking tokens and irrelevant artist-name stuffing.
   Low score if it uses generic "quality" tokens or names artists like artgerm,
   greg rutkowski, wlop, etc. without a subject-specific reason.

Hard scoring rules:
- If fidelity <= 4, overall must be <= 5.
- If anti_magic_word_score <= 4, overall must be <= 5.
- If the expansion mostly restates the prompt, overall must be <= 6.
- If it includes irrelevant famous artists, overall must be <= 5.
- If it is off-topic or contradicts the base concept, overall must be <= 4.

Compute overall using this weighting, then apply the hard scoring rules:
  overall = 0.25*fidelity
          + 0.20*visual_specificity
          + 0.15*style_fit
          + 0.15*composition_lighting
          + 0.15*non_genericness
          + 0.10*coherence

Return JSON only, with this exact schema:
{{
  "fidelity": <integer 1-10>,
  "visual_specificity": <integer 1-10>,
  "style_fit": <integer 1-10>,
  "composition_lighting": <integer 1-10>,
  "non_genericness": <integer 1-10>,
  "coherence": <integer 1-10>,
  "brevity_control": <integer 1-10>,
  "anti_magic_word_score": <integer 1-10>,
  "overall": <integer 1-10>,
  "failure_modes": [
    "none" | "magic_words" | "irrelevant_artist" | "restatement" |
    "off_topic" | "contradiction" | "too_generic" | "too_verbose" |
    "incoherent"
  ],
  "reason": "<one short sentence>"
}}
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

    # `score` is kept as an alias for overall so older downstream code and
    # diagnostics that expect `score` continue to work.
    score: Optional[int] = None
    overall: Optional[int] = None

    fidelity: Optional[int] = None
    visual_specificity: Optional[int] = None
    style_fit: Optional[int] = None
    composition_lighting: Optional[int] = None
    non_genericness: Optional[int] = None
    coherence: Optional[int] = None
    brevity_control: Optional[int] = None
    anti_magic_word_score: Optional[int] = None

    failure_modes: List[str] = field(default_factory=list)
    reason: str = ""


def _as_int_1_to_10(value: Any) -> Optional[int]:
    try:
        # Handles 8, "8", and 8.0. It intentionally rejects strings like "8/10".
        if isinstance(value, str):
            value = value.strip()
        score = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 10:
        return score
    return None


def _normalize_failure_modes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        return []

    modes = []
    for item in raw:
        if not isinstance(item, str):
            continue
        normalized = item.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in ALLOWED_FAILURE_MODES and normalized not in modes:
            modes.append(normalized)
    if not modes:
        return []
    if "none" in modes and len(modes) > 1:
        modes = [m for m in modes if m != "none"]
    return modes


def _extract_json_objects(text: str) -> List[dict]:
    """Extract candidate JSON objects from a model response.

    Reasoning models sometimes emit prose before the final JSON. This function
    tries a decoder-based scan first, then falls back to regex-based extraction.
    """
    objects = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue

    if objects:
        return objects

    # Simpler fallback for responses with no nested objects. Arrays are OK.
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


def parse_score_response(text: str) -> ScoredItem:
    """Parse a multi-objective judge response into a ScoredItem skeleton.

    The caller fills prompt_id/candidate_id/prompt/expansion. Returning a
    ScoredItem keeps parse failures representable as score=None.
    """
    if not text:
        return ScoredItem(-1, -1, "", "", reason="empty_response")

    text = text.strip()
    text_clean = re.sub(r"```(?:json)?\s*", "", text)
    text_clean = re.sub(r"```", "", text_clean)

    # Prefer the last valid JSON-looking object, because some reasoning models
    # emit exploratory JSON before the final answer.
    for data in reversed(_extract_json_objects(text_clean)):
        parsed: Dict[str, Any] = {}

        # Backward compatibility: accept either `overall` or old `score`.
        overall = _as_int_1_to_10(data.get("overall"))
        if overall is None:
            overall = _as_int_1_to_10(data.get("score"))
        if overall is None:
            continue

        parsed["score"] = overall
        parsed["overall"] = overall
        for field_name in OBJECTIVE_FIELDS:
            parsed[field_name] = _as_int_1_to_10(data.get(field_name))

        parsed["failure_modes"] = _normalize_failure_modes(data.get("failure_modes"))
        parsed["reason"] = str(data.get("reason", ""))
        return ScoredItem(-1, -1, "", "", **parsed)

    # Fallback regex for old/simple responses.
    m = re.search(r'"?(?:overall|score)"?\s*[:=]\s*(\d+)', text_clean)
    if m:
        score = _as_int_1_to_10(m.group(1))
        if score is not None:
            reason_m = re.search(r'"?reason"?\s*[:=]\s*"([^"]*)"', text_clean)
            return ScoredItem(
                -1,
                -1,
                "",
                "",
                score=score,
                overall=score,
                reason=reason_m.group(1) if reason_m else "",
            )

    return ScoredItem(-1, -1, "", "", reason=f"parse_failed: {text[:150]}")


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
                item = parse_score_response(text or "")
                item.prompt_id = prompt_id
                item.candidate_id = candidate_id
                item.prompt = prompt
                item.expansion = expansion
                return item

            except (RateLimitError, APIConnectionError):
                await asyncio.sleep(2 ** attempt)
            except APIStatusError as e:
                # Retry on 5xx, give up on 4xx
                if e.status_code >= 500 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return ScoredItem(
                        prompt_id,
                        candidate_id,
                        prompt,
                        expansion,
                        reason=f"api_error_{e.status_code}: {e}",
                    )
            except Exception as e:
                return ScoredItem(
                    prompt_id,
                    candidate_id,
                    prompt,
                    expansion,
                    reason=f"error: {type(e).__name__}: {e}",
                )

        return ScoredItem(
            prompt_id,
            candidate_id,
            prompt,
            expansion,
            reason="max_retries_exceeded",
        )


def scored_item_from_dict(d: dict) -> ScoredItem:
    """Load old or new checkpoint rows robustly."""
    allowed = {f.name for f in fields(ScoredItem)}
    clean = {k: v for k, v in d.items() if k in allowed}

    # Required fields for older rows should exist, but make loading tolerant.
    clean.setdefault("prompt_id", -1)
    clean.setdefault("candidate_id", -1)
    clean.setdefault("prompt", "")
    clean.setdefault("expansion", "")
    clean.setdefault("reason", "")

    # Backward compatibility with old schema: score existed, overall did not.
    if clean.get("overall") is None and clean.get("score") is not None:
        clean["overall"] = clean["score"]
    if clean.get("score") is None and clean.get("overall") is not None:
        clean["score"] = clean["overall"]

    clean["failure_modes"] = _normalize_failure_modes(clean.get("failure_modes"))
    return ScoredItem(**clean)


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
                item = scored_item_from_dict(d)
                out[(item.prompt_id, item.candidate_id)] = item
            except (json.JSONDecodeError, KeyError, TypeError):
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

def _item_score(item: ScoredItem) -> Optional[int]:
    return item.overall if item.overall is not None else item.score


def _passes_chosen_guardrails(
    item: ScoredItem,
    min_chosen_fidelity: int,
    min_chosen_style_fit: int,
    min_chosen_non_genericness: int,
    min_chosen_anti_magic_word_score: int,
    min_chosen_coherence: int,
    hard_failure_modes: List[str],
) -> Tuple[bool, str]:
    checks = [
        ("fidelity", item.fidelity, min_chosen_fidelity),
        ("style_fit", item.style_fit, min_chosen_style_fit),
        ("non_genericness", item.non_genericness, min_chosen_non_genericness),
        (
            "anti_magic_word_score",
            item.anti_magic_word_score,
            min_chosen_anti_magic_word_score,
        ),
        ("coherence", item.coherence, min_chosen_coherence),
    ]
    for name, value, threshold in checks:
        # If old checkpoint rows do not have sub-scores, do not fail the row on
        # missing data. New rows should have these values.
        if value is not None and value < threshold:
            return False, f"chosen_{name}_below_{threshold}"

    item_modes = set(item.failure_modes or [])
    hard_modes = set(hard_failure_modes or [])
    bad_modes = sorted(item_modes.intersection(hard_modes))
    if bad_modes:
        return False, "chosen_hard_failure_" + "+".join(bad_modes)

    return True, ""


def _candidate_beats_on_core_objectives(chosen: ScoredItem, rejected: ScoredItem) -> Tuple[bool, str]:
    """Avoid selecting a candidate that wins overall while losing core objectives.

    We allow missing objective fields for backward compatibility. For new rows,
    this prevents a candidate from winning because it is pretty but less faithful
    or more generic than the rejected candidate.
    """
    core_fields = [
        "fidelity",
        "style_fit",
        "non_genericness",
        "anti_magic_word_score",
    ]
    for name in core_fields:
        chosen_value = getattr(chosen, name)
        rejected_value = getattr(rejected, name)
        if chosen_value is not None and rejected_value is not None:
            if chosen_value < rejected_value:
                return False, f"chosen_loses_{name}"
    return True, ""


def _score_payload(item: ScoredItem) -> Dict[str, Any]:
    return {
        "overall": _item_score(item),
        "fidelity": item.fidelity,
        "visual_specificity": item.visual_specificity,
        "style_fit": item.style_fit,
        "composition_lighting": item.composition_lighting,
        "non_genericness": item.non_genericness,
        "coherence": item.coherence,
        "brevity_control": item.brevity_control,
        "anti_magic_word_score": item.anti_magic_word_score,
        "failure_modes": item.failure_modes,
        "reason": item.reason,
    }


def build_pairs(
    prompts: List[str],
    candidates: List[List[str]],
    scored: List[ScoredItem],
    min_score_gap: int,
    min_chosen_fidelity: int,
    min_chosen_style_fit: int,
    min_chosen_non_genericness: int,
    min_chosen_anti_magic_word_score: int,
    min_chosen_coherence: int,
    hard_failure_modes: List[str],
) -> List[Dict]:
    items_by_key: Dict[Tuple[int, int], ScoredItem] = {}
    for s in scored:
        if _item_score(s) is not None:
            items_by_key[(s.prompt_id, s.candidate_id)] = s

    pairs = []
    skipped_no_scores = 0
    skipped_small_gap = 0
    skipped_guardrail = 0
    skipped_core_loss = 0
    skip_reasons: Dict[str, int] = {}

    for pi, (prompt, comps) in enumerate(zip(prompts, candidates)):
        scored_cands = [
            (ci, items_by_key.get((pi, ci))) for ci in range(len(comps))
        ]
        scored_cands = [(ci, item) for ci, item in scored_cands if item is not None]
        if len(scored_cands) < 2:
            skipped_no_scores += 1
            continue

        best_ci, best_item = max(scored_cands, key=lambda x: _item_score(x[1]) or -1)
        worst_ci, worst_item = min(scored_cands, key=lambda x: _item_score(x[1]) or 999)
        best_score = _item_score(best_item)
        worst_score = _item_score(worst_item)
        if best_score is None or worst_score is None:
            skipped_no_scores += 1
            continue

        gap = best_score - worst_score
        if gap < min_score_gap:
            skipped_small_gap += 1
            continue

        ok, reason = _passes_chosen_guardrails(
            best_item,
            min_chosen_fidelity=min_chosen_fidelity,
            min_chosen_style_fit=min_chosen_style_fit,
            min_chosen_non_genericness=min_chosen_non_genericness,
            min_chosen_anti_magic_word_score=min_chosen_anti_magic_word_score,
            min_chosen_coherence=min_chosen_coherence,
            hard_failure_modes=hard_failure_modes,
        )
        if not ok:
            skipped_guardrail += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        ok, reason = _candidate_beats_on_core_objectives(best_item, worst_item)
        if not ok:
            skipped_core_loss += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": comps[best_ci],
            "rejected": comps[worst_ci],

            # Keep legacy fields for train_dpo.py compatibility and quick audits.
            "chosen_score": best_score,
            "rejected_score": worst_score,
            "score_gap": gap,

            # New audit metadata.
            "chosen_candidate_id": best_ci,
            "rejected_candidate_id": worst_ci,
            "chosen_scores": _score_payload(best_item),
            "rejected_scores": _score_payload(worst_item),
        })

    print(f"[pairs] built {len(pairs)} pairs")
    print(f"[pairs]   skipped {skipped_no_scores} prompts with < 2 valid scores")
    print(f"[pairs]   skipped {skipped_small_gap} prompts with gap < {min_score_gap}")
    print(f"[pairs]   skipped {skipped_guardrail} prompts where best candidate failed guardrails")
    print(f"[pairs]   skipped {skipped_core_loss} prompts where chosen lost a core objective")
    if skip_reasons:
        print("[pairs] skip reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    return pairs


def write_pairs(pairs: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"[pairs] wrote {len(pairs)} pairs to {path}")


def write_audit_sample(pairs: List[Dict], path: Optional[Path], n: int) -> None:
    if path is None or n <= 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs[:n]:
            f.write(json.dumps(pair, indent=2) + "\n\n")
    print(f"[audit] wrote first {min(n, len(pairs))} pairs to {path}")


def print_single_score_distribution(scored: List[ScoredItem], field_name: str) -> None:
    values = [getattr(s, field_name) for s in scored if getattr(s, field_name) is not None]
    if not values:
        print(f"[diag] no valid {field_name} scores")
        return

    counts = [0] * 11
    for value in values:
        counts[value] += 1
    print(f"\n[diag] {field_name} distribution:")
    max_c = max(counts) if max(counts) > 0 else 1
    for i in range(1, 11):
        bar = "█" * (counts[i] * 40 // max_c)
        print(f"  {i:2d}: {counts[i]:5d}  {bar}")
    mean = sum(values) / len(values)
    print(f"[diag] {field_name} mean: {mean:.2f}, n={len(values)}")

    if field_name in {"score", "overall"}:
        if mean > 8.5:
            print("[diag] WARNING: judge is over-rating — tighten the prompt.")
        if mean < 3:
            print("[diag] WARNING: judge is under-rating — check prompt fairness.")
        bottom_heavy = (counts[1] + counts[2] + counts[3]) > 0.7 * len(values)
        top_heavy = (counts[8] + counts[9] + counts[10]) > 0.7 * len(values)
        if bottom_heavy or top_heavy:
            print("[diag] WARNING: score distribution is one-sided — "
                  "DPO learns from contrast.")


def print_score_distribution(scored: List[ScoredItem]) -> None:
    valid = [s for s in scored if _item_score(s) is not None]
    if not valid:
        print("[diag] no valid scores — something is very wrong")
        return

    print_single_score_distribution(valid, "overall")

    print("\n[diag] objective means:")
    for field_name in OBJECTIVE_FIELDS:
        values = [getattr(s, field_name) for s in valid if getattr(s, field_name) is not None]
        if values:
            print(f"  {field_name:24s}: {sum(values) / len(values):.2f}  n={len(values)}")
        else:
            print(f"  {field_name:24s}: missing")

    failure_counts: Dict[str, int] = {}
    for s in valid:
        modes = s.failure_modes or ["none"]
        for mode in modes:
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
    print("\n[diag] failure modes:")
    for mode, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode:20s}: {count}")


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
                   default="/workspace/podvodka/data/preferences_openrouter_mo.jsonl")
    p.add_argument("--score_checkpoint",
                   default="/workspace/podvodka/data/openrouter_scores_mo.jsonl")
    p.add_argument("--candidates_cache",
                   default="/workspace/podvodka/data/sft_candidates.jsonl")
    p.add_argument("--audit_path", default=None,
                   help="Optional path for a human-readable sample of generated pairs.")
    p.add_argument("--audit_n", type=int, default=50)

    # Pairing / guardrails
    p.add_argument("--min_score_gap", type=int, default=2)
    p.add_argument("--min_chosen_fidelity", type=int, default=6)
    p.add_argument("--min_chosen_style_fit", type=int, default=5)
    p.add_argument("--min_chosen_non_genericness", type=int, default=6)
    p.add_argument("--min_chosen_anti_magic_word_score", type=int, default=7)
    p.add_argument("--min_chosen_coherence", type=int, default=6)
    p.add_argument("--hard_failure_modes", default=",".join(DEFAULT_HARD_FAILURE_MODES),
                   help="Comma-separated failure modes that disqualify a chosen candidate.")
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
    hard_failure_modes = [
        mode.strip().lower().replace(" ", "_").replace("-", "_")
        for mode in args.hard_failure_modes.split(",")
        if mode.strip()
    ]
    pairs = build_pairs(
        prompts,
        candidates,
        scored,
        min_score_gap=args.min_score_gap,
        min_chosen_fidelity=args.min_chosen_fidelity,
        min_chosen_style_fit=args.min_chosen_style_fit,
        min_chosen_non_genericness=args.min_chosen_non_genericness,
        min_chosen_anti_magic_word_score=args.min_chosen_anti_magic_word_score,
        min_chosen_coherence=args.min_chosen_coherence,
        hard_failure_modes=hard_failure_modes,
    )
    write_pairs(pairs, Path(args.output_path))
    write_audit_sample(pairs, Path(args.audit_path) if args.audit_path else None, args.audit_n)

    print(f"\nNext step:")
    print(f"  python train_dpo.py --dataset_path {args.output_path}")
    print("\nRecommended sanity check before training:")
    print("  Manually inspect --audit_path output or the first 50 JSONL rows.")
    print("  Confirm chosen samples are genuinely better than rejected samples.")


if __name__ == "__main__":
    main()
