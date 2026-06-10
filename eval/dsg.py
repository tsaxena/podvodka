"""
Davidsonian Scene Graph (DSG) - Text-to-Image Fidelity Evaluation
Based on: https://github.com/j-min/DSG  (ICLR 2024)

Models:
  Image : runwayml/stable-diffusion-v1-5 (local, via diffusers)
  LLM   : any text model  (default: google/gemma-3-27b-it, via OpenRouter)
  VQA   : qwen/qwen3-vl-32b-instruct  (via OpenRouter)

Pipeline:
  0. Stable Diffusion generates an image from the prompt (if --image not given)
  1. LLM generates skill-specific tuples from the prompt
  2. LLM infers dependency graph between tuples
  3. LLM rewrites each tuple as a natural-language yes/no question
  4. Qwen VL answers each question given the generated image
  5. Dependency-aware scoring: skip child questions if parent fails
  6. Report stated_fidelity / implied_coherence / invented_rate + overall DSG score

Hallucination classification:
  stated   - explicitly in the prompt          → must be present (penalise if missing)
  implied  - strongly implied by the prompt    → desirable if present
  invented - not stated or implied             → undesirable if present (penalised)

Usage:
    export OPENROUTER_API_KEY=sk-or-...

    # Generate image from prompt, then evaluate
    python dsg.py --prompt "a red car next to a blue bicycle"

    # Use an existing image
    python dsg.py --prompt "a red car next to a blue bicycle" --image out.jpg

    # Batch (CSV columns: prompt, image_path — leave image_path blank to auto-generate)
    python dsg.py --csv prompts.csv --output results.json

    # Override models / device
    python dsg.py --prompt "..." \\
        --sd-model "runwayml/stable-diffusion-v1-5" \\
        --device cuda \\
        --llm-model "anthropic/claude-3.5-haiku" \\
        --vqa-model "qwen/qwen3-vl-32b-instruct"
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = "google/gemma-3-27b-it"
DEFAULT_VQA_MODEL = "qwen/qwen3-vl-32b-instruct"
DEFAULT_SD_MODEL  = "runwayml/stable-diffusion-v1-5"
DEFAULT_DEVICE    = "cuda"
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Stable Diffusion image generation
# ---------------------------------------------------------------------------

_sd_pipe = None  # cached pipeline — loaded once per process


def _load_sd_pipeline(model_name: str, device: str):
    global _sd_pipe
    if _sd_pipe is not None:
        return _sd_pipe
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        sys.exit("Error: diffusers and torch are required for image generation.\n"
                 "Install with: pip install diffusers torch accelerate")

    print(f"[SD] Loading {model_name} on {device} …")
    _sd_pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    return _sd_pipe


def generate_sd_image(prompt: str, save_path: str,
                      model_name: str = DEFAULT_SD_MODEL,
                      device: str = DEFAULT_DEVICE) -> str:
    """Generate an image with Stable Diffusion and save it to save_path."""
    pipe = _load_sd_pipeline(model_name, device)
    print(f"[SD] Generating: {prompt!r}")
    image = pipe(prompt).images[0]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    print(f"[SD] Saved to {save_path}")
    return save_path

# ---------------------------------------------------------------------------
# LLM prompts (adapted from DSG paper / repo)
# ---------------------------------------------------------------------------

PROMPT_TUPLE = """\
Task: given an input prompt, decompose the scene into skill-specific semantic tuples.
Classify each tuple by how it relates to the prompt.

Skill categories:
  entity    : (entity_id, entity_name)
  attribute : (attribute_id, attribute_type, value, entity_id)
  relation  : (relation_id, relation_type, entity_id_1, entity_id_2)
  count     : (count_id, number, entity_id)
  global    : (global_id, global_type, value)   # e.g. time-of-day, weather, style

Grounding types:
  "stated"   - explicitly mentioned in the prompt
  "implied"  - not stated but strongly implied (e.g. "park" implies grass/trees/sky)
  "invented" - present in typical scenes but NOT implied by this specific prompt

Rules:
- Assign unique IDs like E1, A1, R1, C1, G1, R1 ...
- One fact per tuple; keep each tuple atomic.
- Include stated AND implied tuples; also add 2-3 likely invented ones to test the model.
- If the prompt mentions two or more entities that could be visually confused or merged
  (e.g. baby + doll, man + statue, dog + stuffed animal, person + mannequin), add a
  "stated" relation tuple with relation_type "distinct" to check they are rendered as
  separate objects. Use the existing entity IDs as arguments.

Input prompt: {prompt}

Output the tuples as a JSON list, e.g.:
[
  {{"id": "E1", "skill": "entity",    "args": ["E1", "car"],                "grounding": "stated"}},
  {{"id": "A1", "skill": "attribute", "args": ["A1", "color", "red", "E1"], "grounding": "stated"}},
  {{"id": "E2", "skill": "entity",    "args": ["E2", "road"],               "grounding": "implied"}},
  {{"id": "E3", "skill": "entity",    "args": ["E3", "cat"],                "grounding": "invented"}},
  {{"id": "R1", "skill": "relation",  "args": ["R1", "distinct", "E_baby", "E_doll"], "grounding": "stated"}},
  ...
]
Output ONLY the JSON list, no explanation."""

PROMPT_DEPENDENCY = """\
Task: given an input prompt and its tuples, identify the parent tuple of each tuple.

Rules:
- An attribute/count/relation tuple depends on the entity tuple(s) it describes.
- A relation tuple depends on BOTH entity tuples it connects.
- An entity tuple has no parent (null).
- Return a JSON object mapping each tuple id to its parent id(s) (null or a list).

Input prompt: {prompt}
Tuples: {tuples}

Output ONLY a JSON object, e.g.:
{{"E1": null, "A1": ["E1"], "R1": ["E1", "E2"]}}"""

PROMPT_QUESTION = """\
Task: rewrite each tuple as a natural-language yes/no question about an image.

Rules:
- The question must be answerable with "yes" or "no".
- Be specific: reference the exact entity/attribute mentioned.
- For relation tuples with relation_type "distinct", ask whether the two entities
  appear as clearly separate, visually distinguishable objects in the image.
  e.g. "Are the baby and the doll clearly two separate objects in the image?"
- Output a JSON object mapping each tuple id to its question string.

Input prompt: {prompt}
Tuples: {tuples}

Output ONLY a JSON object, e.g.:
{{"E1": "Is there a car in the image?", "A1": "Is the car red?", "R1": "Are the baby and the doll two visually distinct objects?"}}"""

PROMPT_VQA = "Answer with ONLY 'yes' or 'no', nothing else.\n\nQuestion: {question}"

# ---------------------------------------------------------------------------
# OpenRouter client factory
# ---------------------------------------------------------------------------

def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://github.com/j-min/DSG",
            "X-Title":      "DSG-Eval",
        },
    )

# ---------------------------------------------------------------------------
# LLM call (text only)
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, system: str, user: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# VQA call via OpenRouter image_url content block
# ---------------------------------------------------------------------------

def image_to_data_url(image_path: str) -> tuple[str, str]:
    """Return (data_url, media_type) for a local image."""
    ext = Path(image_path).suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png",  ".gif":  "image/gif",
                 ".webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    return f"data:{media_type};base64,{b64}", media_type


def call_vqa(client: OpenAI, question: str, data_url: str, model: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": PROMPT_VQA.format(question=question)},
            ],
        }],
    )
    return resp.choices[0].message.content.strip().lower()

# ---------------------------------------------------------------------------
# JSON extraction helper (strips markdown fences if present)
# ---------------------------------------------------------------------------

def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    # Sometimes models wrap the list/dict inside extra text; try to isolate it
    for start, end in [('{', '}'), ('[', ']')]:
        si = text.find(start)
        ei = text.rfind(end)
        if si != -1 and ei != -1 and ei > si:
            try:
                return json.loads(text[si:ei+1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)  # final attempt — will raise if broken

# ---------------------------------------------------------------------------
# DSG pipeline steps
# ---------------------------------------------------------------------------

def generate_tuples(client: OpenAI, prompt: str, model: str) -> list[dict]:
    raw = call_llm(client,
                   "You are a semantic scene decomposition assistant. Respond only with JSON.",
                   PROMPT_TUPLE.format(prompt=prompt),
                   model)
    return extract_json(raw)


def generate_dependencies(client: OpenAI, prompt: str, tuples: list[dict], model: str) -> dict:
    raw = call_llm(client,
                   "You are a dependency analysis assistant. Respond only with JSON.",
                   PROMPT_DEPENDENCY.format(prompt=prompt, tuples=json.dumps(tuples, indent=2)),
                   model)
    return extract_json(raw)


def generate_questions(client: OpenAI, prompt: str, tuples: list[dict], model: str) -> dict:
    raw = call_llm(client,
                   "You are a visual question generation assistant. Respond only with JSON.",
                   PROMPT_QUESTION.format(prompt=prompt, tuples=json.dumps(tuples, indent=2)),
                   model)
    return extract_json(raw)


def run_vqa(client: OpenAI, questions: dict, data_url: str, vqa_model: str) -> dict:
    """Run VQA for each question. Returns {id: 'yes'|'no'}."""
    answers = {}
    for qid, question in questions.items():
        print(f"  [VQA] {qid}: {question}")
        raw = call_vqa(client, question, data_url, vqa_model)
        answers[qid] = "yes" if "yes" in raw else "no"
        print(f"         → {answers[qid]}")
    return answers

# ---------------------------------------------------------------------------
# Dependency-aware scoring
# ---------------------------------------------------------------------------

def dependency_aware_score(tuples: list[dict], dependencies: dict,
                            questions: dict, answers: dict) -> dict:
    """
    Dependency masking: if any parent tuple answers 'no', skip child from scoring.

    Hallucination scoring:
      stated   → model SHOULD show these  (high weight, penalise absence)
      implied  → model SHOULD plausibly show these  (medium weight, reward presence)
      invented → model SHOULD NOT show these  (penalty for presence)
    """
    results = {}
    for t in tuples:
        tid       = t["id"]
        parents   = dependencies.get(tid)
        grounding = t.get("grounding", "stated")  # default for backward compat

        invalid = False
        if parents:
            for pid in parents:
                if answers.get(pid, "no") == "no":
                    invalid = True
                    break

        results[tid] = {
            "skill":      t["skill"],
            "args":       t["args"],
            "question":   questions.get(tid, ""),
            "answer":     answers.get(tid, "no"),
            "grounding":  grounding,
            "parents":    parents,
            "invalid":    invalid,
            "counted":    not invalid,
        }

    valid    = [r for r in results.values() if r["counted"]]
    stated   = [r for r in valid if r["grounding"] == "stated"]
    implied  = [r for r in valid if r["grounding"] == "implied"]
    invented = [r for r in valid if r["grounding"] == "invented"]

    # Entity confusion: distinct-relation tuples that failed (answer == "no")
    distinct_failures = [
        r for r in stated
        if r["skill"] == "relation"
        and len(r["args"]) >= 2 and r["args"][1] == "distinct"
        and r["answer"] == "no"
    ]

    def acc(bucket):
        if not bucket:
            return None
        return round(sum(1 for r in bucket if r["answer"] == "yes") / len(bucket), 4)

    stated_acc   = acc(stated)    # fidelity  — high = good
    implied_acc  = acc(implied)   # coherence — high = good (desirable hallucination)
    invented_acc = acc(invented)  # invention — high = bad  (undesirable hallucination)

    # Weighted composite — tune these to your task:
    #   increase W_INVENTED penalty for strict prompt-following tasks
    #   reduce   W_INVENTED toward 0 for creative/generative tasks
    W_STATED   =  1.0
    W_IMPLIED  =  0.5
    W_INVENTED = -0.3

    weight_map = [
        (W_STATED,   stated_acc),
        (W_IMPLIED,  implied_acc),
        (W_INVENTED, invented_acc),
    ]
    active = [(w, v) for w, v in weight_map if v is not None]
    if active:
        numerator   = sum(w * v for w, v in active)
        denominator = sum(abs(w) for w, _ in active)
        overall = max(0.0, round(numerator / denominator, 4))
    else:
        overall = 0.0

    # Per-skill accuracy across all valid tuples regardless of grounding
    skill_scores: dict[str, dict] = {}
    for r in valid:
        s = r["skill"]
        skill_scores.setdefault(s, {"yes": 0, "total": 0})
        skill_scores[s]["total"] += 1
        if r["answer"] == "yes":
            skill_scores[s]["yes"] += 1
    skill_acc = {s: round(v["yes"] / v["total"], 4) for s, v in skill_scores.items()}

    return {
        "overall_dsg_score":    overall,
        "stated_fidelity":      stated_acc,    # prompt faithfulness        (↑ good)
        "implied_coherence":    implied_acc,   # plausible elaboration      (↑ good)
        "invented_rate":        invented_acc,  # undesirable hallucination  (↓ good)
        "entity_confusion":     [r["question"] for r in distinct_failures],  # merged/confused entities
        "per_skill_accuracy":   skill_acc,
        "tuple_results":        results,
    }

# ---------------------------------------------------------------------------
# Top-level evaluate function
# ---------------------------------------------------------------------------

def evaluate(prompt: str, image_path: str, client: OpenAI,
             llm_model: str, vqa_model: str, verbose: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"Prompt    : {prompt}")
    print(f"Image     : {image_path}")
    print(f"LLM model : {llm_model}")
    print(f"VQA model : {vqa_model}")
    print(f"{'='*60}")

    print("\n[Step 1] Generating semantic tuples …")
    tuples = generate_tuples(client, prompt, llm_model)
    if verbose:
        for t in tuples:
            g = t.get('grounding', 'stated')
            print(f"  {t['id']} ({t['skill']}, {g}): {t['args']}")

    print("\n[Step 2] Inferring dependency graph …")
    dependencies = generate_dependencies(client, prompt, tuples, llm_model)
    if verbose:
        print(f"  {dependencies}")

    print("\n[Step 3] Generating yes/no questions …")
    questions = generate_questions(client, prompt, tuples, llm_model)
    if verbose:
        for qid, q in questions.items():
            print(f"  {qid}: {q}")

    print("\n[Step 4] Loading image …")
    data_url, _ = image_to_data_url(image_path)

    print("\n[Step 5] Running VQA …")
    answers = run_vqa(client, questions, data_url, vqa_model)

    print("\n[Step 6] Computing DSG score (dependency-aware) …")
    result = dependency_aware_score(tuples, dependencies, questions, answers)
    result["prompt"]     = prompt
    result["image_path"] = image_path
    result["llm_model"]  = llm_model
    result["vqa_model"]  = vqa_model

    print(f"\n  ✓ Overall DSG score   : {result['overall_dsg_score']:.2%}")
    print(f"  ✓ Stated fidelity     : {result['stated_fidelity']:.2%}"   if result['stated_fidelity']   is not None else "  - Stated fidelity     : n/a")
    print(f"  ✓ Implied coherence   : {result['implied_coherence']:.2%}" if result['implied_coherence'] is not None else "  - Implied coherence   : n/a")
    print(f"  ✓ Invented rate       : {result['invented_rate']:.2%}"     if result['invented_rate']     is not None else "  - Invented rate       : n/a")
    if result["entity_confusion"]:
        print(f"  ✗ Entity confusion    : {len(result['entity_confusion'])} failure(s)")
        for q in result["entity_confusion"]:
            print(f"      - {q}")
    print(f"  ✓ Per-skill           : {result['per_skill_accuracy']}")

    return result

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSG Text-to-Image Fidelity Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Single text prompt")
    group.add_argument("--csv",    help="CSV file with columns: prompt, image_path")

    parser.add_argument("--image",     help="Path to image; if omitted (or blank in CSV), "
                                            "an image is generated via Stable Diffusion")
    parser.add_argument("--sd-model",  default=DEFAULT_SD_MODEL,
                                       help="Stable Diffusion model for image generation")
    parser.add_argument("--device",    default=DEFAULT_DEVICE,
                                       help="Device for Stable Diffusion (cuda / cpu / mps)")
    parser.add_argument("--output",    default="dsg_results.json",   help="Output JSON path")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,    help="OpenRouter model for LLM steps")
    parser.add_argument("--vqa-model", default=DEFAULT_VQA_MODEL,    help="OpenRouter model for VQA step")
    parser.add_argument("--api-key",   default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API key")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("Error: set OPENROUTER_API_KEY env var or pass --api-key")

    client = make_client(args.api_key)

    def resolve_image(prompt: str, image_path: str, index: int = 0) -> str:
        """Return image_path, generating the image first if the file doesn't exist."""
        if image_path and Path(image_path).exists():
            return image_path
        save_path = image_path if image_path else f"dsg_generated_{index}.png"
        return generate_sd_image(prompt, save_path, args.sd_model, args.device)

    if args.prompt:
        image_path = resolve_image(args.prompt, args.image, index=0)
        results = [evaluate(args.prompt, image_path, client, args.llm_model, args.vqa_model)]
    else:
        import csv
        results = []
        with open(args.csv, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                prompt     = row["prompt"].strip()
                image_path = resolve_image(prompt, row.get("image_path", "").strip(), index=i)
                res = evaluate(prompt, image_path, client, args.llm_model, args.vqa_model)
                results.append(res)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {args.output}")

    if len(results) > 1:
        print(f"\n{'Prompt':<40} {'DSG':>7} {'Fidelity':>9} {'Coherence':>10} {'Invented':>9}")
        print("-" * 78)
        for r in results:
            sf = f"{r['stated_fidelity']:.0%}"   if r['stated_fidelity']   is not None else "n/a"
            ic = f"{r['implied_coherence']:.0%}" if r['implied_coherence'] is not None else "n/a"
            iv = f"{r['invented_rate']:.0%}"     if r['invented_rate']     is not None else "n/a"
            print(f"{r['prompt'][:38]:<40} {r['overall_dsg_score']:>7.0%} {sf:>9} {ic:>10} {iv:>9}")
        avg = sum(r["overall_dsg_score"] for r in results) / len(results)
        print(f"\n  Average DSG score: {avg:.2%}")


if __name__ == "__main__":
    main()