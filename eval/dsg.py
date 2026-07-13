"""
Davidsonian Scene Graph (DSG) - Text-to-Image Fidelity Evaluation
+ HPSv2 (Human Preference Score v2) - Aesthetic/Preference Evaluation
Based on: https://github.com/j-min/DSG  (ICLR 2024)
          https://github.com/tgxs002/HPSv2

Models:
  Image : runwayml/stable-diffusion-v1-5 (local, via diffusers)
  LLM   : any text model  (default: google/gemma-3-27b-it, via OpenRouter)
  VQA   : qwen/qwen3-vl-32b-instruct  (via OpenRouter)
  HPS   : hpsv2 package (local, CLIP-based preference model checkpoint auto-downloaded on first use)

Pipeline:
  0. Stable Diffusion generates an image from the prompt (if --image not given)
  1. LLM generates skill-specific tuples from the prompt
  2. LLM infers dependency graph between tuples
  3. LLM rewrites each tuple as a natural-language yes/no question
  4. Qwen VL answers each question given the generated image
  5. Dependency-aware scoring: skip child questions if parent fails
  6. Report stated_fidelity / implied_coherence / invented_rate + overall DSG score
  7. HPSv2 scores the same image against the prompt (independent of steps 1-6)

Hallucination classification:
  stated   - explicitly in the prompt          → must be present (penalise if missing)
  implied  - strongly implied by the prompt    → desirable if present
  invented - not stated or implied             → undesirable if present (penalised)

HPSv2 note: the score is a CLIP-based preference prediction. It is most meaningful when
comparing multiple images generated for the SAME prompt (that's what the model was trained
for); across different prompts, some prompts are just easier to score well on regardless of
image quality, so treat cross-prompt HPS comparisons with a bit of caution — DSG's
per-prompt fidelity/coherence breakdown is the more apples-to-apples signal for that.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    pip install hpsv2   # first run auto-downloads the HPS v2/v2.1 checkpoint

    # Generate image from prompt, then evaluate (DSG + HPS, both on by default)
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
        --vqa-model "qwen/qwen3-vl-32b-instruct" \\
        --hps-version v2.1

    # Enrich prompt before image generation (base / dpo / ppo)
    python dsg.py --prompt "a red car next to a blue bicycle" \\
        --enrich-method ppo \\
        --enrich-device cpu \\
        --enrich-max-tokens 80

    # Only run one of the two scores
    python dsg.py --prompt "..." --skip-hps   # DSG only
    python dsg.py --prompt "..." --skip-dsg   # HPS only (no OPENROUTER_API_KEY needed)

    # Evaluate a trained checkpoint over the 200 DrawBench prompts (shunk031/DrawBench).
    # Generates --num-samples completions per prompt, builds "<prompt><completion>" as the
    # final image prompt, then runs DSG+HPS on each. Results APPEND to --output across runs,
    # so evaluating multiple checkpoints over time builds one comparable table.
    python dsg.py --checkpoint-path /path/to/checkpoint --step-label v7_step_7177 \\
        --num-samples 2 --output dsg_hps_results.json
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Workaround for a diffusers/torch incompatibility (not a dsg.py bug):
# diffusers' attention_dispatch.py gates a flash-attention-3 custom op registration
# behind `if torch.__version__ >= "2.4.0"`, assuming any torch >= 2.4.0 fully supports
# it. In practice, some torch 2.4.x builds have torch.library.custom_op available but
# their infer_schema() can't parse this specific op's type hints, causing
# `from diffusers import StableDiffusionPipeline` to fail at import time — even though
# nothing in this project ever touches flash-attention-3 or the models that use it.
#
# IMPORTANT: this must be SELECTIVE by op name, not a blanket replacement of
# torch.library.custom_op/register_fake. A blanket patch was tried first and broke a
# completely different, unrelated thing: diffusers' own import chain transitively
# imports torch._dynamo / torch.distributed.tensor, which ALSO call custom_op/
# register_fake internally and depend on the real implementation's return value
# (a proper CustomOpDef with a .register_fake attribute, which a naive no-op stub
# doesn't provide) — a blanket patch broke those with an unrelated AttributeError.
# This version only intercepts the exact two ops that are actually broken, and
# passes every other call straight through to the real implementation.
try:
    import torch.library as _torch_library

    _real_custom_op = _torch_library.custom_op
    _real_register_fake = _torch_library.register_fake
    _DSG_PROBLEM_OPS = {
        "_diffusers_flash_attn_3::_flash_attn_forward",
        "_diffusers_flash_attn_3::_flash_attn_backward",
    }

    def _dsg_selective_custom_op(name, fn=None, /, *, mutates_args=None, device_types=None, schema=None):
        if name in _DSG_PROBLEM_OPS:
            def _wrap(func):
                return func
            return _wrap if fn is None else fn
        return _real_custom_op(name, fn, mutates_args=mutates_args, device_types=device_types, schema=schema)

    def _dsg_selective_register_fake(op, fn=None, /, *, lib=None, _stacklevel=1):
        if isinstance(op, str) and op in _DSG_PROBLEM_OPS:
            def _wrap(func):
                return func
            return _wrap if fn is None else fn
        return _real_register_fake(op, fn, lib=lib, _stacklevel=_stacklevel)

    _torch_library.custom_op = _dsg_selective_custom_op
    _torch_library.register_fake = _dsg_selective_register_fake
except ImportError:
    pass  # torch not installed — nothing to patch

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


def _image_cache_path(prompt: str, model_name: str, cache_dir: Path) -> Path:
    key = hashlib.md5(f"{model_name}:{prompt}".encode()).hexdigest()
    return cache_dir / f"{key}.png"


def generate_sd_image(prompt: str,
                      model_name: str = DEFAULT_SD_MODEL,
                      device: str = DEFAULT_DEVICE,
                      cache_dir: Path | None = None):
    """Generate an image with Stable Diffusion and return it as a PIL Image.

    If cache_dir is given, the image is saved there on first generation and loaded
    from disk on subsequent calls with the same prompt + model — skipping SD entirely.
    Cache files are named by MD5(model:prompt) so they are stable across restarts.
    """
    from PIL import Image as PILImage

    if cache_dir is not None:
        cached = _image_cache_path(prompt, model_name, cache_dir)
        if cached.exists():
            print(f"[SD] Cache hit — loading {cached}")
            return PILImage.open(cached).convert("RGB")

    pipe = _load_sd_pipeline(model_name, device)
    print(f"[SD] Generating: {prompt!r}")
    image = pipe(prompt).images[0]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        image.save(cached)
        print(f"[SD] Cached to {cached}")

    return image

# ---------------------------------------------------------------------------
# Prompt enrichment via local GPT-2 models (base / DPO / PPO)
# ---------------------------------------------------------------------------

BASE_ENRICH_MODEL = "tsaxena/gpt2-large-prompt-tags"
PPO_ENRICH_MODEL  = "tsaxena/gpt2-large-ppo-prompt-tags"
DPO_ENRICH_MODEL  = "tsaxena/gpt2-large-dpo-corrected"

def _load_drawbench_prompts() -> list[str]:
    """Load all 200 DrawBench benchmark prompts (Saharia et al., Imagen 2022).

    Dataset : shunk031/DrawBench on HuggingFace (11 categories, 200 prompts total).
    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Error: the `datasets` package is required for --checkpoint-path evaluation.\n"
                 "Install with: pip install datasets")
    # datasets >= 3.0 removed support for dataset scripts entirely; trust_remote_code no
    # longer helps. Try without it first — works when the dataset has Parquet snapshots.
    try:
        ds = load_dataset("shunk031/DrawBench", split="test")
    except Exception:
        try:
            ds = load_dataset("shunk031/DrawBench", split="test", trust_remote_code=True)
        except RuntimeError as e:
            sys.exit(
                f"Error loading DrawBench: {e}\n"
                "The installed 'datasets' version no longer supports dataset scripts.\n"
                "Fix: pip install 'datasets<3.0'"
            )
    return [row["prompts"] for row in ds]

_enrich_cache: dict = {}  # {model_path: (model, tokenizer)} — loaded once per process


def _load_enrich_model(model_path: str, device: str):
    """Load and cache a causal-LM enrichment model from HuggingFace."""
    if model_path in _enrich_cache:
        return _enrich_cache[model_path]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("Error: transformers and torch are required for prompt enrichment.\n"
                 "Install with: pip install transformers torch accelerate")

    import torch  # noqa: F811 (re-import after guard for type checker)
    print(f"[Enrich] Loading {model_path} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    _enrich_cache[model_path] = (model, tokenizer)
    return model, tokenizer


def enrich_prompt(prompt: str, method: str = "ppo", device: str = "cpu",
                  max_new_tokens: int = 80) -> str:
    """
    Enrich a text-to-image prompt using a fine-tuned GPT-2 model.

    method:
        "base" — tsaxena/gpt2-large-prompt-tags       (SFT baseline)
        "ppo"  — tsaxena/gpt2-large-ppo-prompt-tags   (PPO-optimised)
        "dpo"  — tsaxena/gpt2-large-dpo-corrected     (DPO-optimised)

    Model input format : "<prompt></s>"
    Model output       : enrichment tags appended after the separator
    Returns            : "<original prompt>, <generated tags>"
    """
    model_map = {
        "base": BASE_ENRICH_MODEL,
        "ppo":  PPO_ENRICH_MODEL,
        "dpo":  DPO_ENRICH_MODEL,
    }
    if method not in model_map:
        raise ValueError(f"Unknown enrich method {method!r}; choose from {list(model_map)}")

    import torch
    model_path = model_map[method]
    model, tokenizer = _load_enrich_model(model_path, device)

    input_text = f"{prompt}</s>"
    enc = tokenizer(input_text, return_tensors="pt",
                    truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[0, enc["input_ids"].shape[1]:]
    tags = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    enriched = f"{prompt}, {tags}" if tags else prompt
    print(f"[Enrich/{method.upper()}] Original : {prompt!r}")
    print(f"[Enrich/{method.upper()}] Enriched : {enriched!r}")
    return enriched

# ---------------------------------------------------------------------------
# Generate completions from an arbitrary checkpoint (mirrors eval_checkpoint.py)
# ---------------------------------------------------------------------------

def generate_completions_from_checkpoint(checkpoint_path: str, prompts: list[str],
                                         num_samples: int = 2, max_new_tokens: int = 80,
                                         temperature: float = 1.0, device: str = "cuda") -> list[dict]:
    """Generate `num_samples` completions per prompt from a GRPO/PPO/DPO-style checkpoint.

    Same generation settings and prompt formatting as eval_checkpoint.py (prompt + "</s>",
    left-padded batched sampling), so completions here are produced the same way as the
    reward-model eval — the only difference is what happens to them afterward (DSG/HPS on a
    rendered image here, vs. a text reward model there).

    Returns a flat list of {"prompt": str, "completion": str} dicts, length
    len(prompts) * num_samples.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Checkpoint] Loading {checkpoint_path} …")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generation with a causal LM

    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=model_dtype).to(device)
    model.eval()

    rows = []
    for prompt in prompts:
        prompt_text = prompt + "</s>"
        inputs = tokenizer([prompt_text] * num_samples, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]  # same for every row: left-padded to equal length
        completions = [
            tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in output_ids
        ]
        for completion in completions:
            rows.append({"prompt": prompt, "completion": completion})
        print(f"  [{prompt[:40]:40s}] generated {len(completions)} completion(s)")

    return rows

# ---------------------------------------------------------------------------
# HPSv2 (Human Preference Score v2) scoring
# ---------------------------------------------------------------------------

DEFAULT_HPS_VERSION = "v2.1"

_hpsv2_module = None  # cached import — the package lazily loads its CLIP checkpoint on first score() call


def compute_hps_score(image, prompt: str, hps_version: str = DEFAULT_HPS_VERSION) -> float:
    """Score a single (PIL Image, prompt) pair with HPSv2.

    hpsv2.score() always returns a list (confirmed from the installed package's source —
    even for a single image), so we always take element [0] here rather than branching on
    the return type.

    Note: v2.0 and v2.1 scores are NOT comparable to each other (different checkpoints,
    different scales) — keep hps_version consistent across everything you intend to compare.
    """
    global _hpsv2_module
    if _hpsv2_module is None:
        try:
            import hpsv2
        except ImportError:
            sys.exit("Error: hpsv2 is required for HPS scoring.\n"
                     "Install with: pip install hpsv2\n"
                     "(Use --skip-hps to run DSG only without this dependency.)")
        _hpsv2_module = hpsv2

    result = _hpsv2_module.score(image, prompt, hps_version=hps_version)
    return float(result[0])

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

def image_to_data_url(image) -> tuple[str, str]:
    """Return (data_url, media_type) for a PIL Image (encoded in-memory as PNG)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64, "image/png"


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


def call_llm_json(client: OpenAI, system: str, user: str, model: str, max_retries: int = 3):
    """call_llm + extract_json, retrying on malformed JSON.

    A single bad JSON response from the LLM (truncated output, a stray comment, an unescaped
    quote) doesn't mean the model can't do the task — resampling usually produces valid JSON on
    the next attempt. Retrying the LLM CALL (not just re-parsing the same broken text) is what
    actually recovers from this. Only raises after every attempt has failed, and when it does,
    includes the raw response text so the failure is debuggable from the error message alone —
    "Expecting ':' delimiter at char 1107" tells you nothing about what the model actually said.
    """
    last_error = None
    last_raw = None
    for attempt in range(1, max_retries + 1):
        raw = call_llm(client, system, user, model)
        last_raw = raw
        try:
            return extract_json(raw)
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  [warn] malformed JSON from {model} (attempt {attempt}/{max_retries}): {e}")
            continue
    raise RuntimeError(
        f"LLM ({model}) failed to produce valid JSON after {max_retries} attempts.\n"
        f"Last parse error: {last_error}\n"
        f"Last raw response:\n{last_raw}"
    )

# ---------------------------------------------------------------------------
# DSG pipeline steps
# ---------------------------------------------------------------------------

def generate_tuples(client: OpenAI, prompt: str, model: str) -> list[dict]:
    return call_llm_json(client,
                         "You are a semantic scene decomposition assistant. Respond only with JSON.",
                         PROMPT_TUPLE.format(prompt=prompt),
                         model)


def generate_dependencies(client: OpenAI, prompt: str, tuples: list[dict], model: str) -> dict:
    return call_llm_json(client,
                         "You are a dependency analysis assistant. Respond only with JSON.",
                         PROMPT_DEPENDENCY.format(prompt=prompt, tuples=json.dumps(tuples, indent=2)),
                         model)


def generate_questions(client: OpenAI, prompt: str, tuples: list[dict], model: str) -> dict:
    return call_llm_json(client,
                         "You are a visual question generation assistant. Respond only with JSON.",
                         PROMPT_QUESTION.format(prompt=prompt, tuples=json.dumps(tuples, indent=2)),
                         model)


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

def evaluate(prompt: str, image, client: OpenAI,
             llm_model: str, vqa_model: str,
             run_dsg: bool = True, run_hps: bool = True,
             hps_version: str = DEFAULT_HPS_VERSION,
             verbose: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"Prompt    : {prompt}")
    if run_dsg:
        print(f"LLM model : {llm_model}")
        print(f"VQA model : {vqa_model}")
    if run_hps:
        print(f"HPS ver.  : {hps_version}")
    print(f"{'='*60}")

    result = {"prompt": prompt}

    if run_dsg:
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

        print("\n[Step 4] Encoding image …")
        data_url, _ = image_to_data_url(image)

        print("\n[Step 5] Running VQA …")
        answers = run_vqa(client, questions, data_url, vqa_model)

        print("\n[Step 6] Computing DSG score (dependency-aware) …")
        dsg_result = dependency_aware_score(tuples, dependencies, questions, answers)
        result.update(dsg_result)
        result["llm_model"] = llm_model
        result["vqa_model"] = vqa_model

        print(f"\n  ✓ Overall DSG score   : {result['overall_dsg_score']:.2%}")
        print(f"  ✓ Stated fidelity     : {result['stated_fidelity']:.2%}"   if result['stated_fidelity']   is not None else "  - Stated fidelity     : n/a")
        print(f"  ✓ Implied coherence   : {result['implied_coherence']:.2%}" if result['implied_coherence'] is not None else "  - Implied coherence   : n/a")
        print(f"  ✓ Invented rate       : {result['invented_rate']:.2%}"     if result['invented_rate']     is not None else "  - Invented rate       : n/a")
        if result["entity_confusion"]:
            print(f"  ✗ Entity confusion    : {len(result['entity_confusion'])} failure(s)")
            for q in result["entity_confusion"]:
                print(f"      - {q}")
        print(f"  ✓ Per-skill           : {result['per_skill_accuracy']}")

    if run_hps:
        print("\n[Step 7] Computing HPSv2 score …")
        result["hps_score"] = compute_hps_score(image, prompt, hps_version)
        result["hps_version"] = hps_version
        print(f"  ✓ HPSv2 score ({hps_version}) : {result['hps_score']:.4f}")

    return result

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DSG Text-to-Image Fidelity Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", help="Single text prompt")
    group.add_argument("--csv",    help="CSV file with columns: prompt, image_path")
    group.add_argument("--checkpoint-path",
                       help="Generate completions from this checkpoint over the 200 DrawBench "
                            "prompts (shunk031/DrawBench on HuggingFace) and evaluate each. "
                            "Final image prompt = '<prompt><completion>'.")

    parser.add_argument("--image",     help="Path to image; if omitted (or blank in CSV), "
                                            "an image is generated via Stable Diffusion")
    parser.add_argument("--sd-model",  default=DEFAULT_SD_MODEL,
                                       help="Stable Diffusion model for image generation")
    parser.add_argument("--device",    default=DEFAULT_DEVICE,
                                       help="Device for Stable Diffusion (cuda / cpu / mps)")
    parser.add_argument("--image-cache-dir", default="dsg_image_cache",
                                       help="Directory for caching SD-generated images so they "
                                            "are not regenerated on retry (default: dsg_image_cache). "
                                            "Pass '' to disable caching.")
    parser.add_argument("--output",    default="dsg_results.json",   help="Output JSON path")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,    help="OpenRouter model for LLM steps")
    parser.add_argument("--vqa-model", default=DEFAULT_VQA_MODEL,    help="OpenRouter model for VQA step")
    parser.add_argument("--api-key",   default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API key")
    parser.add_argument("--hps-version", default=DEFAULT_HPS_VERSION, choices=["v2.0", "v2.1"],
                                         help="HPSv2 checkpoint version (default: v2.1). "
                                              "Scores from different versions are not comparable "
                                              "to each other — keep this consistent across a comparison.")
    parser.add_argument("--skip-dsg",  action="store_true",
                                       help="Skip DSG scoring (HPS only — no OPENROUTER_API_KEY needed)")
    parser.add_argument("--skip-hps",  action="store_true",
                                       help="Skip HPSv2 scoring (DSG only — no hpsv2 package needed)")

    ckpt_group = parser.add_argument_group("checkpoint evaluation (--checkpoint-path)")
    ckpt_group.add_argument("--step-label", default=None,
                            help="Label recorded in each result (e.g. the training step this "
                                 "checkpoint corresponds to). Defaults to the checkpoint dir/repo name.")
    ckpt_group.add_argument("--num-samples", type=int, default=2,
                            help="Completions sampled per prompt (default: 2 — each sample costs "
                                 "one full SD image + several LLM/VQA calls + one HPS score, so "
                                 "this is kept low by default; raise it if you want a more stable "
                                 "per-prompt average at proportionally higher cost).")
    ckpt_group.add_argument("--gen-max-new-tokens", type=int, default=80,
                            help="Max new tokens when generating completions from --checkpoint-path")
    ckpt_group.add_argument("--gen-temperature", type=float, default=1.0,
                            help="Sampling temperature when generating completions from --checkpoint-path")
    ckpt_group.add_argument("--gen-device", default="cuda",
                            help="Device for the checkpoint's generation model")

    enrich_group = parser.add_argument_group("prompt enrichment")
    enrich_group.add_argument(
        "--enrich-method", default="none",
        choices=["none", "base", "dpo", "ppo"],
        help="Enrich the prompt before image generation using a fine-tuned GPT-2 model. "
             "'base'=SFT, 'dpo'=DPO-optimised, 'ppo'=PPO-optimised (default: none)")
    enrich_group.add_argument(
        "--enrich-device", default="cpu",
        help="Device for the enrichment model (default: cpu)")
    enrich_group.add_argument(
        "--enrich-max-tokens", type=int, default=80,
        help="Max new tokens to generate for enrichment tags (default: 80)")

    args = parser.parse_args()

    if args.skip_dsg and args.skip_hps:
        sys.exit("Error: --skip-dsg and --skip-hps both set — nothing to compute.")

    run_dsg = not args.skip_dsg
    run_hps = not args.skip_hps

    if run_dsg and not args.api_key:
        sys.exit("Error: DSG scoring needs an OpenRouter key — set OPENROUTER_API_KEY env var, "
                 "pass --api-key, or use --skip-dsg to run HPS only.")

    client = make_client(args.api_key) if run_dsg else None

    # Load existing results up-front so periodic flushes can include them.
    existing: list = []
    if Path(args.output).exists():
        try:
            with open(args.output) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    def _flush(current_results: list, n: int) -> None:
        """Write existing + current_results to the output file and print a progress line."""
        with open(args.output, "w") as f:
            json.dump(existing + current_results, f, indent=2)
        print(f"  [flush] {n} prompt(s) done — partial results written to {args.output}")

    image_cache_dir = Path(args.image_cache_dir) if args.image_cache_dir else None

    def resolve_image(prompt: str, image_path: str):
        """Return a PIL Image. Loads from disk if image_path exists, otherwise generates via SD.
        Generated images are cached in image_cache_dir (if set) to avoid re-running SD on retry.
        """
        from PIL import Image as PILImage
        if image_path and Path(image_path).exists():
            return PILImage.open(image_path).convert("RGB")
        return generate_sd_image(prompt, args.sd_model, args.device, cache_dir=image_cache_dir)

    def maybe_enrich(prompt: str) -> str:
        """Return enriched prompt if --enrich-method is set, otherwise the original."""
        if args.enrich_method == "none":
            return prompt
        return enrich_prompt(prompt, method=args.enrich_method,
                             device=args.enrich_device,
                             max_new_tokens=args.enrich_max_tokens)

    if args.prompt:
        gen_prompt = maybe_enrich(args.prompt)
        image = resolve_image(gen_prompt, args.image)
        res = evaluate(args.prompt, image, client, args.llm_model, args.vqa_model,
                       run_dsg=run_dsg, run_hps=run_hps, hps_version=args.hps_version)
        res["enriched_prompt"] = gen_prompt
        results = [res]
    elif args.csv:
        import csv
        results = []
        with open(args.csv, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                prompt     = row["prompt"].strip()
                gen_prompt = maybe_enrich(prompt)
                image = resolve_image(gen_prompt, row.get("image_path", "").strip())
                try:
                    res = evaluate(prompt, image, client, args.llm_model, args.vqa_model,
                                   run_dsg=run_dsg, run_hps=run_hps, hps_version=args.hps_version)
                    res["enriched_prompt"] = gen_prompt
                except Exception as e:
                    # One bad LLM/API response shouldn't lose every other already-computed
                    # result in the batch — record the failure and keep going.
                    print(f"  [ERROR] row {i} ({prompt[:50]!r}) failed, skipping: {e}")
                    res = {"prompt": prompt, "enriched_prompt": gen_prompt, "error": str(e)}
                results.append(res)
                if len(results) % 20 == 0:
                    _flush(results, len(results))
    else:
        # --checkpoint-path: generate completions over DrawBench (200 prompts), evaluate each.
        step_label = args.step_label or Path(args.checkpoint_path).name
        print(f"[Checkpoint] step_label = {step_label!r}")

        drawbench_prompts = _load_drawbench_prompts()
        print(f"[Checkpoint] Loaded {len(drawbench_prompts)} DrawBench prompts")
        completions = generate_completions_from_checkpoint(
            args.checkpoint_path, drawbench_prompts,
            num_samples=args.num_samples,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            device=args.gen_device,
        )

        results = []
        for i, row in enumerate(completions):
            # Plain concatenation, matching the reward-model scoring convention used
            # throughout this project (prompt + completion — completions already carry
            # whatever leading punctuation/spacing they were trained to produce).
            final_prompt = row["prompt"] + row["completion"]
            image = resolve_image(final_prompt, "")
            try:
                res = evaluate(final_prompt, image, client, args.llm_model, args.vqa_model,
                               run_dsg=run_dsg, run_hps=run_hps, hps_version=args.hps_version)
            except Exception as e:
                # Same reasoning as the CSV loop: a batch here can be 400 items
                # (200 DrawBench prompts * num_samples) — one bad response must not cost every
                # other already-computed result.
                print(f"  [ERROR] item {i} ({row['prompt'][:50]!r}) failed, skipping: {e}")
                res = {"prompt": final_prompt, "error": str(e)}
            res["checkpoint_path"]   = args.checkpoint_path
            res["step_label"]        = step_label
            res["original_prompt"]   = row["prompt"]
            res["completion"]        = row["completion"]
            results.append(res)
            if len(results) % 20 == 0:
                _flush(results, len(results))

    # Append rather than overwrite, so evaluating multiple checkpoints over time (via repeated
    # --checkpoint-path runs with different --step-label values) builds one comparable table,
    # the same idea as eval_checkpoint.py's eval_results.csv.
    all_results = existing + results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ {len(results)} new result(s) appended to {args.output} ({len(all_results)} total)")

    ok_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]

    if len(results) > 1:
        header = f"{'Prompt':<40}"
        if run_dsg:
            header += f" {'DSG':>7} {'Fidelity':>9} {'Coherence':>10} {'Invented':>9}"
        if run_hps:
            header += f" {'HPS':>8}"
        print(f"\n{header}")
        print("-" * len(header))
        for r in ok_results:
            line = f"{r['prompt'][:38]:<40}"
            if run_dsg:
                sf = f"{r['stated_fidelity']:.0%}"   if r['stated_fidelity']   is not None else "n/a"
                ic = f"{r['implied_coherence']:.0%}" if r['implied_coherence'] is not None else "n/a"
                iv = f"{r['invented_rate']:.0%}"     if r['invented_rate']     is not None else "n/a"
                line += f" {r['overall_dsg_score']:>7.0%} {sf:>9} {ic:>10} {iv:>9}"
            if run_hps:
                line += f" {r['hps_score']:>8.4f}"
            print(line)
        for r in failed_results:
            print(f"{r['prompt'][:38]:<40} FAILED — {r['error'][:60]}")

        if failed_results:
            print(f"\n  ⚠ {len(failed_results)}/{len(results)} item(s) failed and are excluded "
                 f"from the averages below (see 'error' field in {args.output} for details)")
        if run_dsg and ok_results:
            avg_dsg = sum(r["overall_dsg_score"] for r in ok_results) / len(ok_results)
            print(f"\n  Average DSG score: {avg_dsg:.2%}  (n={len(ok_results)})")
        if run_hps and ok_results:
            avg_hps = sum(r["hps_score"] for r in ok_results) / len(ok_results)
            print(f"  Average HPS score ({args.hps_version}): {avg_hps:.4f}  (n={len(ok_results)})")

    # Cross-checkpoint trend, if --output now contains more than one distinct step_label
    # (i.e. this or a previous run used --checkpoint-path).
    labeled = [r for r in all_results if r.get("step_label")]
    step_labels = sorted(set(r["step_label"] for r in labeled))
    if len(step_labels) > 1:
        print(f"\n=== Trend across all checkpoints evaluated so far (same fixed prompts) ===")
        for label in step_labels:
            rows = [r for r in labeled if r["step_label"] == label]
            rows_ok = [r for r in rows if "error" not in r]
            n_failed = len(rows) - len(rows_ok)
            line = f"  {label:<24} n={len(rows_ok):<4}" + (f" ({n_failed} failed)" if n_failed else "")
            if run_dsg and rows_ok and all("overall_dsg_score" in r for r in rows_ok):
                line += f" DSG={sum(r['overall_dsg_score'] for r in rows_ok) / len(rows_ok):.2%}"
            if run_hps and rows_ok and all("hps_score" in r for r in rows_ok):
                line += f" HPS={sum(r['hps_score'] for r in rows_ok) / len(rows_ok):.4f}"
            print(line)


if __name__ == "__main__":
    main()