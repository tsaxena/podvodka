"""
reward_model/score_images.py

Score generated images against their prompts using:
  - HPSv2  (Human Preference Score v2)
  - DSG    (Davidsonian Scene Graph – compositional text-image alignment)

Install dependencies:
    pip install hpsv2
    pip install openai          # for DSG question generation (--llm_backend openai)
    pip install transformers    # for DSG VQA backbone (BLIP-2)

Usage examples:

    # Single image + prompt, both metrics
    python reward_model/score_images.py \\
        --image imgs/0.png \\
        --prompt "a red car on a mountain road at sunset" \\
        --metrics hpsv2 dsg

    # Image directory (all PNG/JPG), shared prompt
    python reward_model/score_images.py \\
        --image_dir imgs/ \\
        --prompt "a red car on a mountain road at sunset" \\
        --output data/scores.jsonl

    # CSV batch  (columns: prompt, image_path)
    python reward_model/score_images.py \\
        --prompts_file data/eval.csv \\
        --output data/scores.jsonl \\
        --metrics hpsv2 dsg \\
        --llm_backend openai \\
        --openai_model gpt-4o-mini

    # HPSv2 only, CSV output
    python reward_model/score_images.py \\
        --prompts_file data/eval.csv \\
        --output data/scores.csv \\
        --metrics hpsv2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# HPSv2
# ──────────────────────────────────────────────────────────────────────────────

def _load_hpsv2():
    try:
        import hpsv2
        return hpsv2
    except ImportError:
        print(
            "[HPSv2] hpsv2 not installed. Run: pip install hpsv2",
            file=sys.stderr,
        )
        return None


def _score_hpsv2(hpsv2_mod, image: Image.Image, prompt: str, version: str) -> float:
    """Return a single HPSv2 score for one (image, prompt) pair."""
    scores = hpsv2_mod.score(image, prompt, hps_version=version)
    if isinstance(scores, (list, tuple)):
        return float(scores[0])
    return float(scores)


# ──────────────────────────────────────────────────────────────────────────────
# DSG – question generation
# ──────────────────────────────────────────────────────────────────────────────

_DSG_SYSTEM = """You are an image-evaluation assistant.
Given a text-to-image prompt, decompose it into atomic yes/no questions \
that can be verified by looking at the image.

Return a JSON object with key "questions" whose value is an array.
Each element must have:
  "id"        : integer (1-based, unique)
  "question"  : a short yes/no question directly about image content
  "parent_id" : id of the question this depends on, or null for root questions

Cover: objects, attributes (color, shape, texture), counts, spatial relations.
Keep each question self-contained and simple.

Example output:
{
  "questions": [
    {"id": 1, "question": "Is there a car in the image?", "parent_id": null},
    {"id": 2, "question": "Is the car red?",              "parent_id": 1},
    {"id": 3, "question": "Is the car on a road?",        "parent_id": 1}
  ]
}"""


def _parse_with_openai(prompt: str, model: str) -> List[Dict]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Run: pip install openai")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DSG_SYSTEM},
            {"role": "user",   "content": f'Prompt: "{prompt}"'},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(response.choices[0].message.content)
    # Accept {"questions": [...]} or a bare list
    if isinstance(raw, list):
        return raw
    return raw.get("questions", [])


def _parse_fallback(prompt: str) -> List[Dict]:
    """
    Minimal rule-based fallback when no LLM is available.
    Returns a single question covering the whole prompt.
    """
    return [
        {
            "id": 1,
            "question": f"Does the image accurately depict '{prompt}'?",
            "parent_id": None,
        }
    ]


def _generate_questions(prompt: str, llm_backend: str, openai_model: str) -> List[Dict]:
    if llm_backend == "openai":
        return _parse_with_openai(prompt, openai_model)
    # "local" / unknown → fallback
    return _parse_fallback(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# DSG – VQA backbone (BLIP-2)
# ──────────────────────────────────────────────────────────────────────────────

def _load_dsg_vqa(device: str) -> Tuple:
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"[DSG] Loading VQA backbone ({model_id}) on {device}…")
    processor = Blip2Processor.from_pretrained(model_id)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = (
        Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    return processor, model


@torch.no_grad()
def _ask_vqa(processor, model, device: str, image: Image.Image, question: str) -> str:
    """Ask a yes/no question; return the model's short answer string."""
    text = f"Question: {question} Answer yes or no. Answer:"
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# DSG – scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_dsg(
    processor,
    model,
    device: str,
    image: Image.Image,
    prompt: str,
    llm_backend: str,
    openai_model: str,
) -> Dict:
    """
    Compute DSG score for one (image, prompt) pair.

    Algorithm:
      1. LLM decomposes prompt into a dependency tree of yes/no questions.
      2. BLIP-2 answers each question.
      3. If a parent is answered "no", all its descendants are set to 0
         (Davidsonian dependency propagation).
      4. Final score = fraction of questions answered "yes".
    """
    questions = _generate_questions(prompt, llm_backend, openai_model)
    if not questions:
        return {"dsg_score": 0.0, "num_questions": 0, "question_scores": []}

    # Sort by id so parents are processed before children
    questions = sorted(questions, key=lambda q: int(q.get("id", 0)))

    id_to_score: Dict[int, float] = {}
    question_results = []

    for q in questions:
        qid = int(q.get("id", 0))
        parent_id = q.get("parent_id")
        question_text = q.get("question", "")

        # Propagate failure: if parent failed, this node automatically fails
        if parent_id is not None and id_to_score.get(int(parent_id), 1.0) == 0.0:
            score = 0.0
            answer = "skipped (parent failed)"
        else:
            answer = _ask_vqa(processor, model, device, image, question_text)
            score = 1.0 if answer.startswith("yes") else 0.0

        id_to_score[qid] = score
        question_results.append(
            {
                "id": qid,
                "question": question_text,
                "answer": answer,
                "score": score,
            }
        )

    dsg_score = sum(r["score"] for r in question_results) / len(question_results)
    return {
        "dsg_score": dsg_score,
        "num_questions": len(question_results),
        "question_scores": question_results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pair evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_pair(
    image_path: str,
    prompt: str,
    metrics: List[str],
    hpsv2_mod=None,
    hps_version: str = "v2.1",
    dsg_processor=None,
    dsg_model=None,
    device: str = "cuda",
    llm_backend: str = "openai",
    openai_model: str = "gpt-3.5-turbo",
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    result: Dict = {"image": image_path, "prompt": prompt}

    if "hpsv2" in metrics:
        if hpsv2_mod is not None:
            result["hpsv2"] = _score_hpsv2(hpsv2_mod, image, prompt, hps_version)
        else:
            result["hpsv2"] = None

    if "dsg" in metrics:
        if dsg_processor is not None and dsg_model is not None:
            dsg = _score_dsg(
                dsg_processor, dsg_model, device, image, prompt,
                llm_backend, openai_model,
            )
            result["dsg_score"] = dsg["dsg_score"]
            result["dsg_num_questions"] = dsg["num_questions"]
            result["dsg_question_scores"] = dsg["question_scores"]
        else:
            result["dsg_score"] = None

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_pairs(args) -> List[Tuple[str, str]]:
    if args.image:
        return [(args.image, args.prompt)]

    if args.image_dir:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        pairs = [
            (str(p), args.prompt)
            for p in sorted(Path(args.image_dir).iterdir())
            if p.suffix.lower() in exts
        ]
        if not pairs:
            print(f"No images found in {args.image_dir}", file=sys.stderr)
            sys.exit(1)
        return pairs

    # prompts_file
    import pandas as pd

    df = pd.read_csv(args.prompts_file)
    missing = {"prompt", "image_path"} - set(df.columns)
    if missing:
        print(f"CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    return list(zip(df["image_path"].tolist(), df["prompt"].tolist()))


def _print_summary(results: List[Dict], metrics: List[str]) -> None:
    print("\n── Summary " + "─" * 50)
    for metric in metrics:
        key = "hpsv2" if metric == "hpsv2" else "dsg_score"
        vals = [r[key] for r in results if r.get(key) is not None]
        if vals:
            print(
                f"  {metric.upper():<8}  n={len(vals)}"
                f"  mean={sum(vals)/len(vals):.4f}"
                f"  min={min(vals):.4f}"
                f"  max={max(vals):.4f}"
            )
    print("─" * 60)


def _save_results(results: List[Dict], output: str) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".csv":
        import pandas as pd

        flat = [
            {k: v for k, v in r.items() if k != "dsg_question_scores"}
            for r in results
        ]
        pd.DataFrame(flat).to_csv(out_path, index=False)
    else:
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} results → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score generated images with HPSv2 and/or DSG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",        type=str, help="Path to a single image.")
    src.add_argument("--image_dir",    type=str, help="Directory of images (PNG/JPG).")
    src.add_argument("--prompts_file", type=str, help="CSV with 'prompt' and 'image_path' columns.")

    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt (required with --image or --image_dir).",
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--metrics", nargs="+", choices=["hpsv2", "dsg"], default=["hpsv2", "dsg"],
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--hps_version", default="v2.1", choices=["v2", "v2.1"],
        help="HPSv2 model version.",
    )

    # ── DSG options ───────────────────────────────────────────────────────
    parser.add_argument(
        "--llm_backend", default="openai", choices=["openai", "local"],
        help="LLM for DSG question generation. 'local' uses a rule-based fallback.",
    )
    parser.add_argument(
        "--openai_model", default="gpt-4o-mini",
        help="OpenAI model for DSG question generation.",
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file (.jsonl or .csv). Defaults to stdout (JSONL, no question detail).",
    )

    # ── Device ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )

    args = parser.parse_args()

    if (args.image or args.image_dir) and args.prompt is None:
        parser.error("--prompt is required when using --image or --image_dir.")

    # ── Load models ───────────────────────────────────────────────────────
    hpsv2_mod = None
    if "hpsv2" in args.metrics:
        hpsv2_mod = _load_hpsv2()

    dsg_processor, dsg_model = None, None
    if "dsg" in args.metrics:
        dsg_processor, dsg_model = _load_dsg_vqa(args.device)

    # ── Collect (image_path, prompt) pairs ────────────────────────────────
    pairs = _build_pairs(args)
    print(f"Scoring {len(pairs)} image(s) with metrics: {args.metrics}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    results = []
    for image_path, prompt in tqdm(pairs, desc="Scoring"):
        try:
            result = evaluate_pair(
                image_path=image_path,
                prompt=prompt,
                metrics=args.metrics,
                hpsv2_mod=hpsv2_mod,
                hps_version=args.hps_version,
                dsg_processor=dsg_processor,
                dsg_model=dsg_model,
                device=args.device,
                llm_backend=args.llm_backend,
                openai_model=args.openai_model,
            )
        except Exception as exc:
            print(f"  ERROR {image_path}: {exc}", file=sys.stderr)
            result = {"image": image_path, "prompt": prompt, "error": str(exc)}

        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    if results:
        _print_summary(results, args.metrics)

    # ── Write / print output ──────────────────────────────────────────────
    if args.output:
        _save_results(results, args.output)
    else:
        for r in results:
            print(json.dumps({k: v for k, v in r.items() if k != "dsg_question_scores"}))


if __name__ == "__main__":
    main()
