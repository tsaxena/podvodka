#!/usr/bin/env python3
"""
Judge-labeled DPO pipeline for prompt-expansion models.

This script is designed for the situation where you already paid for judge
scores, e.g.:

  1,000 prompts x 8 candidates = 8,000 scored candidates

It does three things:

  1. build_pairs
     Rebuilds multiple high-confidence DPO pairs per prompt from the existing
     OpenRouter multi-objective score checkpoint and candidate cache.

  2. train_dpo
     Trains a model with direct DPO loss using prompt/chosen/rejected JSONL.
     This avoids depending on TRL's changing DPOTrainer API.

  3. layer3
     Generates side-by-side SFT vs DPO samples and writes a human-readable
     Layer 3 review markdown with simple anti-hacking diagnostics.

Example:

  python dpo_judge_pipeline.py build_pairs \
    --scores /workspace/podvodka/data/openrouter_scores_mo.jsonl \
    --candidates /workspace/podvodka/data/sft_candidates.jsonl \
    --output /workspace/podvodka/data/preferences_openrouter_mo_multi_pairs.jsonl \
    --audit /workspace/podvodka/data/preferences_openrouter_mo_multi_pairs_audit.md \
    --target_pairs 3000 \
    --max_pairs_per_prompt 4 \
    --min_score_gap 1

  python dpo_judge_pipeline.py train_dpo \
    --model_name_or_path /workspace/qwen2_5_1_5b_prompt_sft/final \
    --dataset_path /workspace/podvodka/data/preferences_openrouter_mo_multi_pairs.jsonl \
    --output_dir /workspace/qwen2_5_1_5b_prompt_dpo_judge \
    --use_lora \
    --load_in_4bit

  python dpo_judge_pipeline.py layer3 \
    --sft_model /workspace/qwen2_5_1_5b_prompt_sft/final \
    --dpo_model /workspace/qwen2_5_1_5b_prompt_dpo_judge/final \
    --train_path /workspace/podvodka/data/train_strings.csv \
    --output_md /workspace/podvodka/reports/layer3_sft_vs_dpo.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------

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

CORE_FIELDS = [
    "fidelity",
    "style_fit",
    "non_genericness",
    "anti_magic_word_score",
]

DEFAULT_HARD_FAILURE_MODES = {
    "magic_words",
    "irrelevant_artist",
    "off_topic",
    "contradiction",
    "incoherent",
}

MAGIC_WORDS = [
    "8k",
    "4k",
    "artstation",
    "trending on artstation",
    "octane render",
    "unreal engine",
    "unreal engine 5",
    "masterpiece",
    "ultra detailed",
    "ultradetailed",
    "hyper detailed",
    "highly detailed",
    "award winning",
    "cinematic lighting",
    "hd",
]

ARTIST_WORDS = [
    "greg rutkowski",
    "artgerm",
    "wlop",
    "alphonse mucha",
    "makoto shinkai",
    "beeple",
    "loish",
    "rossdraws",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        x = int(x)
    except Exception:
        return None
    if 1 <= x <= 10:
        return x
    return None


def normalize_modes(modes: Any) -> List[str]:
    if modes is None:
        return []
    if isinstance(modes, str):
        modes = [modes]
    if not isinstance(modes, list):
        return []
    out = []
    for m in modes:
        if not isinstance(m, str):
            continue
        m2 = m.strip().lower().replace(" ", "_").replace("-", "_")
        if m2 and m2 not in out:
            out.append(m2)
    if "none" in out and len(out) > 1:
        out = [m for m in out if m != "none"]
    return out


def item_score(item: dict) -> Optional[int]:
    return as_int_or_none(item.get("overall", item.get("score")))


def count_phrase_hits(text: str, phrases: Sequence[str]) -> int:
    low = text.lower()
    return sum(low.count(p.lower()) for p in phrases)


def repeated_ngram_count(text: str, n: int = 3) -> int:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return 0
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values() if c > 1)


def token_len(text: str) -> int:
    return len(re.findall(r"\w+", text))


def prompt_copy_ratio(prompt: str, completion: str) -> float:
    p = set(re.findall(r"\w+", prompt.lower().replace("</s>", "")))
    c = re.findall(r"\w+", completion.lower())
    if not c:
        return 0.0
    return sum(1 for t in c if t in p) / max(1, len(c))


def ensure_tokenizer_pad(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# -----------------------------------------------------------------------------
# Stage 1: rebuild multiple high-confidence DPO pairs
# -----------------------------------------------------------------------------

@dataclass
class PairBuildConfig:
    scores_path: str
    candidates_path: str
    output_path: str
    audit_path: Optional[str]
    target_pairs: int
    max_pairs_per_prompt: int
    min_score_gap: int
    min_chosen_fidelity: int
    min_chosen_style_fit: int
    min_chosen_non_genericness: int
    min_chosen_anti_magic_word_score: int
    min_chosen_coherence: int
    hard_failure_modes: set
    prefer_hard_negative_pairs: bool
    seed: int


def load_scored_items(scores_path: str | Path) -> Dict[Tuple[int, int], dict]:
    items: Dict[Tuple[int, int], dict] = {}
    missing_objectives = 0
    for d in read_jsonl(scores_path):
        pid = d.get("prompt_id")
        cid = d.get("candidate_id")
        if pid is None or cid is None:
            continue
        if d.get("overall") is None and d.get("score") is not None:
            d["overall"] = d["score"]
        if d.get("score") is None and d.get("overall") is not None:
            d["score"] = d["overall"]
        d["failure_modes"] = normalize_modes(d.get("failure_modes"))
        for f in OBJECTIVE_FIELDS:
            d[f] = as_int_or_none(d.get(f))
        d["overall"] = as_int_or_none(d.get("overall"))
        d["score"] = as_int_or_none(d.get("score"))
        if any(d.get(f) is None for f in OBJECTIVE_FIELDS):
            missing_objectives += 1
        items[(int(pid), int(cid))] = d
    print(f"[load] loaded {len(items)} scored items from {scores_path}")
    if missing_objectives:
        print(f"[load] WARNING: {missing_objectives} rows are missing at least one objective score")
    return items


def load_candidates(candidates_path: str | Path) -> List[dict]:
    rows = list(read_jsonl(candidates_path))
    lens = Counter(len(r.get("candidates", [])) for r in rows)
    print(f"[load] loaded {len(rows)} prompt rows from {candidates_path}")
    print(f"[load] candidates-per-prompt distribution: {dict(lens)}")
    return rows


def chosen_passes_guardrails(item: dict, cfg: PairBuildConfig) -> Tuple[bool, str]:
    checks = [
        ("fidelity", cfg.min_chosen_fidelity),
        ("style_fit", cfg.min_chosen_style_fit),
        ("non_genericness", cfg.min_chosen_non_genericness),
        ("anti_magic_word_score", cfg.min_chosen_anti_magic_word_score),
        ("coherence", cfg.min_chosen_coherence),
    ]
    for field, threshold in checks:
        v = item.get(field)
        if v is not None and v < threshold:
            return False, f"chosen_{field}_below_{threshold}"

    hard = sorted(set(item.get("failure_modes", [])).intersection(cfg.hard_failure_modes))
    if hard:
        return False, "chosen_hard_failure_" + "+".join(hard)

    return True, ""


def chosen_does_not_lose_core(chosen: dict, rejected: dict) -> Tuple[bool, str]:
    for f in CORE_FIELDS:
        c = chosen.get(f)
        r = rejected.get(f)
        if c is not None and r is not None and c < r:
            return False, f"chosen_loses_{f}"
    return True, ""


def pair_priority(chosen: dict, rejected: dict, gap: int, cfg: PairBuildConfig) -> float:
    rejected_modes = set(rejected.get("failure_modes", []))
    hard_bonus = 2.0 if rejected_modes.intersection(cfg.hard_failure_modes) else 0.0
    generic_bonus = 1.0 if rejected_modes.intersection({"too_generic", "restatement", "too_verbose"}) else 0.0
    chosen_quality = (item_score(chosen) or 0) / 10.0
    anti_magic = (chosen.get("anti_magic_word_score") or 5) / 10.0
    return 3.0 * gap + hard_bonus + generic_bonus + chosen_quality + anti_magic


def score_payload(item: dict) -> Dict[str, Any]:
    return {
        "overall": item_score(item),
        "fidelity": item.get("fidelity"),
        "visual_specificity": item.get("visual_specificity"),
        "style_fit": item.get("style_fit"),
        "composition_lighting": item.get("composition_lighting"),
        "non_genericness": item.get("non_genericness"),
        "coherence": item.get("coherence"),
        "brevity_control": item.get("brevity_control"),
        "anti_magic_word_score": item.get("anti_magic_word_score"),
        "failure_modes": item.get("failure_modes", []),
        "reason": item.get("reason", ""),
    }


def build_multiple_pairs(cfg: PairBuildConfig) -> List[dict]:
    random.seed(cfg.seed)
    scored = load_scored_items(cfg.scores_path)
    cand_rows = load_candidates(cfg.candidates_path)

    all_pairs: List[dict] = []
    per_prompt_counts = []
    skip_reasons = Counter()
    valid_prompt_count = 0

    for prompt_id, row in enumerate(cand_rows):
        prompt = row.get("prompt", "")
        comps = row.get("candidates", [])
        candidates = []
        for cid, expansion in enumerate(comps):
            item = scored.get((prompt_id, cid))
            if item is None or item_score(item) is None:
                continue
            candidates.append((cid, expansion, item))

        if len(candidates) < 2:
            skip_reasons["prompt_lt_2_scored_candidates"] += 1
            continue
        valid_prompt_count += 1

        prompt_pairs = []
        for c_cid, c_text, c_item in candidates:
            c_score = item_score(c_item)
            if c_score is None:
                continue
            ok, reason = chosen_passes_guardrails(c_item, cfg)
            if not ok:
                skip_reasons[reason] += 1
                continue

            for r_cid, r_text, r_item in candidates:
                if c_cid == r_cid:
                    continue
                r_score = item_score(r_item)
                if r_score is None:
                    continue
                gap = c_score - r_score
                if gap < cfg.min_score_gap:
                    skip_reasons[f"pair_gap_lt_{cfg.min_score_gap}"] += 1
                    continue

                ok, reason = chosen_does_not_lose_core(c_item, r_item)
                if not ok:
                    skip_reasons[reason] += 1
                    continue

                if cfg.prefer_hard_negative_pairs:
                    # Do not require a hard negative, but give them priority.
                    pass

                prompt_pairs.append({
                    "prompt": prompt,
                    "chosen": c_text,
                    "rejected": r_text,
                    "chosen_score": c_score,
                    "rejected_score": r_score,
                    "score_gap": gap,
                    "prompt_id": prompt_id,
                    "chosen_candidate_id": c_cid,
                    "rejected_candidate_id": r_cid,
                    "chosen_scores": score_payload(c_item),
                    "rejected_scores": score_payload(r_item),
                    "pair_priority": pair_priority(c_item, r_item, gap, cfg),
                })

        # Take the best N pairs for this prompt. This prevents one prompt from
        # producing 20 near-duplicate pairs and dominating DPO.
        prompt_pairs.sort(key=lambda p: p["pair_priority"], reverse=True)
        selected = prompt_pairs[: cfg.max_pairs_per_prompt]
        all_pairs.extend(selected)
        per_prompt_counts.append(len(selected))

    # Global target cap: preserve diversity by shuffling among already capped
    # per-prompt pairs, but prefer high-priority pairs if we have far too many.
    if len(all_pairs) > cfg.target_pairs:
        all_pairs.sort(key=lambda p: p["pair_priority"], reverse=True)
        all_pairs = all_pairs[: cfg.target_pairs]
        random.shuffle(all_pairs)
    else:
        random.shuffle(all_pairs)

    for p in all_pairs:
        p.pop("pair_priority", None)

    print("\n[build] summary")
    print(f"  valid prompts with >=2 scored candidates: {valid_prompt_count}")
    print(f"  output pairs: {len(all_pairs)}")
    print(f"  target pairs: {cfg.target_pairs}")
    if per_prompt_counts:
        print(f"  prompts producing >=1 selected pair: {sum(1 for x in per_prompt_counts if x > 0)}")
        print(f"  mean selected pairs/prompt: {sum(per_prompt_counts)/len(per_prompt_counts):.2f}")
    print("\n[build] skip reasons, top 20")
    for reason, count in skip_reasons.most_common(20):
        print(f"  {reason}: {count}")

    write_jsonl(all_pairs, cfg.output_path)
    print(f"\n[build] wrote {len(all_pairs)} DPO pairs to {cfg.output_path}")

    if cfg.audit_path:
        write_pair_audit(all_pairs, cfg.audit_path, n=min(100, len(all_pairs)))

    return all_pairs


def write_pair_audit(pairs: List[dict], audit_path: str, n: int = 100) -> None:
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# DPO Pair Audit Sample\n\n")
        f.write(f"Showing first {n} pairs. Manually inspect these before training.\n\n")
        for i, row in enumerate(pairs[:n], start=1):
            f.write(f"## Pair {i}\n\n")
            f.write(f"**Prompt**: `{row['prompt']}`\n\n")
            f.write(f"**Chosen** ({row['chosen_score']}): {row['chosen']}\n\n")
            f.write(f"**Rejected** ({row['rejected_score']}): {row['rejected']}\n\n")
            f.write("**Chosen scores**:\n\n")
            f.write("```json\n" + json.dumps(row.get("chosen_scores", {}), indent=2) + "\n```\n\n")
            f.write("**Rejected scores**:\n\n")
            f.write("```json\n" + json.dumps(row.get("rejected_scores", {}), indent=2) + "\n```\n\n")
    print(f"[audit] wrote audit sample to {audit_path}")


# -----------------------------------------------------------------------------
# Stage 2: train DPO with a custom Trainer
# -----------------------------------------------------------------------------

class DPODataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int = 256, limit: Optional[int] = None):
        self.rows = list(read_jsonl(path))
        if limit:
            self.rows = self.rows[:limit]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def _encode(self, prompt: str, completion: str) -> Dict[str, List[int]]:
        # We compute loss only on completion tokens.
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        comp_ids = self.tokenizer(completion + self.tokenizer.eos_token, add_special_tokens=False).input_ids
        input_ids = prompt_ids + comp_ids
        labels = [-100] * len(prompt_ids) + comp_ids

        if len(input_ids) > self.max_length:
            # Prefer preserving the completion. Truncate prompt from the left if needed.
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
            # If truncation ate into completion labels, labels still work; but avoid all -100.
            if all(x == -100 for x in labels):
                input_ids = (prompt_ids + comp_ids)[-self.max_length:]
                labels = ([-100] * len(prompt_ids) + comp_ids)[-self.max_length:]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        prompt = r["prompt"]
        chosen = self._encode(prompt, r["chosen"])
        rejected = self._encode(prompt, r["rejected"])
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }


class DPOCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def _pad(self, examples: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(x) for x in examples)
        out = []
        for x in examples:
            out.append(x + [pad_value] * (max_len - len(x)))
        return torch.tensor(out, dtype=torch.long)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        out = {}
        for prefix in ["chosen", "rejected"]:
            out[f"{prefix}_input_ids"] = self._pad([b[f"{prefix}_input_ids"] for b in batch], self.pad_id)
            out[f"{prefix}_attention_mask"] = self._pad([b[f"{prefix}_attention_mask"] for b in batch], 0)
            out[f"{prefix}_labels"] = self._pad([b[f"{prefix}_labels"] for b in batch], -100)
        return out


def sequence_logps(model, input_ids, attention_mask, labels) -> torch.Tensor:
    # Sum token log probs over non-masked labels.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    loss_mask = shifted_labels.ne(-100)

    safe_labels = shifted_labels.clone()
    safe_labels[~loss_mask] = 0
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * loss_mask).sum(dim=-1)


class SimpleDPOTrainer(Trainer):
    def __init__(self, *args, ref_model=None, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        if self.ref_model is not None:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_logps = sequence_logps(
            model,
            inputs["chosen_input_ids"],
            inputs["chosen_attention_mask"],
            inputs["chosen_labels"],
        )
        rejected_logps = sequence_logps(
            model,
            inputs["rejected_input_ids"],
            inputs["rejected_attention_mask"],
            inputs["rejected_labels"],
        )

        with torch.no_grad():
            ref_chosen_logps = sequence_logps(
                self.ref_model,
                inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"],
                inputs["chosen_labels"],
            )
            ref_rejected_logps = sequence_logps(
                self.ref_model,
                inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"],
                inputs["rejected_labels"],
            )

        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits)
        loss = losses.mean()

        # Useful metrics. Trainer will not always display all of these depending
        # on HF version, but returning them in outputs helps debugging.
        reward_chosen = self.beta * (chosen_logps - ref_chosen_logps)
        reward_rejected = self.beta * (rejected_logps - ref_rejected_logps)
        outputs = {
            "loss": loss.detach(),
            "dpo_accuracy": (logits > 0).float().mean().detach(),
            "reward_margin": (reward_chosen - reward_rejected).mean().detach(),
        }
        return (loss, outputs) if return_outputs else loss


def load_causal_lm_for_training(
    model_name_or_path: str,
    load_in_4bit: bool,
    bf16: bool,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto" if load_in_4bit else None,
    )

    if use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. Install peft or remove --use_lora.")
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        # GPT-2 fallback module names.
        if "gpt2" in model_name_or_path.lower():
            target_modules = ["c_attn", "c_proj", "c_fc"]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model


def train_dpo(args) -> None:
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    ensure_tokenizer_pad(tokenizer)

    policy = load_causal_lm_for_training(
        args.model_name_or_path,
        load_in_4bit=args.load_in_4bit,
        bf16=args.bf16,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Reference model should be frozen SFT model.
    ref = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto" if args.load_in_4bit else None,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if args.load_in_4bit else None,
    )
    ref.eval()

    dataset = DPODataset(args.dataset_path, tokenizer, max_length=args.max_length, limit=args.limit_train_examples)
    if len(dataset) == 0:
        raise RuntimeError("DPO dataset is empty.")

    # Simple prompt-level-ish split after shuffling. If you need exact prompt split,
    # build separate train/eval JSONLs upstream.
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    eval_n = max(1, int(len(indices) * args.eval_fraction)) if args.eval_fraction > 0 else 0
    eval_indices = set(indices[:eval_n])
    train_rows = [dataset.rows[i] for i in indices[eval_n:]]
    eval_rows = [dataset.rows[i] for i in indices[:eval_n]] if eval_n else []

    tmp_dir = Path(args.output_dir) / "tmp_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    train_path = tmp_dir / "train.jsonl"
    eval_path = tmp_dir / "eval.jsonl"
    write_jsonl(train_rows, train_path)
    if eval_rows:
        write_jsonl(eval_rows, eval_path)

    train_ds = DPODataset(train_path, tokenizer, max_length=args.max_length)
    eval_ds = DPODataset(eval_path, tokenizer, max_length=args.max_length) if eval_rows else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        remove_unused_columns=False,
    )

    trainer = SimpleDPOTrainer(
        model=policy,
        ref_model=ref,
        beta=args.beta,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DPOCollator(tokenizer),
    )

    trainer.train()
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train] saved final model to {final_dir}")


# -----------------------------------------------------------------------------
# Stage 3: Layer 3 review
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def text_flags(prompt: str, text: str) -> Dict[str, Any]:
    return {
        "tokens": token_len(text),
        "magic_word_hits": count_phrase_hits(text, MAGIC_WORDS),
        "artist_hits": count_phrase_hits(text, ARTIST_WORDS),
        "repeated_3grams": repeated_ngram_count(text, 3),
        "prompt_copy_ratio": round(prompt_copy_ratio(prompt, text), 3),
    }


def heuristic_risk(flags: dict) -> float:
    return (
        2.0 * flags["magic_word_hits"]
        + 2.5 * flags["artist_hits"]
        + 1.0 * flags["repeated_3grams"]
        + max(0, flags["tokens"] - 70) / 10
        + max(0, flags["prompt_copy_ratio"] - 0.45) * 4
    )


def load_eval_prompts(train_path: str, n: int, seed: int, offset: int = 0) -> List[str]:
    df = pd.read_csv(train_path)
    prompts = [str(x).split("</s>")[0] + "</s>" for x in df["text"].tolist()]
    # Use an offset to avoid first rows if they were likely used for training.
    prompts = prompts[offset:] if offset else prompts
    rnd = random.Random(seed)
    rnd.shuffle(prompts)
    return prompts[:n]


def load_inference_model(model_path: str, base_model: Optional[str], bf16: bool, device: str):
    """Load either a full causal LM checkpoint or a PEFT adapter checkpoint."""
    dtype = torch.bfloat16 if bf16 else torch.float16
    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        if not PEFT_AVAILABLE:
            raise RuntimeError("PEFT adapter detected but peft is not installed.")
        if base_model is None:
            try:
                cfg = json.loads(adapter_config.read_text())
                base_model = cfg.get("base_model_name_or_path")
            except Exception:
                base_model = None
        if not base_model:
            raise RuntimeError(
                f"{model_path} looks like a PEFT adapter. Pass --dpo_base_model or --sft_base_model."
            )
        print(f"[layer3] loading base model {base_model} + adapter {model_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()
        if device != "cuda":
            model.to(device)
        tokenizer_path = model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return model, ensure_tokenizer_pad(tokenizer)

    print(f"[layer3] loading full model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    ).eval()
    if device != "cuda":
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, ensure_tokenizer_pad(tokenizer)


def layer3_review(args) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sft, tokenizer_sft = load_inference_model(args.sft_model, args.sft_base_model, args.bf16, device)
    dpo, tokenizer_dpo = load_inference_model(args.dpo_model, args.dpo_base_model, args.bf16, device)

    prompts = load_eval_prompts(args.train_path, args.num_prompts, args.seed, offset=args.prompt_offset)
    rows = []
    for prompt in tqdm(prompts, desc="generating layer3 samples"):
        sft_out = generate_one(sft, tokenizer_sft, prompt, args.max_new_tokens, args.temperature, args.top_p, device)
        dpo_out = generate_one(dpo, tokenizer_dpo, prompt, args.max_new_tokens, args.temperature, args.top_p, device)
        sft_flags = text_flags(prompt, sft_out)
        dpo_flags = text_flags(prompt, dpo_out)
        rows.append({
            "prompt": prompt,
            "sft": sft_out,
            "dpo": dpo_out,
            "sft_flags": sft_flags,
            "dpo_flags": dpo_flags,
            "dpo_risk": heuristic_risk(dpo_flags),
            "sft_risk": heuristic_risk(sft_flags),
        })

    # Sort by suspicious DPO outputs first for manual review.
    rows_sorted = sorted(rows, key=lambda r: (r["dpo_risk"] - r["sft_risk"], r["dpo_risk"]), reverse=True)
    write_layer3_markdown(rows_sorted, args.output_md)
    csv_path = str(Path(args.output_md).with_suffix(".csv"))
    pd.DataFrame([
        {
            "prompt": r["prompt"],
            "sft": r["sft"],
            "dpo": r["dpo"],
            **{f"sft_{k}": v for k, v in r["sft_flags"].items()},
            **{f"dpo_{k}": v for k, v in r["dpo_flags"].items()},
            "sft_risk": r["sft_risk"],
            "dpo_risk": r["dpo_risk"],
        }
        for r in rows_sorted
    ]).to_csv(csv_path, index=False)
    print(f"[layer3] wrote markdown review to {args.output_md}")
    print(f"[layer3] wrote CSV diagnostics to {csv_path}")


def write_layer3_markdown(rows: List[dict], output_md: str) -> None:
    path = Path(output_md)
    path.parent.mkdir(parents=True, exist_ok=True)

    def avg(key: str, field: str) -> float:
        if not rows:
            return 0.0
        return sum(r[key][field] for r in rows) / len(rows)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Layer 3 Review: SFT vs Judge-DPO\n\n")
        f.write("This file is sorted with the most suspicious DPO outputs first. ")
        f.write("Use it for manual inspection, not as an automatic pass/fail.\n\n")
        f.write("## Aggregate diagnostics\n\n")
        f.write("| Metric | SFT avg | DPO avg |\n")
        f.write("|---|---:|---:|\n")
        for field in ["tokens", "magic_word_hits", "artist_hits", "repeated_3grams", "prompt_copy_ratio"]:
            f.write(f"| {field} | {avg('sft_flags', field):.3f} | {avg('dpo_flags', field):.3f} |\n")
        f.write("\n")

        for i, r in enumerate(rows, start=1):
            f.write(f"## Example {i}\n\n")
            f.write(f"**Prompt**: `{r['prompt']}`\n\n")
            f.write("**SFT output**\n\n")
            f.write(f"> {r['sft']}\n\n")
            f.write("**DPO output**\n\n")
            f.write(f"> {r['dpo']}\n\n")
            f.write("**Flags**\n\n")
            f.write("```json\n")
            f.write(json.dumps({"sft": r["sft_flags"], "dpo": r["dpo_flags"]}, indent=2))
            f.write("\n```\n\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def add_build_pairs_parser(subparsers):
    p = subparsers.add_parser("build_pairs", help="Rebuild multiple DPO pairs from existing judge scores")
    p.add_argument("--scores", required=True, help="Path to openrouter_scores_mo.jsonl")
    p.add_argument("--candidates", required=True, help="Path to sft_candidates.jsonl")
    p.add_argument("--output", required=True, help="Output DPO JSONL path")
    p.add_argument("--audit", default=None, help="Optional markdown audit sample path")
    p.add_argument("--target_pairs", type=int, default=3000)
    p.add_argument("--max_pairs_per_prompt", type=int, default=4)
    p.add_argument("--min_score_gap", type=int, default=1)
    p.add_argument("--min_chosen_fidelity", type=int, default=5)
    p.add_argument("--min_chosen_style_fit", type=int, default=4)
    p.add_argument("--min_chosen_non_genericness", type=int, default=5)
    p.add_argument("--min_chosen_anti_magic_word_score", type=int, default=6)
    p.add_argument("--min_chosen_coherence", type=int, default=5)
    p.add_argument("--hard_failure_modes", default=",".join(sorted(DEFAULT_HARD_FAILURE_MODES)))
    p.add_argument("--prefer_hard_negative_pairs", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def add_train_dpo_parser(subparsers):
    p = subparsers.add_parser("train_dpo", help="Train DPO with a custom DPO loss")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--eval_fraction", type=float, default=0.05)
    p.add_argument("--limit_train_examples", type=int, default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", default="none")
    p.add_argument("--seed", type=int, default=42)

    # LoRA / QLoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p


def add_layer3_parser(subparsers):
    p = subparsers.add_parser("layer3", help="Generate SFT vs DPO side-by-side review")
    p.add_argument("--sft_model", required=True)
    p.add_argument("--dpo_model", required=True)
    p.add_argument("--sft_base_model", default=None, help="Base model if --sft_model is a PEFT adapter")
    p.add_argument("--dpo_base_model", default=None, help="Base model if --dpo_model is a PEFT adapter")
    p.add_argument("--train_path", required=True, help="CSV with text column in prompt</s>completion format")
    p.add_argument("--output_md", required=True)
    p.add_argument("--num_prompts", type=int, default=50)
    p.add_argument("--prompt_offset", type=int, default=1000, help="Skip first N prompts to reduce train-set overlap")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=123)
    return p


def main():
    parser = argparse.ArgumentParser(description="DPO judge pipeline: rebuild pairs, train DPO, Layer 3 compare")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_build_pairs_parser(subparsers)
    add_train_dpo_parser(subparsers)
    add_layer3_parser(subparsers)
    args = parser.parse_args()

    if args.command == "build_pairs":
        hard_modes = {
            m.strip().lower().replace(" ", "_").replace("-", "_")
            for m in args.hard_failure_modes.split(",")
            if m.strip()
        }
        cfg = PairBuildConfig(
            scores_path=args.scores,
            candidates_path=args.candidates,
            output_path=args.output,
            audit_path=args.audit,
            target_pairs=args.target_pairs,
            max_pairs_per_prompt=args.max_pairs_per_prompt,
            min_score_gap=args.min_score_gap,
            min_chosen_fidelity=args.min_chosen_fidelity,
            min_chosen_style_fit=args.min_chosen_style_fit,
            min_chosen_non_genericness=args.min_chosen_non_genericness,
            min_chosen_anti_magic_word_score=args.min_chosen_anti_magic_word_score,
            min_chosen_coherence=args.min_chosen_coherence,
            hard_failure_modes=hard_modes,
            prefer_hard_negative_pairs=args.prefer_hard_negative_pairs,
            seed=args.seed,
        )
        build_multiple_pairs(cfg)
    elif args.command == "train_dpo":
        train_dpo(args)
    elif args.command == "layer3":
        layer3_review(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
