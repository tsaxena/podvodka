"""
PPO model sanity check.

Runs a series of checks on a trained PPO checkpoint and prints PASS/WARN/FAIL
for each. No W&B or training dependencies required — only transformers + trl.

Usage:
    python sanity_check_ppo.py --model_path /path/to/ppo/best
    python sanity_check_ppo.py --model_path tsaxena/gpt2-large-ppo-prompt-tags
    python sanity_check_ppo.py --model_path ./best --val_path /path/to/val_strings.csv --n_samples 50
"""

import argparse
import sys
import textwrap
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────── helpers ───────────────────────────

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"


def _tag(label: str, color: str) -> str:
    return f"{BOLD}{color}[{label}]{RESET}"


PASS = _tag("PASS", GREEN)
WARN = _tag("WARN", YELLOW)
FAIL = _tag("FAIL", RED)

results: List[tuple] = []  # (check_name, status, detail)


def record(name: str, passed: Optional[bool], detail: str):
    """passed=True → PASS, passed=False → FAIL, passed=None → WARN."""
    if passed is True:
        tag = PASS
        key = "pass"
    elif passed is False:
        tag = FAIL
        key = "fail"
    else:
        tag = WARN
        key = "warn"
    print(f"  {tag} {detail}")
    results.append((name, key, detail))


def section(title: str):
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


# ─────────────────────────── checks ───────────────────────────

def check_model_loads(model_path: str, device: str):
    section("1. Model loading")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        record("tokenizer_load", True, f"Tokenizer loaded  vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        record("tokenizer_load", False, f"Tokenizer failed to load: {e}")
        return None, None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        total = sum(p.numel() for p in model.parameters()) / 1e6
        record("model_load", True, f"Model loaded  params={total:.1f}M  device={device}")
    except Exception as e:
        record("model_load", False, f"Model failed to load: {e}")
        return None, None

    return model, tokenizer


def check_forward_pass(model, tokenizer, device: str):
    section("2. Forward pass (single token prediction)")
    text = "a portrait of a woman</s>"
    enc = tokenizer(text, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        record("output_shape", True,
               f"logits shape={tuple(logits.shape)}  dtype={logits.dtype}")

        if torch.isnan(logits).any():
            record("nan_check", False, "NaN values found in logits")
        elif torch.isinf(logits).any():
            record("nan_check", False, "Inf values found in logits")
        else:
            record("nan_check", True, "No NaN/Inf in logits")

        # Check last-token probabilities are a valid distribution
        probs = F.softmax(logits[0, -1], dim=-1)
        top5_ids = probs.topk(5).indices.tolist()
        top5_tok = [tokenizer.decode([i]) for i in top5_ids]
        top5_p = probs[top5_ids].tolist()
        top_str = "  ".join(f"{t!r}:{p:.3f}" for t, p in zip(top5_tok, top5_p))
        record("top5_tokens", True, f"Top-5 next tokens: {top_str}")
    except Exception as e:
        record("forward_pass", False, f"Forward pass raised: {e}")


# ---- a small set of representative T2I prompts (same domain as training) ----
FIXED_PROMPTS = [
    "a portrait of a young woman</s>",
    "cyberpunk cityscape at night</s>",
    "a cozy cabin in the autumn forest</s>",
    "an astronaut floating in outer space</s>",
    "oil painting of a medieval knight</s>",
    "close-up macro photo of a hummingbird</s>",
    "a futuristic city with flying cars</s>",
    "watercolor painting of cherry blossoms</s>",
]


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: List[str], device: str,
                   max_new_tokens: int = 80) -> List[str]:
    completions = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
    return completions


def check_generation_quality(model, tokenizer, device: str):
    section("3. Generation quality (fixed prompts)")
    completions = generate_batch(model, tokenizer, FIXED_PROMPTS, device)

    empty = sum(1 for c in completions if c.strip() == "")
    record("empty_outputs", empty == 0,
           f"Empty completions: {empty}/{len(completions)}")

    # Detect degenerate repetition: any completion where 1 token > 60% of output
    degenerate = 0
    total_len = 0
    for comp in completions:
        toks = tokenizer(comp, add_special_tokens=False)["input_ids"]
        total_len += len(toks)
        if len(toks) > 5:
            from collections import Counter
            most_common_count = Counter(toks).most_common(1)[0][1]
            if most_common_count / len(toks) > 0.6:
                degenerate += 1
    avg_len = total_len / max(len(completions), 1)

    record("repetition_check", degenerate == 0,
           f"Degenerate repetition: {degenerate}/{len(completions)}  avg_len={avg_len:.1f} tokens")

    # Print all generations for manual inspection
    print()
    for prompt, comp in zip(FIXED_PROMPTS, completions):
        short_p = prompt.replace("</s>", "")
        short_c = comp.replace("</s>", "").strip()
        wrapped = textwrap.fill(short_c, width=70, subsequent_indent="          ")
        print(f"  Prompt: {short_p!r}")
        print(f"  Output: {wrapped}")
        print()

    return completions


def check_reward_scores(completions: List[str], reward_pipeline, batch_size: int = 8):
    section("4. Reward model scoring")
    reward_texts = [
        p + "</s>" + c
        for p, c in zip(FIXED_PROMPTS, completions)
    ]
    try:
        raw = reward_pipeline(
            reward_texts,
            function_to_apply="none",
            batch_size=batch_size,
            truncation=True,
        )
        scores = [o["score"] for o in raw]
    except Exception as e:
        record("reward_scoring", False, f"Reward pipeline raised: {e}")
        return []

    mean_r = sum(scores) / len(scores)
    min_r = min(scores)
    max_r = max(scores)
    std_r = (sum((s - mean_r) ** 2 for s in scores) / len(scores)) ** 0.5

    record("reward_mean", mean_r > 0,
           f"Mean reward={mean_r:+.3f}  min={min_r:+.3f}  max={max_r:+.3f}  std={std_r:.3f}")

    if std_r < 0.01:
        record("reward_diversity", None,
               f"Reward std={std_r:.4f} — very low, possible mode collapse")
    else:
        record("reward_diversity", True,
               f"Reward std={std_r:.3f} — good diversity")

    print()
    for prompt, comp, score in zip(FIXED_PROMPTS, completions, scores):
        short_p = prompt.replace("</s>", "")
        short_c = comp.replace("</s>", "").strip()[:60]
        print(f"  {score:+.3f}  {short_p!r} → {short_c!r}")

    return scores


def check_ppo_vs_base(model, tokenizer, base_model_path: str,
                      reward_pipeline, device: str):
    section("5. PPO vs base model reward comparison")
    print("  Loading base model for comparison...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    except Exception as e:
        record("base_model_load", False, f"Base model failed to load: {e}")
        return

    ppo_completions = generate_batch(model, tokenizer, FIXED_PROMPTS, device)
    base_completions = generate_batch(base_model, tokenizer, FIXED_PROMPTS, device)

    del base_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    def score(prompts, completions):
        texts = [p + "</s>" + c for p, c in zip(prompts, completions)]
        raw = reward_pipeline(texts, function_to_apply="none",
                              batch_size=8, truncation=True)
        return [o["score"] for o in raw]

    ppo_scores = score(FIXED_PROMPTS, ppo_completions)
    base_scores = score(FIXED_PROMPTS, base_completions)

    ppo_mean = sum(ppo_scores) / len(ppo_scores)
    base_mean = sum(base_scores) / len(base_scores)
    delta = ppo_mean - base_mean
    wins = sum(p > b for p, b in zip(ppo_scores, base_scores))

    record("reward_improvement", delta > 0,
           f"PPO mean={ppo_mean:+.3f}  base mean={base_mean:+.3f}  delta={delta:+.3f}")
    record("win_rate", wins / len(ppo_scores) >= 0.5,
           f"PPO win rate vs base: {wins}/{len(ppo_scores)} = {wins/len(ppo_scores):.0%}")

    print()
    print(f"  {'prompt':<38} {'base':>7} {'ppo':>7} {'Δ':>7}")
    print(f"  {'─'*38} {'─'*7} {'─'*7} {'─'*7}")
    for prompt, bs, ps in zip(FIXED_PROMPTS, base_scores, ppo_scores):
        short_p = prompt.replace("</s>", "")[:37]
        print(f"  {short_p:<38} {bs:>+7.3f} {ps:>+7.3f} {ps-bs:>+7.3f}")


def check_kl_divergence(model, tokenizer, base_model_path: str, device: str):
    section("6. Approximate KL divergence from base model")
    print("  Loading base model for KL check...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device).eval()
    except Exception as e:
        record("kl_base_load", False, f"Base model failed to load: {e}")
        return

    kl_vals = []
    test_texts = [p + " cinematic lighting, ultra detailed" for p in FIXED_PROMPTS[:4]]
    with torch.no_grad():
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256).to(device)
            ppo_logits = model(**enc).logits[0]
            base_logits = base_model(**enc).logits[0]
            ppo_log_probs = F.log_softmax(ppo_logits.float(), dim=-1)
            base_probs = F.softmax(base_logits.float(), dim=-1)
            kl = F.kl_div(ppo_log_probs, base_probs, reduction="batchmean").item()
            kl_vals.append(kl)

    del base_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    mean_kl = sum(kl_vals) / len(kl_vals)
    # Typical trained PPO: KL 0.5–5.0 nats is healthy.
    # <0.1 → barely trained; >20 → possible reward hacking.
    if mean_kl < 0.1:
        passed = None
        note = "very low — may indicate minimal training or loading issue"
    elif mean_kl > 20:
        passed = None
        note = "very high — possible reward hacking"
    else:
        passed = True
        note = "in healthy range"

    record("kl_divergence", passed,
           f"Mean token KL(ppo‖base)={mean_kl:.3f} nats — {note}")


def check_val_corpus(model, tokenizer, reward_pipeline, val_path: str,
                     n_samples: int, device: str):
    section(f"7. Validation corpus check  (n={n_samples})")
    try:
        import pandas as pd
        df = pd.read_csv(val_path)
        raw_texts = df["text"].tolist()
    except Exception as e:
        record("val_data_load", False, f"Failed to load val data: {e}")
        return

    prompts = [t.split("</s>")[0] + "</s>" for t in raw_texts][:n_samples]
    record("val_data_load", True, f"Loaded {len(prompts)} val prompts")

    completions = generate_batch(model, tokenizer, prompts, device)
    reward_texts = [p + "</s>" + c for p, c in zip(prompts, completions)]

    try:
        raw = reward_pipeline(
            reward_texts, function_to_apply="none",
            batch_size=16, truncation=True,
        )
        scores = [o["score"] for o in raw]
    except Exception as e:
        record("val_reward_scoring", False, f"Reward scoring failed: {e}")
        return

    mean_r = sum(scores) / len(scores)
    std_r = (sum((s - mean_r) ** 2 for s in scores) / len(scores)) ** 0.5
    min_r = min(scores)
    max_r = max(scores)
    frac_pos = sum(1 for s in scores if s > 0) / len(scores)

    record("val_reward_mean", mean_r > 0,
           f"mean={mean_r:+.3f}  std={std_r:.3f}  min={min_r:+.3f}  max={max_r:+.3f}")
    record("val_frac_positive", frac_pos >= 0.5,
           f"Fraction of positive-reward completions: {frac_pos:.0%}")


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sanity-check a trained PPO model.")
    parser.add_argument("--model_path", type=str,
                        default="tsaxena/gpt2-large-ppo-prompt-tags",
                        help="Local checkpoint dir or HuggingFace repo of the PPO model.")
    parser.add_argument("--base_model_path", type=str,
                        default="tsaxena/gpt2-large-prompt-tags",
                        help="Base SFT model for comparison & KL check.")
    parser.add_argument("--reward_model_path", type=str,
                        default="toloka/prompts_reward_model",
                        help="Reward model used during PPO training.")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Optional path to val_strings.csv for corpus-level stats.")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of val prompts to sample for check 7.")
    parser.add_argument("--skip_base_compare", action="store_true",
                        help="Skip checks 5 & 6 that require loading the base model "
                             "(saves memory on small GPUs).")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu', 'cuda', 'cuda:0', etc. "
                             "Auto-detects if not set.")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{BOLD}PPO Model Sanity Check{RESET}")
    print(f"  model_path  : {args.model_path}")
    print(f"  base_model  : {args.base_model_path}")
    print(f"  reward_model: {args.reward_model_path}")
    print(f"  device      : {device}")

    # ── Check 1: load ──
    model, tokenizer = check_model_loads(args.model_path, device)
    if model is None:
        print(f"\n{FAIL} Could not load model — aborting.")
        sys.exit(1)

    # ── Check 2: forward pass ──
    check_forward_pass(model, tokenizer, device)

    # ── Check 3: generation quality ──
    completions = check_generation_quality(model, tokenizer, device)

    # ── Load reward pipeline (used in checks 4, 5, 7) ──
    section("Loading reward model...")
    try:
        reward_device = 0 if device == "cuda" else -1
        reward_pipeline = pipeline(
            "text-classification",
            model=args.reward_model_path,
            device=reward_device,
        )
        print(f"  Reward model loaded on device={reward_device}")
        reward_available = True
    except Exception as e:
        print(f"  {WARN} Could not load reward model: {e}")
        print("  Skipping reward-dependent checks (4, 5, 7).")
        reward_available = False

    # ── Check 4: reward scores ──
    if reward_available:
        check_reward_scores(completions, reward_pipeline)

    # ── Checks 5 & 6: comparison to base ──
    if not args.skip_base_compare:
        if reward_available:
            check_ppo_vs_base(model, tokenizer, args.base_model_path,
                              reward_pipeline, device)
        check_kl_divergence(model, tokenizer, args.base_model_path, device)
    else:
        print(f"\n  Skipping checks 5 & 6 (--skip_base_compare).")

    # ── Check 7: val corpus ──
    if args.val_path and reward_available:
        check_val_corpus(model, tokenizer, reward_pipeline,
                         args.val_path, args.n_samples, device)
    elif not args.val_path:
        print(f"\n  Skipping check 7 (no --val_path provided).")

    # ─────────────── Summary ───────────────
    section("Summary")
    n_pass = sum(1 for _, s, _ in results if s == "pass")
    n_warn = sum(1 for _, s, _ in results if s == "warn")
    n_fail = sum(1 for _, s, _ in results if s == "fail")
    print(f"  {PASS} {n_pass}   {WARN} {n_warn}   {FAIL} {n_fail}\n")

    for name, status, detail in results:
        if status == "pass":
            tag = PASS
        elif status == "warn":
            tag = WARN
        else:
            tag = FAIL
        print(f"  {tag} {name}: {detail}")

    print()
    if n_fail > 0:
        print(f"{BOLD}{RED}Model has issues — see FAIL entries above.{RESET}")
        sys.exit(1)
    elif n_warn > 0:
        print(f"{BOLD}{YELLOW}Model passed with warnings — review WARN entries.{RESET}")
    else:
        print(f"{BOLD}{GREEN}All checks passed.{RESET}")


if __name__ == "__main__":
    main()
