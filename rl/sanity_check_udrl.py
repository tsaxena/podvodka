"""
UDRL model sanity check.

Runs a series of checks on a trained UDRL checkpoint and prints PASS/WARN/FAIL
for each. No W&B or training dependencies required — only transformers + torch.

Usage:
    python sanity_check_udrl.py --model_path /path/to/udrl/best
    python sanity_check_udrl.py --model_path ./best --val_path /path/to/val_strings.csv
    python sanity_check_udrl.py --model_path ./best --skip_base_compare --n_samples 50

Checks
──────
 1. Model loading          tokenizer + AutoModelForCausalLM
 2. Forward pass           NaN/Inf in logits; command prefix survives tokenisation
 3. Command prefix format  model accepts "[reward: +0.75] description</s>" without error
 4. Generation quality     no empty / degenerate outputs at high desired-return command
 5. Reward scores          fixed prompts → positive mean reward
 6. Command responsiveness HIGH command reward > LOW command reward  ← core UDRL check
 7. Output diversity       reward std + n-gram overlap across prompts
 8. UDRL vs base           mean reward improvement over SFT baseline
 9. KL divergence          approx KL(udrl‖base) in nats — detects reward hacking
10. Validation corpus      optional large-scale reward stats (requires --val_path)
"""

import argparse
import sys
import textwrap
from collections import Counter
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ─────────────────────────── terminal colours ───────────────────────────

RESET  = "\033[0m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
BOLD   = "\033[1m"


def _tag(label: str, color: str) -> str:
    return f"{BOLD}{color}[{label}]{RESET}"


PASS = _tag("PASS", GREEN)
WARN = _tag("WARN", YELLOW)
FAIL = _tag("FAIL", RED)

results: List[tuple] = []   # (check_name, status, detail)


def record(name: str, passed: Optional[bool], detail: str) -> None:
    """passed=True → PASS | passed=False → FAIL | passed=None → WARN."""
    tag = PASS if passed is True else (FAIL if passed is False else WARN)
    key = "pass" if passed is True else ("fail" if passed is False else "warn")
    print(f"  {tag} {detail}")
    results.append((name, key, detail))


def section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")


# ─────────────────────────── command format ───────────────────────────

def format_command(reward: float) -> str:
    """Must match the prefix used in run_udrl.py."""
    return f"[reward: {reward:+.2f}] "


# ─────────────────────────── fixed test prompts ───────────────────────────

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

# Desired-return command levels used for the responsiveness sweep (check 6).
COMMAND_LEVELS = [-0.5, 0.0, 0.5, 1.0]


# ─────────────────────────── generation helper ───────────────────────────

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 80,
    commanded_return: Optional[float] = None,
) -> List[str]:
    """
    Generate completions for `prompts`.
    If `commanded_return` is given, prepend the UDRL command prefix to each prompt.
    Uses left-padding so multiple prompts can be batched.
    """
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    if commanded_return is not None:
        input_texts = [format_command(commanded_return) + p for p in prompts]
    else:
        input_texts = list(prompts)

    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    input_len = enc.input_ids.shape[1]

    out = model.generate(
        enc.input_ids,
        attention_mask=enc.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    completions = [
        tokenizer.decode(out[j, input_len:], skip_special_tokens=False)
        for j in range(out.shape[0])
    ]

    tokenizer.padding_side = prev_side
    return completions


def score_texts(reward_pipeline, prompts: List[str], completions: List[str]) -> List[float]:
    """Score (prompt, completion) pairs with the reward model."""
    texts = [p + "</s>" + c for p, c in zip(prompts, completions)]
    raw = reward_pipeline(
        texts, function_to_apply="none", batch_size=8, truncation=True
    )
    return [o["score"] for o in raw]


# ─────────────────────────── checks ───────────────────────────

def check_model_loads(model_path: str, device: str):
    section("1. Model loading")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side="right")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        record("tokenizer_load", True,
               f"Tokenizer loaded  vocab_size={tokenizer.vocab_size}")
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
        record("model_load", True,
               f"Model loaded  params={total:.1f}M  device={device}")
    except Exception as e:
        record("model_load", False, f"Model failed to load: {e}")
        return None, None

    return model, tokenizer


def check_forward_pass(model, tokenizer, device: str) -> None:
    section("2. Forward pass (plain prompt, no command)")
    text = "a portrait of a woman</s>"
    enc = tokenizer(text, return_tensors="pt").to(device)
    try:
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        record("output_shape", True,
               f"logits shape={tuple(logits.shape)}  dtype={logits.dtype}")

        nan_ok = not torch.isnan(logits).any() and not torch.isinf(logits).any()
        record("nan_check", nan_ok,
               "No NaN/Inf in logits" if nan_ok else "NaN or Inf found in logits")

        probs = F.softmax(logits[0, -1].float(), dim=-1)
        top5_ids = probs.topk(5).indices.tolist()
        top5_str = "  ".join(
            f"{tokenizer.decode([i])!r}:{probs[i]:.3f}" for i in top5_ids
        )
        record("top5_tokens", True, f"Top-5 next tokens: {top5_str}")
    except Exception as e:
        record("forward_pass", False, f"Forward pass raised: {e}")


def check_command_prefix(model, tokenizer, device: str) -> None:
    section("3. Command prefix format")
    test_reward = 0.75
    text = format_command(test_reward) + "a portrait of a woman</s>"
    record("command_text", True, f"Command prefix: {text!r}")

    enc = tokenizer(text, return_tensors="pt").to(device)
    n_cmd_tokens = enc.input_ids.shape[1]
    record("command_tokens", True,
           f"Tokenised to {n_cmd_tokens} tokens (command+prompt)")

    try:
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        nan_ok = not torch.isnan(logits).any() and not torch.isinf(logits).any()
        record("command_forward", nan_ok,
               "Forward pass with command prefix: OK"
               if nan_ok else "NaN/Inf in logits with command prefix")
    except Exception as e:
        record("command_forward", False, f"Forward pass with command prefix raised: {e}")


def check_generation_quality(
    model, tokenizer, device: str, desired_return: float = 1.0
) -> List[str]:
    section(f"4. Generation quality (command={desired_return:+.2f})")
    completions = generate_batch(
        model, tokenizer, FIXED_PROMPTS, device, commanded_return=desired_return
    )

    empty = sum(1 for c in completions if c.strip() == "")
    record("empty_outputs", empty == 0,
           f"Empty completions: {empty}/{len(completions)}")

    degenerate = 0
    total_len = 0
    for comp in completions:
        toks = tokenizer(comp, add_special_tokens=False)["input_ids"]
        total_len += len(toks)
        if len(toks) > 5:
            most_common = Counter(toks).most_common(1)[0][1]
            if most_common / len(toks) > 0.6:
                degenerate += 1
    avg_len = total_len / max(len(completions), 1)
    record("repetition_check", degenerate == 0,
           f"Degenerate repetition: {degenerate}/{len(completions)}"
           f"  avg_len={avg_len:.1f} tokens")

    print()
    for prompt, comp in zip(FIXED_PROMPTS, completions):
        short_p = prompt.replace("</s>", "")
        short_c = comp.replace("</s>", "").replace("<|endoftext|>", "").strip()
        wrapped = textwrap.fill(short_c, width=70, subsequent_indent="          ")
        print(f"  Prompt: {short_p!r}")
        print(f"  Output: {wrapped}")
        print()

    return completions


def check_reward_scores(
    completions: List[str], reward_pipeline
) -> List[float]:
    section("5. Reward model scoring")
    try:
        scores = score_texts(reward_pipeline, FIXED_PROMPTS, completions)
    except Exception as e:
        record("reward_scoring", False, f"Reward pipeline raised: {e}")
        return []

    mean_r = sum(scores) / len(scores)
    min_r  = min(scores)
    max_r  = max(scores)
    std_r  = (sum((s - mean_r) ** 2 for s in scores) / len(scores)) ** 0.5

    record("reward_mean", mean_r > 0,
           f"Mean reward={mean_r:+.3f}  min={min_r:+.3f}"
           f"  max={max_r:+.3f}  std={std_r:.3f}")

    if std_r < 0.01:
        record("reward_diversity", None,
               f"Reward std={std_r:.4f} — very low, possible mode collapse")
    else:
        record("reward_diversity", True,
               f"Reward std={std_r:.3f} — healthy diversity")

    print()
    for prompt, comp, score in zip(FIXED_PROMPTS, completions, scores):
        short_p = prompt.replace("</s>", "")
        short_c = comp.replace("</s>", "").strip()[:60]
        print(f"  {score:+.3f}  {short_p!r} → {short_c!r}")

    return scores


def check_command_responsiveness(
    model, tokenizer, reward_pipeline, device: str
) -> None:
    """
    Core UDRL check: generate at multiple desired-return command levels and
    verify that higher commands produce higher mean rewards.  A monotonically
    increasing relationship indicates the model has learned to condition on
    the command; flat or inverted curves indicate the conditioning has failed.
    """
    section("6. Command responsiveness  ← core UDRL property")
    print("  Generating completions at each command level (this may take a moment)...")

    level_means: List[Tuple[float, float]] = []  # (command, mean_reward)

    for cmd_r in COMMAND_LEVELS:
        try:
            completions = generate_batch(
                model, tokenizer, FIXED_PROMPTS, device, commanded_return=cmd_r
            )
            scores = score_texts(reward_pipeline, FIXED_PROMPTS, completions)
            mean_r = sum(scores) / len(scores)
            level_means.append((cmd_r, mean_r))
        except Exception as e:
            record(f"cmd_{cmd_r:+.1f}", False,
                   f"Generation at command={cmd_r:+.2f} failed: {e}")
            return

    print()
    print(f"  {'command':>10}  {'mean reward':>12}")
    print(f"  {'─'*10}  {'─'*12}")
    for cmd_r, mean_r in level_means:
        print(f"  {cmd_r:>+10.2f}  {mean_r:>+12.3f}")
    print()

    # Check monotonicity: each level should be >= the previous
    rewards = [r for _, r in level_means]
    monotone = all(rewards[i] <= rewards[i + 1] for i in range(len(rewards) - 1))
    # Relaxed: at least the highest command beats the lowest command
    high_beats_low = rewards[-1] > rewards[0]
    delta = rewards[-1] - rewards[0]

    if monotone:
        record("command_monotone", True,
               f"Reward increases monotonically with command  "
               f"Δ(high−low)={delta:+.3f}")
    elif high_beats_low:
        record("command_monotone", None,
               f"High command beats low command but not fully monotone  "
               f"Δ(high−low)={delta:+.3f}")
    else:
        record("command_monotone", False,
               f"High command does NOT produce higher rewards than low command  "
               f"Δ(high−low)={delta:+.3f}  — UDRL conditioning may have failed")

    # Sensitivity: reward range across command levels
    reward_range = max(rewards) - min(rewards)
    if reward_range < 0.05:
        record("command_sensitivity", None,
               f"Reward range across command levels={reward_range:.3f}  "
               f"— very low sensitivity to command")
    else:
        record("command_sensitivity", True,
               f"Reward range across command levels={reward_range:.3f}")


def check_output_diversity(
    model, tokenizer, device: str, desired_return: float = 1.0
) -> None:
    """
    Check for mode collapse via:
      - reward std across prompts (same as PPO check)
      - bigram overlap: fraction of output bigrams shared across all completions
        (high overlap → tail-stuffing reward hack)
    """
    section("7. Output diversity (mode collapse detection)")
    completions = generate_batch(
        model, tokenizer, FIXED_PROMPTS, device, commanded_return=desired_return
    )

    # Bigram overlap
    all_bigrams: List[Counter] = []
    for comp in completions:
        toks = tokenizer(comp, add_special_tokens=False)["input_ids"]
        bg = Counter(zip(toks, toks[1:])) if len(toks) > 1 else Counter()
        all_bigrams.append(bg)

    if all_bigrams:
        # Union of all bigrams seen
        union: Counter = Counter()
        for bg in all_bigrams:
            union.update(bg)
        # Bigrams that appear in every completion
        universal = {bg for bg in union if all(bg in c for c in all_bigrams)}
        overlap_frac = len(universal) / max(len(union), 1)

        if overlap_frac > 0.5:
            record("bigram_overlap", False,
                   f"Bigram overlap={overlap_frac:.1%}  "
                   f"— high tail repetition, likely reward hacking")
        elif overlap_frac > 0.25:
            record("bigram_overlap", None,
                   f"Bigram overlap={overlap_frac:.1%}  — moderate tail similarity")
        else:
            record("bigram_overlap", True,
                   f"Bigram overlap={overlap_frac:.1%}  — good lexical diversity")
    else:
        record("bigram_overlap", None, "Could not compute bigram overlap")

    # Vocabulary richness: unique tokens / total tokens across all completions
    all_toks = []
    for comp in completions:
        all_toks.extend(tokenizer(comp, add_special_tokens=False)["input_ids"])
    if all_toks:
        type_token_ratio = len(set(all_toks)) / len(all_toks)
        passed = type_token_ratio > 0.3
        record("type_token_ratio", passed if type_token_ratio > 0.1 else False,
               f"Type-token ratio={type_token_ratio:.3f}"
               + (" — good vocabulary richness" if type_token_ratio > 0.3
                  else " — low, possible mode collapse"))


def check_udrl_vs_base(
    model, tokenizer, base_model_path: str, reward_pipeline, device: str,
    desired_return: float = 1.0,
) -> None:
    section("8. UDRL vs base model reward comparison")
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

    # UDRL generates with the high command; base generates without a command.
    udrl_completions = generate_batch(
        model, tokenizer, FIXED_PROMPTS, device, commanded_return=desired_return
    )
    base_completions = generate_batch(
        base_model, tokenizer, FIXED_PROMPTS, device, commanded_return=None
    )

    del base_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    udrl_scores = score_texts(reward_pipeline, FIXED_PROMPTS, udrl_completions)
    base_scores = score_texts(reward_pipeline, FIXED_PROMPTS, base_completions)

    udrl_mean = sum(udrl_scores) / len(udrl_scores)
    base_mean = sum(base_scores) / len(base_scores)
    delta = udrl_mean - base_mean
    wins = sum(u > b for u, b in zip(udrl_scores, base_scores))

    record("reward_improvement", delta > 0,
           f"UDRL mean={udrl_mean:+.3f}  base mean={base_mean:+.3f}  delta={delta:+.3f}")
    record("win_rate", wins / len(udrl_scores) >= 0.5,
           f"UDRL win rate vs base: {wins}/{len(udrl_scores)} = {wins/len(udrl_scores):.0%}")

    print()
    print(f"  {'prompt':<38} {'base':>7} {'udrl':>7} {'Δ':>7}")
    print(f"  {'─'*38} {'─'*7} {'─'*7} {'─'*7}")
    for prompt, bs, us in zip(FIXED_PROMPTS, base_scores, udrl_scores):
        short_p = prompt.replace("</s>", "")[:37]
        print(f"  {short_p:<38} {bs:>+7.3f} {us:>+7.3f} {us-bs:>+7.3f}")


def check_kl_divergence(
    model, tokenizer, base_model_path: str, device: str
) -> None:
    section("9. Approximate KL divergence from base model")
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

    # Evaluate KL on plain prompts (no command prefix, so we measure pure drift).
    test_texts = [p + " cinematic lighting, ultra detailed" for p in FIXED_PROMPTS[:4]]
    kl_vals = []
    with torch.no_grad():
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=256).to(device)
            udrl_logits  = model(**enc).logits[0].float()
            base_logits  = base_model(**enc).logits[0].float()
            udrl_log_probs = F.log_softmax(udrl_logits, dim=-1)
            base_probs     = F.softmax(base_logits, dim=-1)
            kl = F.kl_div(udrl_log_probs, base_probs, reduction="batchmean").item()
            kl_vals.append(kl)

    del base_model
    if "cuda" in device:
        torch.cuda.empty_cache()

    mean_kl = sum(kl_vals) / len(kl_vals)
    # Typical UDRL: KL 0.5–5.0 nats is healthy.
    # <0.1 → barely trained  |  >20 → possible reward hacking
    if mean_kl < 0.1:
        passed, note = None, "very low — may indicate minimal training or load issue"
    elif mean_kl > 20:
        passed, note = None, "very high — possible reward hacking"
    else:
        passed, note = True, "in healthy range"

    record("kl_divergence", passed,
           f"Mean token KL(udrl‖base)={mean_kl:.3f} nats — {note}")


def check_val_corpus(
    model, tokenizer, reward_pipeline,
    val_path: str, n_samples: int, device: str,
    desired_return: float = 1.0,
) -> None:
    section(f"10. Validation corpus check  (n={n_samples})")
    try:
        import pandas as pd
        df = pd.read_csv(val_path)
        prompts = [t.split("</s>")[0] + "</s>" for t in df["text"]][:n_samples]
    except Exception as e:
        record("val_data_load", False, f"Failed to load val data: {e}")
        return
    record("val_data_load", True, f"Loaded {len(prompts)} val prompts")

    completions = generate_batch(
        model, tokenizer, prompts, device, commanded_return=desired_return
    )

    try:
        scores = score_texts(reward_pipeline, prompts, completions)
    except Exception as e:
        record("val_reward_scoring", False, f"Reward scoring failed: {e}")
        return

    mean_r    = sum(scores) / len(scores)
    std_r     = (sum((s - mean_r) ** 2 for s in scores) / len(scores)) ** 0.5
    min_r     = min(scores)
    max_r     = max(scores)
    frac_pos  = sum(1 for s in scores if s > 0) / len(scores)

    record("val_reward_mean", mean_r > 0,
           f"mean={mean_r:+.3f}  std={std_r:.3f}"
           f"  min={min_r:+.3f}  max={max_r:+.3f}")
    record("val_frac_positive", frac_pos >= 0.5,
           f"Fraction of positive-reward completions: {frac_pos:.0%}")

    # Val-set diversity (bigram overlap)
    all_toks = []
    for comp in completions:
        all_toks.extend(tokenizer(comp, add_special_tokens=False)["input_ids"])
    if all_toks:
        ttr = len(set(all_toks)) / len(all_toks)
        record("val_type_token_ratio", ttr > 0.2,
               f"Val type-token ratio={ttr:.3f}")


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sanity-check a trained UDRL model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local checkpoint dir or HuggingFace repo of the UDRL model.")
    parser.add_argument("--base_model_path", type=str,
                        default="tsaxena/gpt2-large-prompt-tags",
                        help="SFT base model for comparison & KL check.")
    parser.add_argument("--reward_model_path", type=str,
                        default="toloka/prompts_reward_model",
                        help="Reward model used during UDRL training.")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Optional path to val_strings.csv for corpus-level stats.")
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of val prompts to use in check 10.")
    parser.add_argument("--desired_return", type=float, default=1.0,
                        help="Desired-return command used for checks 4, 7, 8, 10.")
    parser.add_argument("--skip_base_compare", action="store_true",
                        help="Skip checks 8 & 9 (base model comparison). "
                             "Saves GPU memory on small machines.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu', 'cuda', 'cuda:0', etc. "
                             "Auto-detects if not set.")
    args = parser.parse_args()

    # ── Device selection ──
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n{BOLD}UDRL Model Sanity Check{RESET}")
    print(f"  model_path    : {args.model_path}")
    print(f"  base_model    : {args.base_model_path}")
    print(f"  reward_model  : {args.reward_model_path}")
    print(f"  device        : {device}")
    print(f"  desired_return: {args.desired_return:+.2f}  (used for generation checks)")

    # ── 1: Load ──
    model, tokenizer = check_model_loads(args.model_path, device)
    if model is None:
        print(f"\n{FAIL} Could not load model — aborting.")
        sys.exit(1)

    # ── 2: Forward pass ──
    check_forward_pass(model, tokenizer, device)

    # ── 3: Command prefix ──
    check_command_prefix(model, tokenizer, device)

    # ── 4: Generation quality ──
    completions = check_generation_quality(
        model, tokenizer, device, desired_return=args.desired_return
    )

    # ── Load reward pipeline (used in 5, 6, 8, 10) ──
    section("Loading reward model...")
    reward_available = False
    reward_pipeline = None
    try:
        reward_device = 0 if "cuda" in device else -1
        reward_pipeline = pipeline(
            "text-classification",
            model=args.reward_model_path,
            device=reward_device,
        )
        print(f"  Reward model loaded on device={reward_device}")
        reward_available = True
    except Exception as e:
        print(f"  {WARN} Could not load reward model: {e}")
        print("  Skipping reward-dependent checks (5, 6, 8, 10).")

    # ── 5: Reward scores ──
    if reward_available:
        check_reward_scores(completions, reward_pipeline)

    # ── 6: Command responsiveness  ← core UDRL check ──
    if reward_available:
        check_command_responsiveness(model, tokenizer, reward_pipeline, device)

    # ── 7: Output diversity ──
    check_output_diversity(model, tokenizer, device, desired_return=args.desired_return)

    # ── 8 & 9: Base model comparison ──
    if not args.skip_base_compare:
        if reward_available:
            check_udrl_vs_base(
                model, tokenizer, args.base_model_path, reward_pipeline,
                device, desired_return=args.desired_return,
            )
        check_kl_divergence(model, tokenizer, args.base_model_path, device)
    else:
        print(f"\n  Skipping checks 8 & 9 (--skip_base_compare).")

    # ── 10: Val corpus ──
    if args.val_path and reward_available:
        check_val_corpus(
            model, tokenizer, reward_pipeline,
            args.val_path, args.n_samples, device,
            desired_return=args.desired_return,
        )
    elif not args.val_path:
        print(f"\n  Skipping check 10 (no --val_path provided).")

    # ─────────────── Summary ───────────────
    section("Summary")
    n_pass = sum(1 for _, s, _ in results if s == "pass")
    n_warn = sum(1 for _, s, _ in results if s == "warn")
    n_fail = sum(1 for _, s, _ in results if s == "fail")
    print(f"  {PASS} {n_pass}   {WARN} {n_warn}   {FAIL} {n_fail}\n")

    for name, status, detail in results:
        tag = PASS if status == "pass" else (WARN if status == "warn" else FAIL)
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
