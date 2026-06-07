"""
Compare DPO checkpoint(s) against the SFT base on the same prompts.

For each checkpoint:
- Generate K completions per prompt
- Score every generation with the reward model
- Report mean/std reward + a side-by-side CSV of generations
- Flag DPO-specific failure modes: length inflation, format gaming, ceiling saturation

DPO-specific failure modes this script watches for:
  - Length inflation: DPO learns longer == preferred if training data has length bias
  - Format gaming: policy mimics surface structure (bullets, hedges) without better content
  - Ceiling saturation: >30% of generations scoring >+0.9 suggests RM exploit
  - Across-prompt variance collapse: policy stopped differentiating between prompts

Use this script's CSV output as Layer 2 of the 3-layer sanity check.
Always read the CSV afterwards — numbers alone won't catch RM gaming.

Example
-------
python compare_checkpoints_dpo.py \\
    --checkpoints \\
        base:tsaxena/gpt2-large-prompt-tags \\
        dpo:/workspace/podvodka/models/gpt2-large-dpo/final \\
    --val_path /workspace/podvodka/data/val_strings.csv \\
    --num_prompts 30 \\
    --samples_per_prompt 3 \\
    --output_csv /workspace/podvodka/sft_vs_dpo.csv

With a chat-template model:
python compare_checkpoints_dpo.py \\
    --checkpoints base:... dpo:... \\
    --val_path ... \\
    --use_chat_template \\
    --prompt_column prompt
"""

import argparse
import gc
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_checkpoints(items: List[str]) -> Dict[str, str]:
    """Parse 'name:path' pairs from CLI into an ordered dict."""
    out = {}
    for item in items:
        if ":" not in item:
            raise SystemExit(f"--checkpoints entry must be 'name:path', got {item!r}")
        name, path = item.split(":", 1)
        out[name.strip()] = path.strip()
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_for_checkpoint(
    path: str,
    prompts: List[str],
    samples_per_prompt: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    use_chat_template: bool,
) -> List[List[str]]:
    """Generate K samples per prompt. Returns list-of-lists shape (num_prompts, K)."""
    print(f"  [load] {path}")
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation with causal LMs

    model = (
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    all_samples: List[List[str]] = []
    for i, prompt in enumerate(prompts):
        if use_chat_template and tok.chat_template is not None:
            formatted = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        ids = tok(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).input_ids.to(device)

        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            temperature=1.0,
            num_return_sequences=samples_per_prompt,
            pad_token_id=tok.pad_token_id,
        )

        # Decode only the newly generated tokens (strip prompt prefix)
        decoded = [
            tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
            for o in out
        ]
        all_samples.append(decoded)

        if (i + 1) % 10 == 0:
            print(f"    generated {i+1}/{len(prompts)}")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

    return all_samples


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_all(
    reward_pipeline,
    prompts: List[str],
    samples_by_ckpt: Dict[str, List[List[str]]],
    batch_size: int,
    rm_separator: str,
) -> Dict[str, np.ndarray]:
    """
    Score every (prompt, sample) pair under every checkpoint.
    Returns dict: checkpoint name -> array of shape (num_prompts, K).
    """
    scores_by_ckpt = {}
    for name, samples in samples_by_ckpt.items():
        print(f"  [score] {name}")
        flat_texts = [
            p + rm_separator + s
            for p, sample_list in zip(prompts, samples)
            for s in sample_list
        ]
        outs = reward_pipeline(
            flat_texts,
            function_to_apply="none",
            batch_size=batch_size,
            truncation=True,
        )
        flat_scores = np.array([o["score"] for o in outs], dtype=np.float32)
        n_prompts = len(samples)
        k = len(samples[0])
        scores_by_ckpt[name] = flat_scores.reshape(n_prompts, k)
    return scores_by_ckpt


# ---------------------------------------------------------------------------
# Summary + flags
# ---------------------------------------------------------------------------

def print_summary(scores_by_ckpt: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, scores in scores_by_ckpt.items():
        per_prompt_mean = scores.mean(axis=1)
        rows.append({
            "checkpoint":             name,
            "rm_mean":                float(scores.mean()),
            "rm_std_overall":         float(scores.std()),
            "rm_std_across_prompts":  float(per_prompt_mean.std()),  # key hacking signal
            "rm_min":                 float(scores.min()),
            "rm_max":                 float(scores.max()),
            "frac_above_+0.5":        float((scores > 0.5).mean()),
            "frac_at_ceiling_+0.9":   float((scores > 0.9).mean()),
        })
    df = pd.DataFrame(rows).set_index("checkpoint")
    print("\n" + "=" * 80)
    print("REWARD MODEL SCORES BY CHECKPOINT")
    print("=" * 80)
    print(df.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("=" * 80)
    return df


def detect_hacking_flags(
    summary_df: pd.DataFrame,
    gen_df: Optional[pd.DataFrame],
    base_name: str,
    dpo_name: str,
) -> None:
    """
    Heuristic warnings for DPO-specific failure modes.
    Not definitive — always read the CSV to confirm.
    """
    print("\nHEURISTIC FLAGS (not definitive — read the generations!):")
    any_flag = False

    s = summary_df

    # 1. Across-prompt variance collapse — policy stopped differentiating prompts
    if base_name in s.index and dpo_name in s.index:
        base_var = s.loc[base_name, "rm_std_across_prompts"]
        dpo_var  = s.loc[dpo_name, "rm_std_across_prompts"]
        if dpo_var < 0.6 * base_var:
            print(f"  ⚠  Across-prompt variance collapsed: "
                  f"{base_name}={base_var:.3f} → {dpo_name}={dpo_var:.3f} "
                  f"(>{(1-dpo_var/base_var)*100:.0f}% drop). "
                  f"Policy may have learned a universal surface trick.")
            any_flag = True

    # 2. Overall std collapse — mode collapse signature
    if base_name in s.index and dpo_name in s.index:
        base_std = s.loc[base_name, "rm_std_overall"]
        dpo_std  = s.loc[dpo_name, "rm_std_overall"]
        if dpo_std < 0.5 * base_std:
            print(f"  ⚠  Overall std collapsed >50%: "
                  f"{base_std:.3f} → {dpo_std:.3f}. Possible mode collapse.")
            any_flag = True

    # 3. Ceiling saturation — RM exploit signal
    if dpo_name in s.index:
        ceiling = s.loc[dpo_name, "frac_at_ceiling_+0.9"]
        base_ceiling = s.loc[base_name, "frac_at_ceiling_+0.9"] if base_name in s.index else 0.0
        if ceiling > 0.3 and base_ceiling < 0.05:
            print(f"  ⚠  {ceiling:.0%} of {dpo_name} generations score >+0.9 "
                  f"(vs {base_ceiling:.0%} for {base_name}). Possible RM saturation exploit.")
            any_flag = True

    # 4. DPO-specific: length inflation
    if gen_df is not None:
        base_col = f"{base_name}__gen"
        dpo_col  = f"{dpo_name}__gen"
        if base_col in gen_df.columns and dpo_col in gen_df.columns:
            base_len = gen_df[base_col].str.split().str.len().mean()
            dpo_len  = gen_df[dpo_col].str.split().str.len().mean()
            ratio = dpo_len / base_len if base_len > 0 else 1.0
            flag = "⚠ " if ratio > 1.4 else "✅"
            print(f"  {flag} Length ratio {dpo_name}/{base_name}: "
                  f"{dpo_len:.1f} vs {base_len:.1f} tokens/gen (ratio={ratio:.2f})"
                  + (" — possible length inflation bias" if ratio > 1.4 else ""))
            if ratio > 1.4:
                any_flag = True

    if not any_flag:
        print("  (no strong flags detected)")

    print()
    print("Layer 3 — READ THE GENERATIONS:")
    print("  See the Layer3 query section printed below.")


def print_layer3_queries(dpo_name: str, base_name: str, output_csv: str) -> None:
    """Print copy-pasteable pandas queries for Layer 3 manual review."""
    print("\n" + "=" * 80)
    print("LAYER 3: MANUAL GENERATION REVIEW QUERIES")
    print("=" * 80)
    print(f"""
import pandas as pd
df = pd.read_csv("{output_csv}")

# Add length columns for length-inflation check
df["{base_name}__gen_len"] = df["{base_name}__gen"].str.split().str.len()
df["{dpo_name}__gen_len"]  = df["{dpo_name}__gen"].str.split().str.len()

print("Mean gen length:")
print(df[["{base_name}__gen_len", "{dpo_name}__gen_len"]].mean())

# Top-scoring DPO generations — most likely to expose RM exploits
suspicious = (
    df[df["{dpo_name}__rm_score"] > 0.9]
    .sort_values("{dpo_name}__rm_score", ascending=False)
)
print("\\nTop-scoring DPO generations (most suspicious for hacking):")
print(suspicious[["prompt", "{base_name}__gen", "{dpo_name}__gen",
                   "{dpo_name}__rm_score"]].to_string())

# Biggest score gaps — where DPO diverged most from SFT
df["gap"] = df["{dpo_name}__rm_score"] - df["{base_name}__rm_score"]
biggest_gaps = df.sort_values("gap", ascending=False).head(15)
print("\\nBiggest DPO vs SFT score gaps:")
print(biggest_gaps[["prompt", "{base_name}__gen", "{dpo_name}__gen", "gap"]].to_string())

# Cases where DPO scored LOWER — regression check
regressions = df[df["gap"] < -0.2].sort_values("gap")
print(f"\\nRegressions (DPO worse by >0.2): {{len(regressions)}} rows")
print(regressions[["prompt", "{base_name}__gen", "{dpo_name}__gen", "gap"]].to_string())
""")
    print("WHAT TO LOOK FOR:")
    print("  - Length inflation: DPO gens uniformly longer regardless of prompt?")
    print("  - Hedge insertion: 'Certainly!', 'Great question', disclaimers added?")
    print("  - Format gaming: bullet points / headers on every response?")
    print("  - Prompt echo: DPO restates the question more than SFT?")
    print("  - Topic drift: DPO ignores prompt content?")
    print("  - Coherence: does DPO actually READ better than SFT?")
    print()
    print("DECISION CRITERION:")
    print("  Pick the policy where generations READ best, not the one with")
    print("  the highest RM score. If DPO reads worse despite scoring higher,")
    print("  that's your proof of reward hacking — SFT is your better policy.")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Compare DPO checkpoint(s) vs SFT base on the same held-out prompts."
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="'name:path' pairs. First entry treated as base/SFT, "
                        "second as DPO. E.g. base:sft_path dpo:dpo_path")
    p.add_argument("--val_path", required=True,
                   help="CSV with prompts. Use --prompt_column to specify the column name.")
    p.add_argument("--prompt_column", default="text",
                   help="Column in val CSV containing raw prompt text (default: 'text').")
    p.add_argument("--prompt_separator", default="</s>",
                   help="Token used to strip context from prompts (default: '</s>'). "
                        "Set to '' to use prompts as-is.")
    p.add_argument("--reward_model_path", default="toloka/prompts_reward_model")
    p.add_argument("--rm_separator", default="</s>",
                   help="Separator inserted between prompt and generation when scoring "
                        "(default: '</s>'). Match what your RM was trained on.")
    p.add_argument("--num_prompts", type=int, default=30)
    p.add_argument("--samples_per_prompt", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--reward_batch_size", type=int, default=32)
    p.add_argument("--use_chat_template", action="store_true",
                   help="Apply the model's chat template before generation. "
                        "Use for instruction-tuned DPO models.")
    p.add_argument("--output_csv", default="sft_vs_dpo.csv")
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    args = p.parse_args()

    assert torch.cuda.is_available(), "Need a GPU."
    device = "cuda:0"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoints = parse_checkpoints(args.checkpoints)
    ckpt_names = list(checkpoints.keys())
    base_name = ckpt_names[0]
    dpo_name  = ckpt_names[-1]
    print(f"Comparing {len(checkpoints)} checkpoints: {ckpt_names}")
    print(f"  Base (SFT): {base_name}")
    print(f"  DPO:        {dpo_name}")
    if args.use_chat_template:
        print("  Chat template: ENABLED")

    # Load prompts
    val_df = pd.read_csv(args.val_path)
    raw = val_df[args.prompt_column].tolist()
    if args.prompt_separator:
        prompts = [r.split(args.prompt_separator)[0] + args.prompt_separator
                   for r in raw][: args.num_prompts]
    else:
        prompts = raw[: args.num_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.val_path} (column: '{args.prompt_column}')")

    # Phase 1: generation
    print("\n--- Phase 1: generation ---")
    samples_by_ckpt: Dict[str, List[List[str]]] = {}
    for name, path in checkpoints.items():
        print(f"[ckpt] {name}")
        torch.manual_seed(args.seed)
        samples_by_ckpt[name] = generate_for_checkpoint(
            path=path,
            prompts=prompts,
            samples_per_prompt=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=dtype,
            use_chat_template=args.use_chat_template,
        )

    # Phase 2: scoring
    print("\n--- Phase 2: reward scoring ---")
    reward_pipe = pipeline(
        "text-classification",
        model=args.reward_model_path,
        device=0,
        torch_dtype=dtype,
    )
    scores_by_ckpt = score_all(
        reward_pipe,
        prompts,
        samples_by_ckpt,
        batch_size=args.reward_batch_size,
        rm_separator=args.rm_separator,
    )

    # Phase 3: summary
    print("\n--- Phase 3: summary ---")
    summary_df = print_summary(scores_by_ckpt)
    if args.summary_csv:
        summary_df.to_csv(args.summary_csv)
        print(f"Wrote summary to {args.summary_csv}")

    # Phase 4: side-by-side CSV (build before flag detection so we can pass it in)
    print("\n--- Phase 4: writing side-by-side CSV ---")
    rows = []
    for i, prompt in enumerate(prompts):
        for k in range(args.samples_per_prompt):
            row = {"prompt": prompt, "sample_idx": k}
            for name in checkpoints:
                row[f"{name}__gen"]      = samples_by_ckpt[name][i][k]
                row[f"{name}__rm_score"] = float(scores_by_ckpt[name][i, k])
            rows.append(row)
    gen_df = pd.DataFrame(rows)
    gen_df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(gen_df)} rows to {args.output_csv}")

    # Phase 5: heuristic flags + Layer 3 queries
    print("\n--- Phase 5: heuristic flags ---")
    detect_hacking_flags(summary_df, gen_df, base_name, dpo_name)
    print_layer3_queries(dpo_name, base_name, args.output_csv)


if __name__ == "__main__":
    main()