import pandas as pd
df = pd.read_csv("/workspace/podvodka/sft_vs_dpo.csv")

# --- existing Layer 3 queries ---
df["base__gen_len"] = df["base__gen"].str.split().str.len()
df["dpo__gen_len"]  = df["dpo__gen"].str.split().str.len()
print("Mean gen length:")
print(df[["base__gen_len", "dpo__gen_len"]].mean())

suspicious = (
    df[df["dpo__rm_score"] > 0.9]
    .sort_values("dpo__rm_score", ascending=False)
)
print("\nTop-scoring DPO generations (most suspicious for hacking):")
print(suspicious[["prompt", "base__gen", "dpo__gen", "dpo__rm_score"]].to_string())

df["gap"] = df["dpo__rm_score"] - df["base__rm_score"]
biggest_gaps = df.sort_values("gap", ascending=False).head(15)
print("\nBiggest DPO vs SFT score gaps:")
print(biggest_gaps[["prompt", "base__gen", "dpo__gen", "gap"]].to_string())

regressions = df[df["gap"] < -0.2].sort_values("gap")
print(f"\nRegressions (DPO worse by >0.2): {len(regressions)} rows")
print(regressions[["prompt", "base__gen", "dpo__gen", "gap"]].to_string())

# --- new: hard prompts check ---
hard = df[df["base__rm_score"] < -0.4].sort_values("base__rm_score")
print(f"\nHard prompts ({len(hard)} rows):")
print(hard[["prompt", "base__gen", "dpo__gen",
            "base__rm_score", "dpo__rm_score"]].to_string())