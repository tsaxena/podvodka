import pandas as pd
df = pd.read_csv("/workspace/podvodka/sft_vs_ppo.csv")
suspicious = df[df["ppo__rm_score"] > 0.9].sort_values("ppo__rm_score", ascending=False)
sdf = suspicious[["prompt", "base__gen", "ppo__gen", "ppo__rm_score"]]
sdf.to_csv('suspicious.csv')