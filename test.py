import json
import pandas as pd
import numpy as np

dataset_root = "data/simulate/L-TOWN-A/leakage/test_v1"
manifest_path = f"{dataset_root}/manifest.jsonl"
per_event_path = "data/temp/per_event.jsonl"

# 读 manifest
mani = []
with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            mani.append(json.loads(line))
mani = pd.DataFrame(mani)
mani = mani[mani["status"].fillna("ok") == "ok"]
mani = mani[mani["kind"] == "leak"]  # 只分析 leak

# 读 per_event
ev = []
with open(per_event_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ev.append(json.loads(line))
ev = pd.DataFrame(ev)

# merge
df = ev.merge(mani[["scenario_id", "tier", "leak_diameter_m", "pipe_id"]], on="scenario_id", how="inner")

# 基础列整理
df["is_leak_pred"] = df["is_leak_pred"].astype(bool)
df["atd_m"] = pd.to_numeric(df["atd_m"], errors="coerce")

# 1) detection rate per tier（避免幸存者偏差）
det = df.groupby("tier")["is_leak_pred"].agg(["mean", "count"]).rename(columns={"mean":"det_rate"})
print("\n[Detection rate by tier]")
print(det)

# 2) Localization@Detected：只看检测到的 leak
loc = df[df["is_leak_pred"] & df["atd_m"].notna()].copy()

def summarize(g):
    x = g["atd_m"].to_numpy()
    return pd.Series({
        "n": len(x),
        "ATD_mean": float(np.mean(x)),
        "ATD_median": float(np.median(x)),
        "succ@50": float(np.mean(x <= 50)),
        "succ@100": float(np.mean(x <= 100)),
        "succ@300": float(np.mean(x <= 300)),
    })

print("\n[Localization@Detected by tier]")
print(loc.groupby("tier").apply(summarize))

# 3) 直接用 leak_diameter_m 做相关性（Spearman 更稳）
loc2 = loc[loc["leak_diameter_m"].notna()].copy()
diam = loc2["leak_diameter_m"].astype(float).to_numpy()
atd = loc2["atd_m"].to_numpy()

# Spearman（不用 scipy 也能做：对 rank 做 Pearson）
diam_rank = pd.Series(diam).rank().to_numpy()
atd_rank = pd.Series(atd).rank().to_numpy()
rho = np.corrcoef(diam_rank, atd_rank)[0,1]
print(f"\n[Spearman rho] corr(diameter, ATD) = {rho:.3f}  (negative means larger leak -> smaller ATD)")
