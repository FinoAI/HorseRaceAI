import os
import sys
import pandas as pd
import numpy as np

# すべてのCSVから Step 1 -> Step 2 のユニークペアを収集
all_evaluated_pairs = set()

# 1. 100チェーン探索 (random_cascade_full1502)
p1 = "artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv"
if os.path.exists(p1):
    df1 = pd.read_csv(p1)
    for cid in df1["chain_id"].unique():
        cdf = df1[df1["chain_id"] == cid]
        s1 = cdf[cdf["step"] == 1]
        s2 = cdf[cdf["step"] == 2]
        if len(s1) > 0 and len(s2) > 0:
            m1 = int(s1["model_idx"].iloc[0])
            m2 = int(s2["model_idx"].iloc[0])
            all_evaluated_pairs.add((m1, m2))
print(f"After {p1}: {len(all_evaluated_pairs)} unique pairs")

# 2. 4起点網羅探索 (exhaustive_196_pairs_summary)
p2 = "artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv"
if os.path.exists(p2):
    df2 = pd.read_csv(p2)
    for _, r in df2.iterrows():
        m1 = int(r["origin_model"])
        m2 = int(r["second_model"])
        all_evaluated_pairs.add((m1, m2))
print(f"After {p2}: {len(all_evaluated_pairs)} unique pairs")

# 3. 新規ペア探索 (search_new_pairs_all)
p3 = "artifacts_cascade_search_new_pairs/search_new_pairs_all.csv"
if os.path.exists(p3):
    df3 = pd.read_csv(p3)
    for _, r in df3.iterrows():
        m1 = int(r["origin_model"])
        m2 = int(r["second_model"])
        all_evaluated_pairs.add((m1, m2))
print(f"After {p3}: {len(all_evaluated_pairs)} unique pairs")

# 4. 未スタート起点探索 (unstarted_origins_cascade_all_steps)
p4 = "artifacts_cascade_unstarted/unstarted_origins_cascade_all_steps.csv"
if os.path.exists(p4):
    df4 = pd.read_csv(p4)
    for cid in df4["chain_id"].unique():
        cdf = df4[df4["chain_id"] == cid]
        s1 = cdf[cdf["step"] == 1]
        s2 = cdf[cdf["step"] == 2]
        if len(s1) > 0 and len(s2) > 0:
            m1 = int(s1["model_idx"].iloc[0])
            m2 = int(s2["model_idx"].iloc[0])
            all_evaluated_pairs.add((m1, m2))
print(f"After {p4}: {len(all_evaluated_pairs)} unique pairs")

# 5. 指定チェーン (specific_chain)
p5 = "artifacts_specific_chain/specific_chain_summary.csv"
if os.path.exists(p5):
    # M17 -> M50
    all_evaluated_pairs.add((17, 50))

TOTAL_PAIRS = 50 * 49 # 2,450 通り
evaluated_count = len(all_evaluated_pairs)
remaining_count = TOTAL_PAIRS - evaluated_count
eval_pct = evaluated_count / TOTAL_PAIRS * 100
rem_pct = remaining_count / TOTAL_PAIRS * 100

print("\n=======================================================")
print("  STEP 1 -> STEP 2 全ペア評価進捗サマリー")
print("=======================================================")
print(f"理論上の全ペア数 (50モデル × 49相手): {TOTAL_PAIRS:,d} 通り (100.0%)")
print(f"これまでに評価完了したペア数:         {evaluated_count:,d} 通り ({eval_pct:.2f}%)")
print(f"【まだ評価できていない残りのペア数】: {remaining_count:,d} 通り ({rem_pct:.2f}%)")

# 各起点モデルごとの進捗内訳
origin_counts = {}
for m1 in range(1, 51):
    m1_pairs = [p for p in all_evaluated_pairs if p[0] == m1]
    origin_counts[m1] = len(m1_pairs)

df_origins = pd.DataFrame([
    {"origin_model": m, "evaluated": origin_counts[m], "remaining": 49 - origin_counts[m], "rate": origin_counts[m] / 49 * 100}
    for m in range(1, 51)
])

print("\n--- 起点モデル別の進捗状況 (上位・完走モデル) ---")
completed = df_origins[df_origins["remaining"] == 0]
print(f"全49相手を完全網羅 (100%完了) した起点モデル数: {len(completed)} モデル")
print(f"完全網羅モデル番号: {completed['origin_model'].tolist()}")

print("\n--- 起点モデル別の進捗詳細 (上位15モデル) ---")
for _, r in df_origins.sort_values(by="evaluated", ascending=False).head(15).iterrows():
    m = int(r["origin_model"])
    ev = int(r["evaluated"])
    rem = int(r["remaining"])
    rate = r["rate"]
    print(f"Model {m:2d} 起点: 評価済 {ev:2d} / 49 通り (残り {rem:2d} 通り, 進捗 {rate:5.1f}%)")

zero_evaluated = df_origins[df_origins["evaluated"] == 0]
print(f"\n一度も起点として試されていないモデル数: {len(zero_evaluated)} モデル")
if len(zero_evaluated) > 0:
    print(f"未着手モデル番号: {zero_evaluated['origin_model'].tolist()}")
