import pandas as pd
import numpy as np

# 1. artifacts_cascade_full1502 (100チェーン探索)
df_full = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
# 2. artifacts_cascade_unstarted (直前の4モデル起点 17チェーン探索)
df_unstarted = pd.read_csv("artifacts_cascade_unstarted/unstarted_origins_cascade_all_steps.csv")

def extract_step1_2_pairs(df):
    pairs = set()
    for cid in df["chain_id"].unique():
        cdf = df[df["chain_id"] == cid]
        if len(cdf) >= 2:
            s1 = cdf[cdf["step"] == 1]
            s2 = cdf[cdf["step"] == 2]
            if len(s1) > 0 and len(s2) > 0:
                m1 = int(s1["model_idx"].iloc[0])
                m2 = int(s2["model_idx"].iloc[0])
                pairs.add((m1, m2))
    return pairs

def extract_all_transition_pairs(df):
    pairs = set()
    for cid in df["chain_id"].unique():
        cdf = df[df["chain_id"] == cid].sort_values("step")
        models = cdf["model_idx"].tolist()
        for i in range(len(models) - 1):
            pairs.add((int(models[i]), int(models[i+1])))
    return pairs

pairs_step1_2_full = extract_step1_2_pairs(df_full)
pairs_step1_2_unstarted = extract_step1_2_pairs(df_unstarted)
all_step1_2_pairs = pairs_step1_2_full | pairs_step1_2_unstarted

all_trans_pairs = extract_all_transition_pairs(df_full) | extract_all_transition_pairs(df_unstarted)

TOTAL_POSSIBLE = 50 * 49 # 2,450 通り

print("==================================================================")
print("1. 全体 (50モデル全体) の Step 1 -> Step 2 ペア集計")
print("==================================================================")
print(f"理論上の全ペア数 (50 × 49): {TOTAL_POSSIBLE:,d} 通り")
print(f"前回の100チェーンで試したペア: {len(pairs_step1_2_full)} 通り")
print(f"今回の17チェーンで試したペア: {len(pairs_step1_2_unstarted)} 通り")
print(f"累計で試行済みのユニークペア: {len(all_step1_2_pairs)} 通り ({len(all_step1_2_pairs)/TOTAL_POSSIBLE*100:.2f}%)")
print(f"【残りの未試行ペア数】: {TOTAL_POSSIBLE - len(all_step1_2_pairs):,d} 通り ({100 - len(all_step1_2_pairs)/TOTAL_POSSIBLE*100:.2f}%)")

print("\n==================================================================")
print("2. 今回の4起点 (Model 45, 2, 8, 47) に限定したペア集計")
print("==================================================================")
targets = [45, 2, 8, 47]
target_total_possible = len(targets) * 49 # 4 * 49 = 196 通り

target_tried_pairs = {p for p in all_step1_2_pairs if p[0] in targets}
print(f"4モデル起点の場合の全ペア数 (4 × 49): {target_total_possible} 通り")
print(f"これまでに試されたペア数: {len(target_tried_pairs)} 通り ({len(target_tried_pairs)/target_total_possible*100:.2f}%)")
print(f"【4モデル起点の残り未試行ペア数】: {target_total_possible - len(target_tried_pairs)} 通り ({100 - len(target_tried_pairs)/target_total_possible*100:.2f}%)")

print("\n各起点ごとの内訳:")
for t in targets:
    t_pairs = {p for p in all_step1_2_pairs if p[0] == t}
    tried_m2 = sorted([p[1] for p in t_pairs])
    remaining = 49 - len(t_pairs)
    print(f"  Model {t:2d} 起点: 試行済み {len(t_pairs):2d} / 49 通り (残り {remaining:2d} 通り)")
    print(f"    試した相手: {tried_m2}")

print("\n==================================================================")
print("3. 【参考】チェーン内の全遷移 (Step 2->3, 3->4 等も含む全隣接ペア)")
print("==================================================================")
print(f"全ステップ間で試されたユニーク遷移ペア: {len(all_trans_pairs)} / 2,450 通り ({len(all_trans_pairs)/TOTAL_POSSIBLE*100:.2f}%)")
print(f"一度も隣接して繋がれていないペア: {TOTAL_POSSIBLE - len(all_trans_pairs):,d} 通り ({100 - len(all_trans_pairs)/TOTAL_POSSIBLE*100:.2f}%)")
