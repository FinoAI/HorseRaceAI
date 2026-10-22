import pandas as pd
import numpy as np
import json

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")

# 1. 今までスタートしていないモデル (Step 1 として選ばれていないモデル)
all_models = set(range(1, 51))
step1_df = df[df["step"] == 1]
started_models = set(step1_df["model_idx"].unique())
not_started_models = sorted(list(all_models - started_models))

step1_counts = step1_df["model_idx"].value_counts().sort_index()

print("==================================================================")
print("1. スタートモデル (Step 1) の集計")
print("==================================================================")
print(f"全モデル数: 50モデル (Model 1 〜 50)")
print(f"一度でもStep 1 (スタート) に選ばれたモデル: {len(started_models)} / 50 モデル")
print(f"一度もStep 1 (スタート) に選ばれていないモデル: {len(not_started_models)} / 50 モデル")
print(f"未スタートのモデル番号: {not_started_models}")

# 2. Step 1 -> Step 2 の遷移状況 (チェーン終了ステップの分布)
chain_max_steps = df.groupby("chain_id")["step"].max()
stop_at_step2 = (chain_max_steps == 2).sum()
progress_to_step3_or_more = (chain_max_steps >= 3).sum()

print("\n==================================================================")
print("2. Step 1 -> Step 2 の推移・終了状況 (100チェーンの内訳)")
print("==================================================================")
print(f"全試行チェーン数: {len(chain_max_steps)} チェーン")
print(f"Step 2 で回収率が下がり即終了したチェーン (Step 2 止まり): {stop_at_step2} / 100 ({stop_at_step2}%)")
print(f"Step 2 で回収率が上がり Step 3 以降へ進展できたチェーン: {progress_to_step3_or_more} / 100 ({progress_to_step3_or_more}%)")

# 3. Step 1 -> Step 2 のペア（組み合わせ）の網羅状況
# 50モデルから2つ選ぶ順列は 50 * 49 = 2,450 通り
step1_2_pairs = set()
for cid in df["chain_id"].unique():
    cdf = df[df["chain_id"] == cid]
    if len(cdf) >= 2:
        m1 = int(cdf[cdf["step"] == 1]["model_idx"].iloc[0])
        m2 = int(cdf[cdf["step"] == 2]["model_idx"].iloc[0])
        step1_2_pairs.add((m1, m2))

total_possible_pairs = 50 * 49
tried_pairs_count = len(step1_2_pairs)
untried_pairs_count = total_possible_pairs - tried_pairs_count

print("\n==================================================================")
print("3. Step 1 -> Step 2 のモデルペア (順列) の網羅率")
print("==================================================================")
print(f"理論上の全ペア数 (50 × 49): {total_possible_pairs:,d} 通り")
print(f"今回の100チェーンで試されたペア数: {tried_pairs_count} 通り ({tried_pairs_count / total_possible_pairs * 100:.2f}%)")
print(f"まだ試されていない未検証ペア数: {untried_pairs_count:,d} 通り ({untried_pairs_count / total_possible_pairs * 100:.2f}%)")

# 4. 全体の登場モデル (Step 1〜6 全ステップを通して一度も登場しなかったモデルはあるか？)
all_appeared_models = set(df["model_idx"].unique())
never_appeared_models = sorted(list(all_models - all_appeared_models))
print("\n==================================================================")
print("4. 全ステップ通して一度も登場しなかったモデル")
print("==================================================================")
print(f"全276ステップ中で一度でも登場したモデル: {len(all_appeared_models)} / 50 モデル")
print(f"一度も学習・実行されなかったモデル: {len(never_appeared_models)} モデル {never_appeared_models}")
