import os
import sys
import pickle
import numpy as np
import pandas as pd
import json

# 1. 100チェーン探索のデータ
df_full = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
# 2. 4起点網羅探索のデータ
df_exh = pd.read_csv("artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv")

# 100チェーン探索から Step 1 -> 2 のペアを抽出
chain_pairs = []
for cid in df_full["chain_id"].unique():
    cdf = df_full[df_full["chain_id"] == cid].sort_values("step")
    if len(cdf) >= 2:
        s1 = cdf[cdf["step"] == 1].iloc[0]
        s2 = cdf[cdf["step"] == 2].iloc[0]
        
        # Step 1
        m1 = int(s1["model_idx"])
        v_roi_1 = s1["val_roi"]
        # test_roi 逆算
        t_pay_1 = s1["full_test_payout"] - (v_roi_1 * 75100)
        t_roi_1 = t_pay_1 / 75100
        
        # Step 2
        m2 = int(s2["model_idx"])
        v_roi_2 = s2["val_roi"]
        t_pay_2 = s2["full_test_payout"] - (v_roi_2 * 75100)
        t_roi_2 = t_pay_2 / 75100
        
        chain_pairs.append({
            "pair": f"M{m1}->M{m2}",
            "origin_m": m1,
            "second_m": m2,
            "s1_val_roi": v_roi_1,
            "s1_test_roi": t_roi_1,
            "s2_val_roi": v_roi_2,
            "s2_test_roi": t_roi_2,
            "delta_val_roi": v_roi_2 - v_roi_1,
            "delta_test_roi": t_roi_2 - t_roi_1,
            "source": f"Chain_{cid}"
        })

df_cp = pd.DataFrame(chain_pairs)

# 4起点網羅探索のペア
exh_pairs = []
for _, r in df_exh.iterrows():
    m1 = int(r["origin_model"])
    m2 = int(r["second_model"])
    v_roi_2 = r["val_roi"]
    t_roi_2 = r["test_roi"]
    d_val = r["delta_val_roi"]
    d_test = r["delta_test_roi"]
    v_roi_1 = v_roi_2 - d_val
    t_roi_1 = t_roi_2 - d_test
    
    exh_pairs.append({
        "pair": f"M{m1}->M{m2}",
        "origin_m": m1,
        "second_m": m2,
        "s1_val_roi": v_roi_1,
        "s1_test_roi": t_roi_1,
        "s2_val_roi": v_roi_2,
        "s2_test_roi": t_roi_2,
        "delta_val_roi": d_val,
        "delta_test_roi": d_test,
        "source": "Exhaustive"
    })

df_ep = pd.DataFrame(exh_pairs)

# 結合
all_pairs_df = pd.concat([df_cp, df_ep], ignore_index=True)
# 重複排除（同じペアがある場合最新を優先）
all_pairs_df = all_pairs_df.drop_duplicates(subset=["origin_m", "second_m"], keep="last").reset_index(drop=True)

print(f"Total Unique Step 1 -> 2 Pairs Evaluated So Far: {len(all_pairs_df)}")

# 条件: 検証(Val)もテスト(Test)も両方上がった組み合わせ (delta_val > 0 and delta_test > 0)
both_up = all_pairs_df[(all_pairs_df["delta_val_roi"] > 0) & (all_pairs_df["delta_test_roi"] > 0)].copy()
print(f"Pairs where BOTH Val and Test IMPROVED (delta > 0): {len(both_up)} / {len(all_pairs_df)} ({len(both_up)/len(all_pairs_df)*100:.1f}%)")

# その中で 100% を超えた組み合わせ (Val >= 100% または Test >= 100%)
both_up_over_100 = both_up[(both_up["s2_val_roi"] >= 1.00) | (both_up["s2_test_roi"] >= 1.00)].copy()
print(f"Among them, pairs reaching ROI >= 100% in Val or Test: {len(both_up_over_100)}")

# Test >= 100%
test_over_100 = both_up[both_up["s2_test_roi"] >= 1.00]
print(f"Among them, pairs reaching TEST ROI >= 100%: {len(test_over_100)}")

# Val >= 100%
val_over_100 = both_up[both_up["s2_val_roi"] >= 1.00]
print(f"Among them, pairs reaching VAL ROI >= 100%: {len(val_over_100)}")

print("\n=== TOP PAIRS WHERE BOTH VAL & TEST IMPROVED & REACHED >= 100% ===")
for _, r in both_up_over_100.sort_values(by="s2_test_roi", ascending=False).iterrows():
    p = r["pair"]
    t1 = r["s1_test_roi"] * 100
    t2 = r["s2_test_roi"] * 100
    dt = r["delta_test_roi"] * 100
    v1 = r["s1_val_roi"] * 100
    v2 = r["s2_val_roi"] * 100
    dv = r["delta_val_roi"] * 100
    src = r["source"]
    print(f"{p:<10} | Test: {t1:5.1f}% -> {t2:6.2f}% ({dt:+5.2f}pt) | Val: {v1:5.1f}% -> {v2:6.2f}% ({dv:+5.2f}pt) | {src}")

print("\n=== ALL PAIRS WHERE BOTH VAL & TEST IMPROVED (Sorted by Test ROI) ===")
for _, r in both_up.sort_values(by="s2_test_roi", ascending=False).head(20).iterrows():
    p = r["pair"]
    t1 = r["s1_test_roi"] * 100
    t2 = r["s2_test_roi"] * 100
    dt = r["delta_test_roi"] * 100
    v1 = r["s1_val_roi"] * 100
    v2 = r["s2_val_roi"] * 100
    dv = r["delta_val_roi"] * 100
    print(f"{p:<10} | Test: {t1:5.1f}% -> {t2:6.2f}% ({dt:+5.2f}pt) | Val: {v1:5.1f}% -> {v2:6.2f}% ({dv:+5.2f}pt)")
