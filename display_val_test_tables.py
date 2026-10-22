import os
import sys
import pickle
import numpy as np
import pandas as pd
import json

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")

# 751レースの投資額
test_inv = 751 * 100
val_inv = 751 * 100

val_payout = df["val_roi"] * val_inv
test_payout = df["full_test_payout"] - val_payout
test_roi = test_payout / test_inv

df["val_payout"] = val_payout
df["test_payout"] = test_payout
df["test_roi"] = test_roi

chains_to_show = [99, 84, 33, 83, 22]

for cid in chains_to_show:
    cdf = df[df["chain_id"] == cid]
    print(f"\n=======================================================")
    print(f"  Chain {cid} 成績推移 (テスト751R vs 検証751R)")
    print(f"=======================================================")
    print(f"{'Step':<6} | {'接続モデル':<14} | {'特徴量数':<6} | {'払戻金額(Test)':<12} | {'テストROI':<10} | {'検証ROI(Val)':<10}")
    print("-" * 75)
    for _, r in cdf.iterrows():
        st = int(r["step"])
        m = int(r["model_idx"])
        n_feat = int(r["num_features"])
        t_pay = r["test_payout"]
        t_roi = r["test_roi"] * 100
        v_roi = r["val_roi"] * 100
        m_str = f"M{m} (起点)" if st == 1 else f"... -> M{m}"
        print(f"Step {st:<2} | {m_str:<14} | {n_feat:<8} | ¥{t_pay:>9,.0f} | {t_roi:>8.2f}% | {v_roi:>8.2f}%")
