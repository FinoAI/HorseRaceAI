import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import json

from src.evaluation.metrics import calculate_race_array_payout_metric
from src.models.base_nn import RaceLevelStage1NN

# キャッシュロード
with open("artifacts_dirt_1win/cached_dirt_1win_data.pkl", "rb") as f:
    d = pickle.load(f)

data_va = d["data_va"]
data_te = d["data_te"]
bac_indices = d["bac_col_indices"]
col_to_idx = d["col_to_idx"]

def get_feat_idx(m):
    with open(f"artifacts_dirt_1win/features_{m}.json") as f:
        meta = json.load(f)
    s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
    return list(bac_indices) + s_idx

# Chain 99 (M34 -> M45) の完全推論
# pattern_3_chain99_step2_m45.pth
print("=== Chain 99 Step 2 (M45) Evaluation on Test (751R) and Val (751R) ===")

# Step 1: M34
idx_34 = get_feat_idx(34)
# Step 2: M45 with M34 prev_probs
# 実際に pattern_3_chain99_step2_m45.pth は input_dim = 18 * 201

# テストデータ単体に対する正確な推論メトリクスを計算
# まず CSV から全量とValの差分から hits を逆算できるか確認
# full_hits = val_hits + test_hits
# full_test_hits = r["full_test_hits"]
# val_hits = int(r["val_hit_rate"] * 751)
# test_hits = full_test_hits - val_hits

df = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
for cid in [99, 83, 84, 33, 22]:
    cdf = df[df["chain_id"] == cid]
    print(f"\n--- Detailed Breakdown for Chain {cid} ---")
    for _, r in cdf.iterrows():
        st = int(r["step"])
        m = int(r["model_idx"])
        n_feat = int(r["num_features"])
        
        v_roi = r["val_roi"] * 100
        v_hit_r = r["val_hit_rate"]
        v_hits = int(round(v_hit_r * 751))
        v_pay = r["val_roi"] * 75100
        
        full_pay = r["full_test_payout"]
        t_pay = full_pay - v_pay
        t_roi = t_pay / 75100 * 100
        
        full_hits = int(r["full_test_hits"])
        t_hits = full_hits - v_hits
        t_hit_r = t_hits / 751 * 100
        
        t_odds = t_pay / (t_hits * 100) if t_hits > 0 else 0
        
        m_str = f"M{m} (起点)" if st == 1 else f"... -> M{m}"
        print(f"Step {st:<2} | {m_str:<14} | {n_feat:<4} | {t_hit_r:5.2f}% ({t_hits:3d}R) | {t_odds:4.2f}倍 | ¥{t_pay:>7,.0f} | {t_roi:6.2f}% | {v_roi:6.2f}%")
