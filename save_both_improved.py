import os
import sys
import pickle
import numpy as np
import pandas as pd
import json

df_full = pd.read_csv("artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv")
df_exh = pd.read_csv("artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv")

chain_pairs = []
for cid in df_full["chain_id"].unique():
    cdf = df_full[df_full["chain_id"] == cid].sort_values("step")
    if len(cdf) >= 2:
        s1 = cdf[cdf["step"] == 1].iloc[0]
        s2 = cdf[cdf["step"] == 2].iloc[0]
        m1 = int(s1["model_idx"])
        m2 = int(s2["model_idx"])
        v_roi_1 = s1["val_roi"]
        t_pay_1 = s1["full_test_payout"] - (v_roi_1 * 75100)
        t_roi_1 = t_pay_1 / 75100
        v_roi_2 = s2["val_roi"]
        t_pay_2 = s2["full_test_payout"] - (v_roi_2 * 75100)
        t_roi_2 = t_pay_2 / 75100
        
        v_hits_2 = int(round(s2["val_hit_rate"] * 751))
        t_hits_2 = int(s2["full_test_hits"] - v_hits_2)
        t_odds_2 = t_pay_2 / (t_hits_2 * 100) if t_hits_2 > 0 else 0

        chain_pairs.append({
            "pair": f"M{m1}->M{m2}",
            "origin_m": m1,
            "second_m": m2,
            "s1_test_roi": t_roi_1,
            "s2_test_roi": t_roi_2,
            "delta_test_roi": t_roi_2 - t_roi_1,
            "s1_val_roi": v_roi_1,
            "s2_val_roi": v_roi_2,
            "delta_val_roi": v_roi_2 - v_roi_1,
            "s2_test_payout": t_pay_2,
            "s2_test_hits": t_hits_2,
            "s2_test_hit_rate": t_hits_2 / 751,
            "s2_test_mean_odds": t_odds_2,
            "source": f"Chain_{cid}"
        })

exh_pairs = []
for _, r in df_exh.iterrows():
    m1 = int(r["origin_model"])
    m2 = int(r["second_model"])
    d_val = r["delta_val_roi"]
    d_test = r["delta_test_roi"]
    v_roi_2 = r["val_roi"]
    t_roi_2 = r["test_roi"]
    v_roi_1 = v_roi_2 - d_val
    t_roi_1 = t_roi_2 - d_test
    exh_pairs.append({
        "pair": f"M{m1}->M{m2}",
        "origin_m": m1,
        "second_m": m2,
        "s1_test_roi": t_roi_1,
        "s2_test_roi": t_roi_2,
        "delta_test_roi": d_test,
        "s1_val_roi": v_roi_1,
        "s2_val_roi": v_roi_2,
        "delta_val_roi": d_val,
        "s2_test_payout": r["test_payout"],
        "s2_test_hits": int(r["test_hits"]),
        "s2_test_hit_rate": r["test_hit_rate"],
        "s2_test_mean_odds": r["test_mean_odds"],
        "source": "Exhaustive"
    })

all_df = pd.concat([pd.DataFrame(chain_pairs), pd.DataFrame(exh_pairs)], ignore_index=True)
all_df = all_df.drop_duplicates(subset=["origin_m", "second_m"], keep="last").reset_index(drop=True)

both_up = all_df[(all_df["delta_val_roi"] > 0) & (all_df["delta_test_roi"] > 0)].copy()

print(f"Total Evaluated: {len(all_df)} | Both Improved: {len(both_up)}")
both_up.to_csv("artifacts_cascade_4origins_exhaustive/both_improved_pairs.csv", index=False)
print("Saved to artifacts_cascade_4origins_exhaustive/both_improved_pairs.csv successfully.")
