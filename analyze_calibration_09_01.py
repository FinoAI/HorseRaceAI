"""
analyze_calibration_09_01.py
========================================================================
アンサンブル (0.9 : 0.1) の AI予測勝率 vs 実際の勝率 キャリブレーション分析
========================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_ensemble_m29m43_m39m37")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. データ読み込み
    with open(os.path.join(CURRENT_DIR, "artifacts_dirt_1win/cached_dirt_1win_data.pkl"), "rb") as f:
        data_dict = pickle.load(f)
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]

    with open(os.path.join(CURRENT_DIR, "artifacts_cascade_step3_m29_m43/step2_m29_m43_probs.pkl"), "rb") as f:
        c29 = pickle.load(f)
    with open(os.path.join(CURRENT_DIR, "artifacts_cascade_step3_m39_m37/step2_m39_m37_probs.pkl"), "rb") as f:
        c39 = pickle.load(f)

    # 0.9 : 0.1 ブレンド
    va_p = 0.9 * c29["va_probs"] + 0.1 * c39["va_probs"]
    te_p = 0.9 * c29["te_probs"] + 0.1 * c39["te_probs"]

    datasets = [
        ("Val (前半751R)", va_p, data_va),
        ("Test (後半751R)", te_p, data_te),
        ("Total (全1502R)", np.vstack([va_p, te_p]), {
            "race_y": np.concatenate([data_va["race_y"], data_te["race_y"]]),
            "odds": np.vstack([data_va["odds"], data_te["odds"]]),
            "payouts": np.vstack([data_va["payouts"], data_te["payouts"]]),
            "race_masks": np.vstack([data_va["race_masks"], data_te["race_masks"]])
        })
    ]

    bins_top1 = [0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0]
    labels_top1 = ["<20%", "20-25%", "25-30%", "30-35%", "35-40%", "40-50%", "50%+"]

    top1_results = {}

    for name, p, d in datasets:
        top1_idx = np.argmax(p, axis=-1)
        top1_probs = np.max(p, axis=-1)
        y = d["race_y"]
        odds = d["odds"]
        payouts = d["payouts"]

        hits = (top1_idx == y)
        cats = pd.cut(top1_probs, bins=bins_top1, labels=labels_top1)
        pay_list = [payouts[i][top1_idx[i]] if hits[i] else 0 for i in range(len(hits))]
        odds_list = [odds[i][top1_idx[i]] for i in range(len(hits))]

        df_t = pd.DataFrame({
            "ai_prob": top1_probs,
            "hit": hits,
            "cat": cats,
            "payout": pay_list,
            "odds": odds_list
        })

        grp = df_t.groupby("cat", observed=False).agg(
            races=("hit", "count"),
            ai_mean_prob=("ai_prob", lambda x: float(x.mean() * 100)),
            actual_win_rate=("hit", lambda x: float(x.mean() * 100)),
            hits=("hit", "sum"),
            mean_odds=("odds", lambda x: float(x.mean())),
            payout=("payout", "sum"),
            roi=("payout", lambda x: float(x.sum() / (len(x) * 100) * 100) if len(x)>0 else 0)
        ).reset_index()

        top1_results[name] = grp.to_dict(orient="records")

    # 全出走馬（全頭レベル）キャリブレーション
    bins_all = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.0]
    labels_all = ["0-2%", "2-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-40%", "40-50%", "50%+"]
    all_horses_results = {}

    for name, p, d in datasets:
        mask = d["race_masks"].astype(bool)
        y = d["race_y"]
        n_races, n_horses = mask.shape

        valid_probs = p[mask]
        y_mat = np.zeros((n_races, n_horses), dtype=int)
        for r in range(n_races):
            y_mat[r, y[r]] = 1
        valid_y = y_mat[mask]

        cats_all = pd.cut(valid_probs, bins=bins_all, labels=labels_all)
        df_all = pd.DataFrame({
            "prob": valid_probs,
            "win": valid_y,
            "cat": cats_all
        })

        grp_all = df_all.groupby("cat", observed=False).agg(
            horses=("win", "count"),
            ai_mean_prob=("prob", lambda x: float(x.mean() * 100)),
            actual_win_rate=("win", lambda x: float(x.mean() * 100)),
            actual_wins=("win", "sum")
        ).reset_index()
        grp_all["ratio"] = grp_all["actual_win_rate"] / grp_all["ai_mean_prob"]
        all_horses_results[name] = grp_all.to_dict(orient="records")

    out_data = {
        "top1_calibration": top1_results,
        "all_horses_calibration": all_horses_results
    }

    out_json = os.path.join(artifacts_dir, "calibration_09_01.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print("キャリブレーションデータを保存しました:", out_json)

if __name__ == "__main__":
    main()
