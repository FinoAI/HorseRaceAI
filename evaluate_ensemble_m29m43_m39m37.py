"""
evaluate_ensemble_m29m43_m39m37.py
========================================================================
M29->M43 と M39->M37 のアンサンブル評価スクリプト
------------------------------------------------------------------------
- M29->M43: 高的中・高回収型（Val 112.5%, Test 117.3%, 的中率 ~22%）
- M39->M37: 超高配当・爆発型（Val 110.3%, Test 127.0%, 的中率 ~14%）
- 重み比率: w_m29_m43 を 0.0 から 1.0 まで 0.1 刻みで全11通り検証
- 指標: Val/Test 回収率, 的中率, 払戻金, 平均オッズ, 収支, 一致時/不一致時分析
========================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.evaluation.metrics import calculate_race_array_payout_metric

def main():
    output_dir = os.path.join(CURRENT_DIR, "artifacts_ensemble_m29m43_m39m37")
    os.makedirs(output_dir, exist_ok=True)

    # 1. データ読み込み
    data_path = os.path.join(CURRENT_DIR, "artifacts_dirt_1win/cached_dirt_1win_data.pkl")
    with open(data_path, "rb") as f:
        data_dict = pickle.load(f)
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]

    # 2. 予測確率読み込み
    p29_path = os.path.join(CURRENT_DIR, "artifacts_cascade_step3_m29_m43/step2_m29_m43_probs.pkl")
    p39_path = os.path.join(CURRENT_DIR, "artifacts_cascade_step3_m39_m37/step2_m39_m37_probs.pkl")
    with open(p29_path, "rb") as f:
        c29 = pickle.load(f)
    with open(p39_path, "rb") as f:
        c39 = pickle.load(f)

    va_p29 = c29["va_probs"]
    te_p29 = c29["te_probs"]

    va_p39 = c39["va_probs"]
    te_p39 = c39["te_probs"]

    num_va_races = len(va_p29)
    num_te_races = len(te_p29)
    va_cost = num_va_races * 100
    te_cost = num_te_races * 100
    tot_cost = va_cost + te_cost

    results = []

    weights = [round(w * 0.1, 1) for w in range(11)]  # 0.0 to 1.0

    print("=" * 80)
    print("M29->M43 × M39->M37 アンサンブル評価 (0.0 〜 1.0)")
    print("=" * 80)

    for w in weights:
        w29 = w
        w39 = round(1.0 - w, 1)

        va_blend = w29 * va_p29 + w39 * va_p39
        te_blend = w29 * te_p29 + w39 * te_p39

        va_res = calculate_race_array_payout_metric(
            va_blend, data_va["race_masks"], data_va["race_y"], data_va["odds"], data_va["payouts"], "prob"
        )
        te_res = calculate_race_array_payout_metric(
            te_blend, data_te["race_masks"], data_te["race_y"], data_te["odds"], data_te["payouts"], "prob"
        )

        va_roi = va_res["roi"]
        te_roi = te_res["roi"]
        tot_roi = (va_res["total_payout"] + te_res["total_payout"]) / tot_cost

        va_hit_rate = va_res["hit_rate"]
        te_hit_rate = te_res["hit_rate"]
        tot_hit_rate = (va_res["hits"] + te_res["hits"]) / (num_va_races + num_te_races)

        va_odds = va_res["total_payout"] / (va_res["hits"] * 100) if va_res["hits"] > 0 else 0
        te_odds = te_res["total_payout"] / (te_res["hits"] * 100) if te_res["hits"] > 0 else 0

        tot_payout = va_res["total_payout"] + te_res["total_payout"]
        tot_profit = tot_payout - tot_cost

        row = {
            "weight_m29_m43": w29,
            "weight_m39_m37": w39,
            "val_roi": va_roi,
            "val_hit_rate": va_hit_rate,
            "val_hits": va_res["hits"],
            "val_payout": va_res["total_payout"],
            "val_mean_odds": va_odds,
            "test_roi": te_roi,
            "test_hit_rate": te_hit_rate,
            "test_hits": te_res["hits"],
            "test_payout": te_res["total_payout"],
            "test_mean_odds": te_odds,
            "total_roi": tot_roi,
            "total_hit_rate": tot_hit_rate,
            "total_hits": va_res["hits"] + te_res["hits"],
            "total_payout": tot_payout,
            "total_profit": tot_profit
        }
        results.append(row)

        print(f"Weight [M29->M43: {w29:.1f} | M39->M37: {w39:.1f}] => "
              f"Val ROI: {va_roi*100:6.2f}% (Hit {va_hit_rate*100:4.1f}%, {va_res['hits']}勝) | "
              f"Test ROI: {te_roi*100:6.2f}% (Hit {te_hit_rate*100:4.1f}%, {te_res['hits']}勝) | "
              f"総合 ROI: {tot_roi*100:6.2f}% | 収支: ¥{int(tot_profit):+,d}")

    # DataFrame保存
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "ensemble_m29m43_m39m37_weights_0_to_1.csv")
    df.to_csv(csv_path, index=False)
    json_path = os.path.join(output_dir, "ensemble_m29m43_m39m37_weights_0_to_1.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 一致時・不一致時（Agreement vs Disagreement）分析
    print("\n" + "=" * 80)
    print("本命馬 一致・不一致 時のパフォーマンス分析 (Testデータ 751レース)")
    print("=" * 80)

    # Testデータでの各馬の選択
    p_te29_flat = te_p29
    p_te39_flat = te_p39
    pred_te29 = np.argmax(p_te29_flat, axis=-1)
    pred_te39 = np.argmax(p_te39_flat, axis=-1)

    agree_mask = (pred_te29 == pred_te39)
    disagree_mask = ~agree_mask

    y_te = data_te["race_y"]
    odds_te = data_te["odds"]
    payouts_te = data_te["payouts"]

    # 1. 一致したレース (Agreement)
    agree_indices = np.where(agree_mask)[0]
    agree_hits = 0
    agree_payout = 0
    for idx in agree_indices:
        horse = pred_te29[idx]
        if horse == y_te[idx]:
            agree_hits += 1
            agree_payout += payouts_te[idx][horse]
    agree_count = len(agree_indices)
    agree_roi = agree_payout / (agree_count * 100) if agree_count > 0 else 0
    agree_hit_rate = agree_hits / agree_count if agree_count > 0 else 0

    print(f"【両モデル本命一致レース】: {agree_count} レース ({agree_count/num_te_races*100:.1f}%)")
    print(f"  - 的中数: {agree_hits} 勝 / 的中率: {agree_hit_rate*100:.2f}%")
    print(f"  - 払戻金: ¥{int(agree_payout):,d} / 投資: ¥{int(agree_count*100):,d} / 回収率: {agree_roi*100:.2f}%")
    print(f"  - 収支: ¥{int(agree_payout - agree_count*100):+,d}")

    # 2. 不一致レース (Disagreement)
    disagree_indices = np.where(disagree_mask)[0]
    disagree_count = len(disagree_indices)
    for model_name, preds in [("M29->M43", pred_te29), ("M39->M37", pred_te39)]:
        d_hits = 0
        d_payout = 0
        for idx in disagree_indices:
            horse = preds[idx]
            if horse == y_te[idx]:
                d_hits += 1
                d_payout += payouts_te[idx][horse]
        d_roi = d_payout / (disagree_count * 100) if disagree_count > 0 else 0
        d_hr = d_hits / disagree_count if disagree_count > 0 else 0
        print(f"【不一致レース ({disagree_count}R) での {model_name} の成績】:")
        print(f"  - 的中数: {d_hits} 勝 / 的中率: {d_hr*100:.2f}%")
        print(f"  - 払戻金: ¥{int(d_payout):,d} / 回収率: {d_roi*100:.2f}% / 収支: ¥{int(d_payout - disagree_count*100):+,d}")

    agreement_summary = {
        "test_total_races": num_te_races,
        "agree_races": int(agree_count),
        "agree_ratio": float(agree_count / num_te_races),
        "agree_hits": int(agree_hits),
        "agree_hit_rate": float(agree_hit_rate),
        "agree_payout": int(agree_payout),
        "agree_roi": float(agree_roi),
        "disagree_races": int(disagree_count),
        "disagree_ratio": float(disagree_count / num_te_races),
    }
    with open(os.path.join(output_dir, "agreement_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(agreement_summary, f, ensure_ascii=False, indent=2)

    print("\n評価完了。結果を保存しました:", csv_path)

if __name__ == "__main__":
    main()
