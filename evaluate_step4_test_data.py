"""
evaluate_step4_test_data.py
========================================================================
芝4段カスケード 的中率25%以上達成 16モデルのTESTデータ(827レース)評価スクリプト
========================================================================
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import setup_logger, set_seed, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("EvalStep4Test")

def predict_batches(net, X_t, mask_t, device, eval_bs=512):
    net.eval()
    probs_list = []
    n = len(X_t)
    with torch.no_grad():
        for i in range(0, n, eval_bs):
            bx = X_t[i:i+eval_bs].to(device)
            bmask = mask_t[i:i+eval_bs].to(device)
            logits = net(bx, mask=bmask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)
    return np.concatenate(probs_list, axis=0)

def main():
    RANDOM_SEED = 2026
    set_seed(RANDOM_SEED)
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. キャッシュデータロード (芝・1勝クラス以上)
    cache_path = os.path.join(CURRENT_DIR, "artifacts_turf_1win/cached_turf_1win_data.pkl")
    logger.info(f"Loading cached precomputed turf data from {cache_path}...")
    with open(cache_path, "rb") as f:
        data_dict = pickle.load(f)

    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    def get_model_features(m):
        feat_path = os.path.join(CURRENT_DIR, f"artifacts_turf_1win/features_{m}.json")
        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_col_indices) + s_idx

    model_feat_indices = {m: get_model_features(m) for m in range(1, 51)}

    # 2. Step 1 (全50モデル) TEST 推論確率 (50, 827, 18)
    te_step1_probs = np.load(os.path.join(CURRENT_DIR, "artifacts_turf_1win/all_test_probs.npy"))
    mask_te_t = torch.tensor(data_te["race_masks"], dtype=torch.float32)

    # 3. 25%以上達成モデルのリストを取得
    ext_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step4/extracted_turf_step4_quads.csv")
    df_ext = pd.read_csv(ext_csv)
    hr25_df = df_ext[df_ext["is_hit_rate_25"] == True].sort_values(by=["s4_val_hit_rate", "s4_val_roi"], ascending=[False, False]).reset_index(drop=True)
    logger.info(f"Loaded {len(hr25_df)} models with Val HitRate >= 25.0%")

    results = []

    for idx, row in hr25_df.iterrows():
        s1 = int(row["m_step1"])
        s2 = int(row["m_step2"])
        s3 = int(row["m_step3"])
        s4 = int(row["m_step4"])
        quad_name = row["quad_name"]

        # Step 1 TEST 推論確率
        p1_te = te_step1_probs[s1 - 1]

        # Step 2 推論
        s2_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_exhaustive/found_M{s1}_to_M{s2}.pth")
        net_s2 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        net_s2.load_state_dict(torch.load(s2_weights_path, map_location=device))
        net_s2.eval()

        f2_idx = model_feat_indices[s2]
        X_te2 = np.concatenate([data_te["race_3d"][:, :, f2_idx], p1_te[:, :, np.newaxis]], axis=-1)
        X_te2_t = torch.tensor(X_te2.reshape(-1, 18*201), dtype=torch.float32)
        p2_te = predict_batches(net_s2, X_te2_t, mask_te_t, device)
        del net_s2, X_te2, X_te2_t

        # Step 3 推論
        s3_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_step3/found_M{s1}_M{s2}_to_M{s3}.pth")
        net_s3 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        net_s3.load_state_dict(torch.load(s3_weights_path, map_location=device))
        net_s3.eval()

        f3_idx = model_feat_indices[s3]
        X_te3 = np.concatenate([data_te["race_3d"][:, :, f3_idx], p2_te[:, :, np.newaxis]], axis=-1)
        X_te3_t = torch.tensor(X_te3.reshape(-1, 18*201), dtype=torch.float32)
        p3_te = predict_batches(net_s3, X_te3_t, mask_te_t, device)
        del net_s3, X_te3, X_te3_t

        # Step 3 の TEST 評価も算出
        m3_te = calculate_race_array_payout_metric(
            probs=p3_te,
            masks=data_te["race_masks"],
            y_true=data_te["race_y"],
            odds=data_te["odds"],
            payouts=data_te["payouts"],
            selection_mode="prob"
        )

        # Step 4 推論
        s4_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_step4/found_M{s1}_M{s2}_M{s3}_to_M{s4}.pth")
        net_s4 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        net_s4.load_state_dict(torch.load(s4_weights_path, map_location=device))
        net_s4.eval()

        f4_idx = model_feat_indices[s4]
        X_te4 = np.concatenate([data_te["race_3d"][:, :, f4_idx], p3_te[:, :, np.newaxis]], axis=-1)
        X_te4_t = torch.tensor(X_te4.reshape(-1, 18*201), dtype=torch.float32)
        p4_te = predict_batches(net_s4, X_te4_t, mask_te_t, device)
        del net_s4, X_te4, X_te4_t

        # Step 4 の TEST 評価
        m4_te = calculate_race_array_payout_metric(
            probs=p4_te,
            masks=data_te["race_masks"],
            y_true=data_te["race_y"],
            odds=data_te["odds"],
            payouts=data_te["payouts"],
            selection_mode="prob"
        )

        res_item = {
            "rank_val": idx + 1,
            "quad_name": quad_name,
            # Test metrics (Step 4)
            "te_hit_rate": m4_te["hit_rate"],
            "te_hits": m4_te["hits"],
            "te_roi": m4_te["roi"],
            "te_payout": m4_te["total_payout"],
            "te_mean_odds": m4_te["total_payout"] / (m4_te["hits"] * 100) if m4_te["hits"] > 0 else 0.0,
            # Step 3 Test metrics (比較用)
            "s3_te_hit_rate": m3_te["hit_rate"],
            "s3_te_hits": m3_te["hits"],
            "s3_te_roi": m3_te["roi"],
            # Test Delta (Step 4 - Step 3 on Test)
            "delta_te_hit_rate": m4_te["hit_rate"] - m3_te["hit_rate"],
            "delta_te_roi": m4_te["roi"] - m3_te["roi"],
            # Val metrics (参考)
            "val_hit_rate": row["s4_val_hit_rate"],
            "val_hits": row["s4_val_hits"],
            "val_roi": row["s4_val_roi"],
            "val_payout": row["s4_val_payout"],
            # Val-Test 乖離
            "gap_hit_rate": m4_te["hit_rate"] - row["s4_val_hit_rate"],
            "gap_roi": m4_te["roi"] - row["s4_val_roi"],
        }
        results.append(res_item)

        kuroji = "★黒字" if m4_te["roi"] >= 1.0 else "      "
        logger.info(
            f"[{idx+1:2d}/16] {quad_name:22s} | "
            f"TEST Hit: {m4_te['hit_rate']*100:5.2f}% ({m4_te['hits']:3d}/827R, ΔS3:{res_item['delta_te_hit_rate']*100:+4.1f}pt) | "
            f"TEST ROI: {m4_te['roi']*100:6.2f}% (払戻:¥{int(m4_te['total_payout']):,}) {kuroji} | "
            f"VAL Hit: {row['s4_val_hit_rate']*100:5.2f}%"
        )

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    df_res = pd.DataFrame(results)
    save_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step4/test_eval_hr25_models.csv")
    df_res.to_csv(save_csv, index=False)
    logger.info(f"Saved test evaluation results to {save_csv}")

if __name__ == "__main__":
    main()
