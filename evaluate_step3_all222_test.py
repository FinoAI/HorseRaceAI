"""
evaluate_step3_all222_test.py
========================================================================
芝3段カスケード抽出全222モデルのTESTデータ(827レース)評価スクリプト
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import setup_logger, set_seed, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("EvalStep3All222Test")

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
    logger.info(f"Loading cached turf data from {cache_path}...")
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

    # 3. 3段カスケード抽出全222モデルロード
    ext_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step3/extracted_turf_step3_triplets.csv")
    df_s3 = pd.read_csv(ext_csv)
    logger.info(f"Loaded {len(df_s3)} Step 3 triplets from {ext_csv}")

    # Step 2 の推論確率キャッシュ (s1, s2) -> P2 (827, 18)
    p2_cache = {}

    results = []
    t_start = time.time()

    for idx, row in df_s3.iterrows():
        s1 = int(row["m_step1"])
        s2 = int(row["m_step2"])
        s3 = int(row["m_step3"])
        triplet_name = row["triplet_name"]

        # Step 2 推論 (キャッシュがあれば再利用)
        pair_key = (s1, s2)
        if pair_key not in p2_cache:
            p1_te = te_step1_probs[s1 - 1]
            s2_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_exhaustive/found_M{s1}_to_M{s2}.pth")
            net_s2 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
            net_s2.load_state_dict(torch.load(s2_weights_path, map_location=device))
            net_s2.eval()

            f2_idx = model_feat_indices[s2]
            X_te2 = np.concatenate([data_te["race_3d"][:, :, f2_idx], p1_te[:, :, np.newaxis]], axis=-1)
            X_te2_t = torch.tensor(X_te2.reshape(-1, 18*201), dtype=torch.float32)
            p2_te = predict_batches(net_s2, X_te2_t, mask_te_t, device)
            p2_cache[pair_key] = p2_te
            del net_s2, X_te2, X_te2_t

        p2_te = p2_cache[pair_key]

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

        # TEST 評価
        m3_te = calculate_race_array_payout_metric(
            probs=p3_te,
            masks=data_te["race_masks"],
            y_true=data_te["race_y"],
            odds=data_te["odds"],
            payouts=data_te["payouts"],
            selection_mode="prob"
        )

        res_item = {
            "triplet_name": triplet_name,
            "m_step1": s1,
            "m_step2": s2,
            "m_step3": s3,
            # Test metrics
            "te_hit_rate": m3_te["hit_rate"],
            "te_hits": m3_te["hits"],
            "te_roi": m3_te["roi"],
            "te_payout": m3_te["total_payout"],
            "te_mean_odds": m3_te["total_payout"] / (m3_te["hits"] * 100) if m3_te["hits"] > 0 else 0.0,
            # Val metrics
            "val_hit_rate": row["s3_val_hit_rate"],
            "val_hits": row["s3_val_hits"],
            "val_roi": row["s3_val_roi"],
            "val_payout": row["s3_val_payout"],
            # Gaps
            "gap_hit_rate": m3_te["hit_rate"] - row["s3_val_hit_rate"],
            "gap_roi": m3_te["roi"] - row["s3_val_roi"],
            "is_te_kuroji": bool(m3_te["roi"] >= 1.0),
            "is_te_hr20": bool(m3_te["hit_rate"] >= 0.20)
        }
        results.append(res_item)

        if (idx + 1) % 30 == 0 or (idx + 1) == len(df_s3):
            kuroji_count = sum(1 for r in results if r["is_te_kuroji"])
            logger.info(
                f"[{idx+1:3d}/{len(df_s3)}] {triplet_name:18s} | "
                f"TEST Hit: {m3_te['hit_rate']*100:5.2f}% ({m3_te['hits']}勝) | "
                f"TEST ROI: {m3_te['roi']*100:6.2f}% | 累計TEST黒字数: {kuroji_count}"
            )

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    df_res = pd.DataFrame(results)
    save_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step3/test_eval_all222_triplets.csv")
    df_res.to_csv(save_csv, index=False)
    logger.info(f"Evaluation complete in {time.time() - t_start:.1f}s! Saved to {save_csv}")

if __name__ == "__main__":
    main()
