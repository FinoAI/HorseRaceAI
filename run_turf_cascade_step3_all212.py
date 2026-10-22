"""
run_turf_cascade_step3_all212.py
========================================================================
芝・1勝クラス以上 抽出212ペア起点 3段カスケード総当たり探索 (10,176通り)
(VALのみ評価 / TEST不要 / チェーン元より回収率・的中率が両方向上したもののみ抽出)
========================================================================
■ 探索・抽出仕様:
1. 対象: 2段カスケード抽出 212ペア (Mi -> Mj) × 残り48モデル (Mk) = 10,176 通り
2. 評価データ: VAL (826レース) のみ使用 (TESTは不要・スキップ)
3. 再学習: 2009〜2024年 (15,547R) のみで Step 3 NN (201特徴量) をゼロから学習 (epochs=3)
4. 抽出条件:
   - チェーン元 (Mi -> Mj) より VAL 回収率が向上 (s3_val_roi > s2_val_roi)
   - チェーン元 (Mi -> Mj) より VAL 的中率が向上 (s3_val_hit_rate > s2_val_hit_rate)
   (回収率・的中率のダブル向上のみ厳選抽出！)
5. 最適化:
   - Step 1 (Mi) の確率は既存キャッシュから即座に取得
   - Step 2 (Mi -> Mj) の推論確率は各ペア開始時に1度だけ計算してメモリ保持
   - 途中再開 (Resume) 完全対応、抽出モデル重み即時保存
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
import torch.optim as optim

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import setup_logger, set_seed, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("TurfCascadeStep3All212")

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
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step3")
    os.makedirs(artifacts_dir, exist_ok=True)

    RANDOM_SEED = 2026
    set_seed(RANDOM_SEED)
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. キャッシュデータロード (芝・1勝クラス以上)
    cache_path = os.path.join(CURRENT_DIR, "artifacts_turf_1win/cached_turf_1win_data.pkl")
    logger.info(f"Loading cached precomputed turf data from {cache_path}...")
    with open(cache_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    # 2. 抽出された2段カスケードペア (212ペア) ロード
    step2_ext_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_exhaustive/extracted_turf_cascade_pairs.csv")
    df_step2 = pd.read_csv(step2_ext_csv)
    logger.info(f"Loaded {len(df_step2)} Step 2 extracted pairs from {step2_ext_csv}!")

    def get_model_features(m):
        feat_path = os.path.join(CURRENT_DIR, f"artifacts_turf_1win/features_{m}.json")
        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_col_indices) + s_idx

    model_feat_indices = {m: get_model_features(m) for m in range(1, 51)}

    # 3. Step 1 (全50モデル) 事前推論確率キャッシュのロード
    step1_cache_path = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_exhaustive/step1_all_50_probs.pkl")
    logger.info(f"Loading Step 1 cache from {step1_cache_path}...")
    with open(step1_cache_path, "rb") as f:
        step1_probs = pickle.load(f)

    # 4. 進行管理と途中再開 (Resume) のセットアップ
    res_csv_path = os.path.join(artifacts_dir, "all_turf_10176_step3_results.csv")
    ext_csv_path = os.path.join(artifacts_dir, "extracted_turf_step3_triplets.csv")

    completed_triplets = set()
    extracted_triplets = []

    if os.path.exists(res_csv_path):
        df_prev = pd.read_csv(res_csv_path)
        for _, r in df_prev.iterrows():
            key = (int(r["m_step1"]), int(r["m_step2"]), int(r["m_step3"]))
            completed_triplets.add(key)
        logger.info(f"Resuming: {len(completed_triplets)} triplets already evaluated.")

    if os.path.exists(ext_csv_path):
        df_ext_prev = pd.read_csv(ext_csv_path)
        extracted_triplets = df_ext_prev.to_dict(orient="records")
        logger.info(f"Loaded {len(extracted_triplets)} already extracted triplets.")

    fieldnames = [
        "m_step1", "m_step2", "m_step3", "triplet_name",
        "s3_val_roi", "s3_val_hit_rate", "s3_val_hits", "s3_val_payout", "s3_val_mean_odds",
        "s2_val_roi", "s2_val_hit_rate", "delta_roi", "delta_hit_rate",
        "is_extracted", "time_sec"
    ]
    if not os.path.exists(res_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(res_csv_path, index=False)
    if not os.path.exists(ext_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(ext_csv_path, index=False)

    num_races_tr = len(data_tr["race_y"])
    mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr_t = torch.tensor(data_tr["race_y"], dtype=torch.int64)
    mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    total_triplets = len(df_step2) * 48 # 212 * 48 = 10,176
    logger.info(f"Total Step 3 triplets to search: {total_triplets} combinations")

    t_start = time.time()
    count_done = len(completed_triplets)

    # 5. 212ペア × 48モデルのループ
    for pair_idx, pair_row in df_step2.iterrows():
        m_s1 = int(pair_row["origin_model"])
        m_s2 = int(pair_row["step2_model"])
        s2_val_roi = float(pair_row["val_roi"])
        s2_val_hr = float(pair_row["val_hit_rate"])

        # 未完了の候補があるかチェック
        candidates_s3 = [m for m in range(1, 51) if m != m_s1 and m != m_s2]
        uncompleted_for_pair = [m for m in candidates_s3 if (m_s1, m_s2, m) not in completed_triplets]
        if not uncompleted_for_pair:
            continue

        # Step 2 モデル (found_M{s1}_to_M{s2}.pth) のロード & 推論確率 P2 の生成
        s2_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_exhaustive/found_M{m_s1}_to_M{m_s2}.pth")
        if not os.path.exists(s2_weights_path):
            logger.warning(f"Weights not found: {s2_weights_path}. Skipping pair.")
            continue

        net_s2 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        net_s2.load_state_dict(torch.load(s2_weights_path, map_location=device))
        net_s2.eval()

        p1_tr = step1_probs[m_s1]["tr_probs"]
        p1_va = step1_probs[m_s1]["va_probs"]
        f2_idx = model_feat_indices[m_s2]

        X_tr2 = np.concatenate([data_tr["race_3d"][:, :, f2_idx], p1_tr[:, :, np.newaxis]], axis=-1)
        X_va2 = np.concatenate([data_va["race_3d"][:, :, f2_idx], p1_va[:, :, np.newaxis]], axis=-1)

        X_tr2_t = torch.tensor(X_tr2.reshape(-1, 18*201), dtype=torch.float32)
        X_va2_t = torch.tensor(X_va2.reshape(-1, 18*201), dtype=torch.float32)

        p2_tr = predict_batches(net_s2, X_tr2_t, mask_tr_t, device)
        p2_va = predict_batches(net_s2, X_va2_t, mask_va_t, device)

        del net_s2, X_tr2, X_va2, X_tr2_t, X_va2_t
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Step 3 (48モデル) の探索ループ
        for m_s3 in candidates_s3:
            triplet_key = (m_s1, m_s2, m_s3)
            if triplet_key in completed_triplets:
                continue

            t_t0 = time.time()

            # Step 3 入力 (M_s3 の 200特徴量 + P2) = 201特徴量
            f3_idx = model_feat_indices[m_s3]
            X_tr3 = np.concatenate([data_tr["race_3d"][:, :, f3_idx], p2_tr[:, :, np.newaxis]], axis=-1)
            X_va3 = np.concatenate([data_va["race_3d"][:, :, f3_idx], p2_va[:, :, np.newaxis]], axis=-1)

            X_tr3_t = torch.tensor(X_tr3.reshape(-1, 18*201), dtype=torch.float32)
            X_va3_t = torch.tensor(X_va3.reshape(-1, 18*201), dtype=torch.float32)

            model_s3 = RaceLevelStage1NN(
                input_dim=18*201,
                max_horses=18,
                multiplier=2,
                reduction_ratio=0.5,
                dropout=0.15
            ).to(device)

            optimizer = optim.AdamW(model_s3.parameters(), lr=0.001, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            criterion = nn.CrossEntropyLoss()

            best_val_score = -1.0
            best_metrics = None
            best_model_state = None

            for epoch in range(1, 4):
                model_s3.train()
                perm = torch.randperm(num_races_tr)
                batch_size = 512

                for i in range(0, num_races_tr, batch_size):
                    idx = perm[i:i + batch_size]
                    bx = X_tr3_t[idx].to(device)
                    bmask = mask_tr_t[idx].to(device)
                    by = y_tr_t[idx].to(device)

                    optimizer.zero_grad()
                    logits = model_s3(bx, mask=bmask)
                    loss = criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_s3.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Val 評価
                val_probs = predict_batches(model_s3, X_va3_t, mask_va_t, device)
                metrics = calculate_race_array_payout_metric(
                    probs=val_probs,
                    masks=data_va["race_masks"],
                    y_true=data_va["race_y"],
                    odds=data_va["odds"],
                    payouts=data_va["payouts"],
                    selection_mode="prob"
                )
                val_score = metrics["roi"] * min(1.0, metrics["hit_rate"] / 0.20) + metrics["hit_rate"]

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_metrics = metrics
                    best_model_state = {k: v.cpu() for k, v in model_s3.state_dict().items()}

            dt_t = time.time() - t_t0

            s3_val_roi = best_metrics["roi"]
            s3_val_hr = best_metrics["hit_rate"]
            s3_val_hits = best_metrics["hits"]
            s3_val_pay = best_metrics["total_payout"]
            s3_val_odds = s3_val_pay / (s3_val_hits * 100) if s3_val_hits > 0 else 0.0

            delta_roi = s3_val_roi - s2_val_roi
            delta_hr = s3_val_hr - s2_val_hr

            # 厳密抽出条件: チェーン元より回収率・的中率が両方向上！
            is_extracted = (s3_val_roi > s2_val_roi) and (s3_val_hr > s2_val_hr)

            triplet_name = f"M{m_s1}->M{m_s2}->M{m_s3}"
            row = {
                "m_step1": m_s1,
                "m_step2": m_s2,
                "m_step3": m_s3,
                "triplet_name": triplet_name,
                "s3_val_roi": s3_val_roi,
                "s3_val_hit_rate": s3_val_hr,
                "s3_val_hits": s3_val_hits,
                "s3_val_payout": s3_val_pay,
                "s3_val_mean_odds": s3_val_odds,
                "s2_val_roi": s2_val_roi,
                "s2_val_hit_rate": s2_val_hr,
                "delta_roi": delta_roi,
                "delta_hit_rate": delta_hr,
                "is_extracted": is_extracted,
                "time_sec": dt_t
            }

            completed_triplets.add(triplet_key)
            count_done += 1

            # リアルタイム追記
            pd.DataFrame([row]).to_csv(res_csv_path, mode="a", header=False, index=False)

            if is_extracted:
                extracted_triplets.append(row)
                pd.DataFrame([row]).to_csv(ext_csv_path, mode="a", header=False, index=False)
                # モデル重み保存
                pth_save = os.path.join(artifacts_dir, f"found_M{m_s1}_M{m_s2}_to_M{m_s3}.pth")
                torch.save(best_model_state, pth_save)

                logger.info(
                    f"★【両方向上抽出!】 {triplet_name:16s} => Val ROI: {s3_val_roi*100:6.2f}% ({delta_roi*100:+5.2f}pt) | "
                    f"Val Hit: {s3_val_hr*100:4.1f}% ({delta_hr*100:+4.1f}pt, {s3_val_hits}勝) | 累計抽出数: {len(extracted_triplets)}"
                )

            del model_s3, optimizer, scheduler, X_tr3, X_va3, X_tr3_t, X_va3_t
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # 進捗ログ
            if count_done % 100 == 0 or count_done == total_triplets:
                elapsed = time.time() - t_start
                rate = count_done / elapsed if elapsed > 0 else 1
                eta_sec = (total_triplets - count_done) / rate if rate > 0 else 0
                logger.info(
                    f"[Progress: {count_done:5d}/{total_triplets} ({count_done/total_triplets*100:5.1f}%)] "
                    f"Pair {pair_idx+1}/{len(df_step2)} | 抽出数: {len(extracted_triplets):4d}件 | "
                    f"経過: {elapsed/60:5.1f}分 | 残り予定: {eta_sec/60:5.1f}分"
                )

    logger.info("==================================================================")
    logger.info(f"全 {total_triplets} 通り走破完了！")
    logger.info(f"両方向上 抽出数: {len(extracted_triplets)} トリプレット")
    logger.info(f"結果保存先: {ext_csv_path}")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
