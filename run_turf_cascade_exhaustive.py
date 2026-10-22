"""
run_turf_cascade_exhaustive.py
========================================================================
芝・1勝クラス以上 2段カスケードチェーン 2,450ペア完全総当たり探索
(VALのみ評価 / TEST不要 / 的中率20%以上 & 回収率90%以上 & 元モデル向上抽出)
========================================================================
■ 探索・抽出仕様:
1. 対象: 50モデル × 49モデル = 2,450 通りの 2段カスケード (Mi -> Mj, i != j)
2. 評価データ: VAL (826R) のみ使用 (TESTは不要・スキップ)
3. 再学習: 2009〜2024年 (15,547R) のみで Step 2 NN (201特徴量) をゼロから学習 (epochs=3)
4. 抽出条件:
   - Val 的中率 >= 20.0%
   - Val 回収率 >= 90.0%
   - 元のモデル (起点 Mi) より回収率が向上していること (Val ROI >= Base Val ROI)
5. 最適化:
   - 全50モデルの Step 1 予測確率 (Train & Val) を事前一括計算・キャッシュ
   - 抽出条件合致モデルの重みと詳細をリアルタイム保存
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

logger = setup_logger("TurfCascadeExhaustive")

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
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_exhaustive")
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

    # 元の単体50モデル成績ロード
    single_csv = os.path.join(CURRENT_DIR, "artifacts_turf_1win/single_models_performance.csv")
    df_single = pd.read_csv(single_csv)
    base_stats = {}
    for _, row in df_single.iterrows():
        base_stats[int(row["model_idx"])] = {
            "val_roi": float(row["val_roi"]),
            "val_hit_rate": float(row["val_hit_rate"]),
            "val_score": float(row["val_score"])
        }

    def get_model_features(m):
        feat_path = os.path.join(CURRENT_DIR, f"artifacts_turf_1win/features_{m}.json")
        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_col_indices) + s_idx

    # 各モデルの特徴量インデックスを事前辞書化
    model_feat_indices = {m: get_model_features(m) for m in range(1, 51)}

    # 2. Step 1 (全50モデル) の事前推論確率の一括生成 & キャッシュ
    step1_cache_path = os.path.join(artifacts_dir, "step1_all_50_probs.pkl")
    if os.path.exists(step1_cache_path):
        logger.info(f"Loading precomputed Step 1 probabilities from {step1_cache_path}...")
        with open(step1_cache_path, "rb") as f:
            step1_probs = pickle.load(f)
    else:
        logger.info("Computing Step 1 probabilities for all 50 models on Train and Val...")
        step1_probs = {}
        mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
        mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

        for m in range(1, 51):
            t0 = time.time()
            f_idx = model_feat_indices[m]
            net = RaceLevelStage1NN(input_dim=18*200, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
            w_path = os.path.join(CURRENT_DIR, f"artifacts_turf_1win/model_{m}.pth")
            net.load_state_dict(torch.load(w_path, map_location=device))
            net.eval()

            X_tr_m = torch.tensor(data_tr["race_3d"][:, :, f_idx].reshape(-1, 18*200), dtype=torch.float32)
            X_va_m = torch.tensor(data_va["race_3d"][:, :, f_idx].reshape(-1, 18*200), dtype=torch.float32)

            tr_p = predict_batches(net, X_tr_m, mask_tr_t, device)
            va_p = predict_batches(net, X_va_m, mask_va_t, device)

            step1_probs[m] = {
                "tr_probs": tr_p.astype(np.float32),
                "va_probs": va_p.astype(np.float32)
            }
            del net, X_tr_m, X_va_m
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if m % 10 == 0 or m == 50:
                logger.info(f"  Step 1 precomputed: {m}/50 models ({time.time()-t0:.1f}s)")

        with open(step1_cache_path, "wb") as f:
            pickle.dump(step1_probs, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved Step 1 cache to {step1_cache_path}!")

    # 3. 2段カスケード (2,450通り) の総当たり探索
    all_pairs = [(i, j) for i in range(1, 51) for j in range(1, 51) if i != j]
    total_pairs = len(all_pairs)
    logger.info(f"Total cascade pairs to evaluate: {total_pairs} pairs (50 x 49)")

    # 既存進捗チェック (途中再開用)
    res_csv_path = os.path.join(artifacts_dir, "all_turf_2450_cascade_results.csv")
    ext_csv_path = os.path.join(artifacts_dir, "extracted_turf_cascade_pairs.csv")

    completed_pairs = set()
    all_results = []
    extracted_results = []

    if os.path.exists(res_csv_path):
        df_prev = pd.read_csv(res_csv_path)
        for _, r in df_prev.iterrows():
            pair_key = (int(r["origin_model"]), int(r["step2_model"]))
            completed_pairs.add(pair_key)
            all_results.append(r.to_dict())
        logger.info(f"Resuming search: {len(completed_pairs)} pairs already evaluated.")

    if os.path.exists(ext_csv_path):
        df_ext_prev = pd.read_csv(ext_csv_path)
        extracted_results = df_ext_prev.to_dict(orient="records")

    # CSVヘッダー初期化
    fieldnames = [
        "origin_model", "step2_model", "pair_name",
        "val_roi", "val_hit_rate", "val_hits", "val_payout", "val_mean_odds",
        "base_val_roi", "base_val_hit_rate", "delta_roi", "delta_hit_rate",
        "is_extracted", "time_sec"
    ]
    if not os.path.exists(res_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(res_csv_path, index=False)
    if not os.path.exists(ext_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(ext_csv_path, index=False)

    num_races_tr = len(data_tr["race_y"])
    num_races_va = len(data_va["race_y"])
    mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr_t = torch.tensor(data_tr["race_y"], dtype=torch.int64)
    mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    logger.info("=== Starting Exhaustive Cascade Search (VAL-only) ===")
    t_start = time.time()
    count_done = len(completed_pairs)

    # 起点ごとにグループ化して高速化
    for m_origin in range(1, 51):
        # 起点の推論確率 (Train & Val)
        p1_tr = step1_probs[m_origin]["tr_probs"] # (15547, 18)
        p1_va = step1_probs[m_origin]["va_probs"] # (826, 18)
        base_roi = base_stats[m_origin]["val_roi"]
        base_hr = base_stats[m_origin]["val_hit_rate"]

        for m_step2 in range(1, 51):
            if m_origin == m_step2:
                continue

            pair_key = (m_origin, m_step2)
            if pair_key in completed_pairs:
                continue

            t_pair0 = time.time()

            # Step 2 の入力特徴量 (201次元)
            f2_idx = model_feat_indices[m_step2]
            X_tr2 = np.concatenate([data_tr["race_3d"][:, :, f2_idx], p1_tr[:, :, np.newaxis]], axis=-1)
            X_va2 = np.concatenate([data_va["race_3d"][:, :, f2_idx], p1_va[:, :, np.newaxis]], axis=-1)

            X_tr2_flat = X_tr2.reshape(-1, 18 * 201)
            X_va2_flat = X_va2.reshape(-1, 18 * 201)

            X_tr2_t = torch.tensor(X_tr2_flat, dtype=torch.float32)
            X_va2_t = torch.tensor(X_va2_flat, dtype=torch.float32)

            # Step 2 NN 構築 & 学習 (epochs=3 高速収束)
            model = RaceLevelStage1NN(
                input_dim=18 * 201,
                max_horses=18,
                multiplier=2,
                reduction_ratio=0.5,
                dropout=0.15
            ).to(device)

            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            criterion = nn.CrossEntropyLoss()

            best_val_score = -1.0
            best_val_probs = None
            best_model_state = None

            for epoch in range(1, 4):
                model.train()
                perm = torch.randperm(num_races_tr)
                batch_size = 512

                for i in range(0, num_races_tr, batch_size):
                    idx = perm[i:i + batch_size]
                    bx = X_tr2_t[idx].to(device)
                    bmask = mask_tr_t[idx].to(device)
                    by = y_tr_t[idx].to(device)

                    optimizer.zero_grad()
                    logits = model(bx, mask=bmask)
                    loss = criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Val 評価
                val_probs = predict_batches(model, X_va2_t, mask_va_t, device)
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
                    best_val_probs = val_probs
                    best_metrics = metrics
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

            dt_pair = time.time() - t_pair0

            val_roi = best_metrics["roi"]
            val_hr = best_metrics["hit_rate"]
            val_hits = best_metrics["hits"]
            val_pay = best_metrics["total_payout"]
            val_odds = val_pay / (val_hits * 100) if val_hits > 0 else 0.0

            delta_roi = val_roi - base_roi
            delta_hr = val_hr - base_hr

            # 抽出条件: 的中率 >= 20% かつ 回収率 >= 90% かつ 元モデルより回収率向上 (delta_roi >= 0)
            is_extracted = (val_hr >= 0.20) and (val_roi >= 0.90) and (val_roi >= base_roi)

            pair_name = f"M{m_origin}->M{m_step2}"
            row = {
                "origin_model": m_origin,
                "step2_model": m_step2,
                "pair_name": pair_name,
                "val_roi": val_roi,
                "val_hit_rate": val_hr,
                "val_hits": val_hits,
                "val_payout": val_pay,
                "val_mean_odds": val_odds,
                "base_val_roi": base_roi,
                "base_val_hit_rate": base_hr,
                "delta_roi": delta_roi,
                "delta_hit_rate": delta_hr,
                "is_extracted": is_extracted,
                "time_sec": dt_pair
            }

            all_results.append(row)
            completed_pairs.add(pair_key)
            count_done += 1

            # リアルタイム追記
            pd.DataFrame([row]).to_csv(res_csv_path, mode="a", header=False, index=False)

            if is_extracted:
                extracted_results.append(row)
                pd.DataFrame([row]).to_csv(ext_csv_path, mode="a", header=False, index=False)
                # モデル重み保存
                pth_save = os.path.join(artifacts_dir, f"found_M{m_origin}_to_M{m_step2}.pth")
                torch.save(best_model_state, pth_save)

                logger.info(
                    f"★【条件合致抽出!】 {pair_name:12s} => Val ROI: {val_roi*100:6.2f}% (元: {base_roi*100:6.2f}%, {delta_roi*100:+5.2f}pt) | "
                    f"Val Hit: {val_hr*100:4.1f}% ({val_hits}勝) | 払戻: ¥{int(val_pay):,d} | 累計抽出数: {len(extracted_results)}"
                )

            del model, optimizer, scheduler, X_tr2, X_va2, X_tr2_flat, X_va2_flat, X_tr2_t, X_va2_t
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # 進捗ログ
            if count_done % 50 == 0 or count_done == total_pairs:
                elapsed = time.time() - t_start
                rate = count_done / elapsed if elapsed > 0 else 1
                eta_sec = (total_pairs - count_done) / rate if rate > 0 else 0
                logger.info(
                    f"[Progress: {count_done:4d}/{total_pairs} ({count_done/total_pairs*100:5.1f}%)] "
                    f"抽出数: {len(extracted_results):3d}件 | 経過: {elapsed/60:5.1f}分 | 残り予定: {eta_sec/60:5.1f}分"
                )

    # 4. 探索終了・まとめ
    logger.info("==================================================================")
    logger.info(f"全 {total_pairs} 通り走破完了！")
    logger.info(f"条件合致抽出数: {len(extracted_results)} ペア")
    logger.info(f"結果保存先: {ext_csv_path}")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
