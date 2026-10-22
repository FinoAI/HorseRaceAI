"""
run_turf_cascade_step4_case1.py
========================================================================
芝・1勝クラス以上 4段カスケード総当たり探索 (案1: 3,680通り)
(VAL 826レース評価 / TEST不要 / M45除外 / 的中率23%以上の3段モデル80組起点)
========================================================================
■ 探索・抽出仕様:
1. 起点: 芝3段カスケード抽出モデルのうち VAL 的中率 >= 23.0% の 80トリプレット (M_s1 -> M_s2 -> M_s3)
2. 4段目候補モデル (M_s4):
   - 全50モデルから (M_s1, M_s2, M_s3) の3モデルを除外
   - さらに非貢献モデル M45 を除外 (残り46モデル)
   - 総組み合わせ数: 80 × 46 = 3,680 通り
3. 評価データ: VAL (826レース) のみで評価 (TESTデータは完全不使用)
4. 入力特徴量: M_s4 の基本200特徴量 + Step 3推論確率 P3 (計201特徴量)
5. 学習: 2009〜2024年 (15,547R) で Step 4 NN をゼロから学習 (epochs=3)
6. 抽出基準:
   - 的中率目標: VAL 的中率 >= 25.0% (is_hit_rate_25)
   - 改善条件: チェーン元 (3段モデル) より回収率・的中率が両方向上 (is_improved)
   - 厳選抽出フラグ: is_extracted = is_hit_rate_25 (的中率25%以上達成)
7. 耐障害性 (Resume):
   - 1試行ごとにCSV即時追記 & キャッシュクリア
   - プロセス中断時も、未実施の組み合わせから自動で完全再開可能
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

logger = setup_logger("TurfCascadeStep4Case1")

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
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step4")
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

    # 2. 3段カスケード抽出結果から「的中率 >= 23%」の80トリプレットをロード
    step3_ext_csv = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_step3/extracted_turf_step3_triplets.csv")
    df_step3_all = pd.read_csv(step3_ext_csv)
    df_step3 = df_step3_all[df_step3_all["s3_val_hit_rate"] >= 0.23].copy().reset_index(drop=True)
    logger.info(f"Loaded {len(df_step3)} Step 3 triplets with Val HitRate >= 23.0% (out of {len(df_step3_all)})")

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
    res_csv_path = os.path.join(artifacts_dir, "all_turf_3680_step4_results.csv")
    ext_csv_path = os.path.join(artifacts_dir, "extracted_turf_step4_quads.csv")

    completed_quads = set()
    extracted_quads = []

    if os.path.exists(res_csv_path):
        df_prev = pd.read_csv(res_csv_path)
        for _, r in df_prev.iterrows():
            key = (int(r["m_step1"]), int(r["m_step2"]), int(r["m_step3"]), int(r["m_step4"]))
            completed_quads.add(key)
        logger.info(f"Resuming: {len(completed_quads)} quads already evaluated.")

    if os.path.exists(ext_csv_path):
        df_ext_prev = pd.read_csv(ext_csv_path)
        extracted_quads = df_ext_prev.to_dict(orient="records")
        logger.info(f"Loaded {len(extracted_quads)} already extracted quads.")

    fieldnames = [
        "m_step1", "m_step2", "m_step3", "m_step4", "quad_name",
        "s4_val_roi", "s4_val_hit_rate", "s4_val_hits", "s4_val_payout", "s4_val_mean_odds",
        "s3_val_roi", "s3_val_hit_rate", "delta_roi", "delta_hit_rate",
        "is_hit_rate_25", "is_improved", "is_extracted", "time_sec"
    ]
    if not os.path.exists(res_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(res_csv_path, index=False)
    if not os.path.exists(ext_csv_path):
        pd.DataFrame(columns=fieldnames).to_csv(ext_csv_path, index=False)

    num_races_tr = len(data_tr["race_y"])
    mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr_t = torch.tensor(data_tr["race_y"], dtype=torch.int64)
    mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    total_quads = len(df_step3) * 46  # 80 × 46 = 3,680
    logger.info(f"Total Step 4 quads to search: {total_quads} combinations (80 triplets × 46 models, excluding M45)")

    t_start = time.time()
    count_done = len(completed_quads)

    # 5. 80トリプレット × 46モデルのループ
    for trip_idx, trip_row in df_step3.iterrows():
        m_s1 = int(trip_row["m_step1"])
        m_s2 = int(trip_row["m_step2"])
        m_s3 = int(trip_row["m_step3"])
        s3_val_roi = float(trip_row["s3_val_roi"])
        s3_val_hr = float(trip_row["s3_val_hit_rate"])

        # 4段目候補 (そのトリプレット構成モデル3個、および M45 を除外)
        candidates_s4 = [m for m in range(1, 51) if m not in (m_s1, m_s2, m_s3) and m != 45]
        uncompleted_for_trip = [m for m in candidates_s4 if (m_s1, m_s2, m_s3, m) not in completed_quads]
        if not uncompleted_for_trip:
            continue

        # ---------------------------------------------------------
        # Step 2推論: M_s1 -> M_s2 の推論確率 P2 を取得
        # ---------------------------------------------------------
        s2_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_exhaustive/found_M{m_s1}_to_M{m_s2}.pth")
        if not os.path.exists(s2_weights_path):
            logger.warning(f"Step 2 weights not found: {s2_weights_path}. Skipping triplet.")
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

        # ---------------------------------------------------------
        # Step 3推論: M_s1 -> M_s2 -> M_s3 の推論確率 P3 を取得
        # ---------------------------------------------------------
        s3_weights_path = os.path.join(CURRENT_DIR, f"artifacts_turf_cascade_step3/found_M{m_s1}_M{m_s2}_to_M{m_s3}.pth")
        if not os.path.exists(s3_weights_path):
            logger.warning(f"Step 3 weights not found: {s3_weights_path}. Skipping triplet.")
            continue

        net_s3 = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        net_s3.load_state_dict(torch.load(s3_weights_path, map_location=device))
        net_s3.eval()

        f3_idx = model_feat_indices[m_s3]
        X_tr3 = np.concatenate([data_tr["race_3d"][:, :, f3_idx], p2_tr[:, :, np.newaxis]], axis=-1)
        X_va3 = np.concatenate([data_va["race_3d"][:, :, f3_idx], p2_va[:, :, np.newaxis]], axis=-1)
        X_tr3_t = torch.tensor(X_tr3.reshape(-1, 18*201), dtype=torch.float32)
        X_va3_t = torch.tensor(X_va3.reshape(-1, 18*201), dtype=torch.float32)

        p3_tr = predict_batches(net_s3, X_tr3_t, mask_tr_t, device)
        p3_va = predict_batches(net_s3, X_va3_t, mask_va_t, device)

        del net_s3, X_tr3, X_va3, X_tr3_t, X_va3_t, p2_tr, p2_va
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # ---------------------------------------------------------
        # Step 4 (46モデル) の探索ループ
        # ---------------------------------------------------------
        for m_s4 in candidates_s4:
            quad_key = (m_s1, m_s2, m_s3, m_s4)
            if quad_key in completed_quads:
                continue

            t_t0 = time.time()

            # Step 4 入力: M_s4 の 200特徴量 + P3 = 201特徴量
            f4_idx = model_feat_indices[m_s4]
            X_tr4 = np.concatenate([data_tr["race_3d"][:, :, f4_idx], p3_tr[:, :, np.newaxis]], axis=-1)
            X_va4 = np.concatenate([data_va["race_3d"][:, :, f4_idx], p3_va[:, :, np.newaxis]], axis=-1)

            X_tr4_t = torch.tensor(X_tr4.reshape(-1, 18*201), dtype=torch.float32)
            X_va4_t = torch.tensor(X_va4.reshape(-1, 18*201), dtype=torch.float32)

            model_s4 = RaceLevelStage1NN(
                input_dim=18*201,
                max_horses=18,
                multiplier=2,
                reduction_ratio=0.5,
                dropout=0.15
            ).to(device)

            optimizer = optim.AdamW(model_s4.parameters(), lr=0.001, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
            criterion = nn.CrossEntropyLoss()

            best_val_score = -1.0
            best_metrics = None
            best_model_state = None

            for epoch in range(1, 4):
                model_s4.train()
                perm = torch.randperm(num_races_tr)
                batch_size = 512

                for i in range(0, num_races_tr, batch_size):
                    idx = perm[i:i + batch_size]
                    bx = X_tr4_t[idx].to(device)
                    bmask = mask_tr_t[idx].to(device)
                    by = y_tr_t[idx].to(device)

                    optimizer.zero_grad()
                    logits = model_s4(bx, mask=bmask)
                    loss = criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_s4.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # Val 評価
                val_probs = predict_batches(model_s4, X_va4_t, mask_va_t, device)
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
                    best_model_state = {k: v.cpu() for k, v in model_s4.state_dict().items()}

            dt_t = time.time() - t_t0

            s4_val_roi = best_metrics["roi"]
            s4_val_hr = best_metrics["hit_rate"]
            s4_val_hits = best_metrics["hits"]
            s4_val_pay = best_metrics["total_payout"]
            s4_val_odds = s4_val_pay / (s4_val_hits * 100) if s4_val_hits > 0 else 0.0

            delta_roi = s4_val_roi - s3_val_roi
            delta_hr = s4_val_hr - s3_val_hr

            is_hit_rate_25 = bool(s4_val_hr >= 0.25)
            is_improved = bool((s4_val_roi > s3_val_roi) and (s4_val_hr > s3_val_hr))
            # 抽出フラグ: 的中率25%以上、またはチェーン元より両方向上
            is_extracted = bool(is_hit_rate_25 or (is_improved and s4_val_roi >= 1.0))

            quad_name = f"M{m_s1}->M{m_s2}->M{m_s3}->M{m_s4}"
            row = {
                "m_step1": m_s1,
                "m_step2": m_s2,
                "m_step3": m_s3,
                "m_step4": m_s4,
                "quad_name": quad_name,
                "s4_val_roi": s4_val_roi,
                "s4_val_hit_rate": s4_val_hr,
                "s4_val_hits": s4_val_hits,
                "s4_val_payout": s4_val_pay,
                "s4_val_mean_odds": s4_val_odds,
                "s3_val_roi": s3_val_roi,
                "s3_val_hit_rate": s3_val_hr,
                "delta_roi": delta_roi,
                "delta_hit_rate": delta_hr,
                "is_hit_rate_25": is_hit_rate_25,
                "is_improved": is_improved,
                "is_extracted": is_extracted,
                "time_sec": dt_t
            }

            completed_quads.add(quad_key)
            count_done += 1

            # リアルタイム追記
            pd.DataFrame([row]).to_csv(res_csv_path, mode="a", header=False, index=False)

            if is_extracted:
                extracted_quads.append(row)
                pd.DataFrame([row]).to_csv(ext_csv_path, mode="a", header=False, index=False)
                # モデル重み保存
                pth_save = os.path.join(artifacts_dir, f"found_M{m_s1}_M{m_s2}_M{m_s3}_to_M{m_s4}.pth")
                torch.save(best_model_state, pth_save)

                star_mark = "🎯【的中率25%超!】" if is_hit_rate_25 else "★【改善黒字!】"
                logger.info(
                    f"{star_mark} {quad_name:20s} => Val Hit: {s4_val_hr*100:4.1f}% ({delta_hr*100:+4.1f}pt, {s4_val_hits}勝) | "
                    f"Val ROI: {s4_val_roi*100:6.2f}% ({delta_roi*100:+5.2f}pt) | 累計抽出数: {len(extracted_quads)}"
                )

            del model_s4, optimizer, scheduler, X_tr4, X_va4, X_tr4_t, X_va4_t
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # 進捗ログ (50件ごと)
            if count_done % 50 == 0 or count_done == total_quads:
                elapsed = time.time() - t_start
                rate = count_done / elapsed if elapsed > 0 else 1
                eta_sec = (total_quads - count_done) / rate if rate > 0 else 0
                logger.info(
                    f"[Progress: {count_done:5d}/{total_quads} ({count_done/total_quads*100:5.1f}%)] "
                    f"Triplet {trip_idx+1}/{len(df_step3)} | 抽出数: {len(extracted_quads):4d}件 | "
                    f"経過: {elapsed/60:5.1f}分 | 残り予定: {eta_sec/60:5.1f}分 ({eta_sec/3600:4.1f}時間)"
                )

        del p3_tr, p3_va

    logger.info("==================================================================")
    logger.info(f"全 {total_quads} 通り走破完了！")
    logger.info(f"厳選抽出数: {len(extracted_quads)} クアッド")
    logger.info(f"結果保存先: {ext_csv_path}")
    logger.info("==================================================================")

if __name__ == "__main__":
    main()
