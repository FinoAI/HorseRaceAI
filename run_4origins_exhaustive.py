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

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("Exhaustive4Origins")

def train_and_eval_step(
    model_idx: int,
    X_tr_3d: np.ndarray,
    X_va_3d: np.ndarray,
    X_te_3d: np.ndarray,
    data_tr: dict,
    data_va: dict,
    data_te: dict,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0001
):
    num_horses = 18
    num_features = X_tr_3d.shape[2]
    input_dim = num_horses * num_features

    X_tr_flat = X_tr_3d.reshape(-1, input_dim)
    X_va_flat = X_va_3d.reshape(-1, input_dim)
    X_te_flat = X_te_3d.reshape(-1, input_dim)

    # CPU テンソル保持 (MPS スライスクラッシュ対策)
    X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32)
    mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr_t = torch.tensor(data_tr["race_y"], dtype=torch.int64)

    X_va_t = torch.tensor(X_va_flat, dtype=torch.float32)
    mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    X_te_t = torch.tensor(X_te_flat, dtype=torch.float32)
    mask_te_t = torch.tensor(data_te["race_masks"], dtype=torch.float32)

    model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=num_horses,
        multiplier=2,
        reduction_ratio=0.5,
        dropout=0.15
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    num_races = len(X_tr_t)

    def predict_batches(net, X_t, mask_t):
        net.eval()
        probs_list = []
        n = len(X_t)
        eval_bs = 256
        with torch.no_grad():
            for i in range(0, n, eval_bs):
                bx = X_t[i:i+eval_bs].to(device)
                bmask = mask_t[i:i+eval_bs].to(device)
                logits = net(bx, mask=bmask)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)

    best_val_score = -1.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_races)

        for i in range(0, num_races, batch_size):
            idx = perm[i:i + batch_size]
            bx = X_tr_t[idx].to(device)
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

        # Val 評価 (エポック選定用)
        val_probs = predict_batches(model, X_va_t, mask_va_t)
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
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 最良モデル復元
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Train, Val, Test に対する推論
    final_tr_probs = predict_batches(model, X_tr_t, mask_tr_t)
    final_va_probs = predict_batches(model, X_va_t, mask_va_t)
    final_te_probs = predict_batches(model, X_te_t, mask_te_t)

    # 1. 検証データ (Val: 前半751R) 成績
    val_res = calculate_race_array_payout_metric(
        probs=final_va_probs,
        masks=data_va["race_masks"],
        y_true=data_va["race_y"],
        odds=data_va["odds"],
        payouts=data_va["payouts"],
        selection_mode="prob"
    )
    val_odds = val_res["total_payout"] / (val_res["hits"] * 100) if val_res["hits"] > 0 else 0

    # 2. 完全未知テストデータ (Test: 後半751R) 成績
    test_res = calculate_race_array_payout_metric(
        probs=final_te_probs,
        masks=data_te["race_masks"],
        y_true=data_te["race_y"],
        odds=data_te["odds"],
        payouts=data_te["payouts"],
        selection_mode="prob"
    )
    test_odds = test_res["total_payout"] / (test_res["hits"] * 100) if test_res["hits"] > 0 else 0

    del model, optimizer, scheduler, X_tr_t, mask_tr_t, y_tr_t, X_va_t, mask_va_t, X_te_t, mask_te_t
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_idx": model_idx,
        "num_features": num_features,
        "best_model_state": best_model_state,
        "tr_probs": final_tr_probs,
        "va_probs": final_va_probs,
        "te_probs": final_te_probs,
        # 検証 (Val: 前半751R)
        "val_roi": val_res["roi"],
        "val_hit_rate": val_res["hit_rate"],
        "val_hits": val_res["hits"],
        "val_mean_odds": val_odds,
        "val_payout": val_res["total_payout"],
        # テスト (Test: 後半751R)
        "test_roi": test_res["roi"],
        "test_hit_rate": test_res["hit_rate"],
        "test_hits": test_res["hits"],
        "test_mean_odds": test_odds,
        "test_payout": test_res["total_payout"]
    }

def main():
    artifacts_dir = "./artifacts_cascade_4origins_exhaustive"
    os.makedirs(artifacts_dir, exist_ok=True)

    RANDOM_SEARCH_SEED = 2026
    set_seed(RANDOM_SEARCH_SEED)

    device = get_device()
    logger.info(f"Compute Device: {device}")

    # キャッシュデータロード
    cached_data_path = "./artifacts_dirt_1win/cached_dirt_1win_data.pkl"
    logger.info(f"Loading cached precomputed data from {cached_data_path}...")
    with open(cached_data_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    # 50モデル分の特徴量インデックスを事前パース
    model_feature_indices = {}
    for m in range(1, 51):
        feat_path = f"./artifacts_dirt_1win/features_{m}.json"
        with open(feat_path, "r", encoding="utf-8") as f:
            f_meta = json.load(f)
        s_yearly = f_meta["yearly_features"]
        s_indices = [col_to_idx[c] for c in s_yearly]
        model_feature_indices[m] = list(bac_col_indices) + s_indices

    TARGET_STARTS = [45, 2, 8, 47]

    # すでに直前までに試行済みのペアを確認
    # artifacts_cascade_unstarted から既試行ペアを読み込み
    prev_csv_path = "./artifacts_cascade_unstarted/unstarted_origins_cascade_all_steps.csv"
    existing_pairs_data = {}
    if os.path.exists(prev_csv_path):
        prev_df = pd.read_csv(prev_csv_path)
        for cid in prev_df["chain_id"].unique():
            cdf = prev_df[prev_df["chain_id"] == cid]
            if len(cdf) >= 2:
                s1 = cdf[cdf["step"] == 1]
                s2 = cdf[cdf["step"] == 2]
                if len(s1) > 0 and len(s2) > 0:
                    m1 = int(s1["model_idx"].iloc[0])
                    m2 = int(s2["model_idx"].iloc[0])
                    if m1 in TARGET_STARTS:
                        existing_pairs_data[(m1, m2)] = {
                            "origin_model": m1,
                            "second_model": m2,
                            "num_features": int(s2["num_features"].iloc[0]),
                            "test_roi": float(s2["test_roi"].iloc[0]),
                            "test_hit_rate": float(s2["test_hit_rate"].iloc[0]),
                            "test_hits": int(s2["test_hits"].iloc[0]),
                            "test_mean_odds": float(s2["test_mean_odds"].iloc[0]),
                            "test_payout": float(s2["test_payout"].iloc[0]),
                            "val_roi": float(s2["val_roi"].iloc[0]),
                            "val_hit_rate": float(s2["val_hit_rate"].iloc[0]),
                            "val_hits": int(s2["val_hits"].iloc[0]),
                            "val_mean_odds": float(s2["val_mean_odds"].iloc[0]),
                            "val_payout": float(s2["val_payout"].iloc[0]),
                            "source": "pre_executed"
                        }

    logger.info(f"Loaded {len(existing_pairs_data)} pre-executed pairs from previous search.")

    # 全 196 ペアのリストアップ
    all_target_pairs = []
    for m1 in TARGET_STARTS:
        for m2 in range(1, 51):
            if m1 != m2:
                all_target_pairs.append((m1, m2))

    pairs_to_run = [p for p in all_target_pairs if p not in existing_pairs_data]
    logger.info(f"Total Target Pairs: {len(all_target_pairs)} | Already Executed: {len(existing_pairs_data)} | To Execute: {len(pairs_to_run)}")

    logger.info(f"\n==================================================================")
    logger.info(f"  Exhaustive Pair Exploration for Origins: {TARGET_STARTS}")
    logger.info(f"  Total Pairs to Evaluate: {len(pairs_to_run)} pairs")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races)")
    logger.info(f"  Val:   2025-2026 Shuffled Half ({len(data_va['race_y']):,d} races)")
    logger.info(f"  Test:  2025-2026 Shuffled Half ({len(data_te['race_y']):,d} races)")
    logger.info(f"==================================================================\n")

    # 1. 起点4モデル (Step 1) の学習＆予測キャッシュ作成
    logger.info(">>> Step 1: Precomputing and Caching Predictions for 4 Origin Models <<<")
    origin_caches = {}
    origin_base_metrics = {}

    for m1 in TARGET_STARTS:
        logger.info(f"Training Base Origin Model {m1:2d} (200 features)...")
        f_idx = model_feature_indices[m1]
        X_tr = data_tr["race_3d"][:, :, f_idx]
        X_va = data_va["race_3d"][:, :, f_idx]
        X_te = data_te["race_3d"][:, :, f_idx]

        t0 = time.time()
        res1 = train_and_eval_step(
            model_idx=m1,
            X_tr_3d=X_tr,
            X_va_3d=X_va,
            X_te_3d=X_te,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            device=device,
            epochs=4
        )
        dt = time.time() - t0
        origin_caches[m1] = {
            "tr_probs": res1["tr_probs"],
            "va_probs": res1["va_probs"],
            "te_probs": res1["te_probs"]
        }
        origin_base_metrics[m1] = res1
        logger.info(
            f"  Origin M{m1:2d} Done ({dt:.1f}s) | Val ROI={res1['val_roi']*100:6.2f}% | "
            f"Test ROI={res1['test_roi']*100:6.2f}% (Hit={res1['test_hit_rate']*100:5.2f}%, {res1['test_hits']}R, {res1['test_mean_odds']:4.2f}x)"
        )

    # 2. Step 2 の網羅的学習
    results_list = list(existing_pairs_data.values())
    val_over_100_list = []
    test_over_100_list = []

    t_start = time.time()
    total_to_run = len(pairs_to_run)

    for i, (m1, m2) in enumerate(pairs_to_run, start=1):
        # m1 の予測確率を取得
        m1_tr_p = origin_caches[m1]["tr_probs"]
        m1_va_p = origin_caches[m1]["va_probs"]
        m1_te_p = origin_caches[m1]["te_probs"]

        # m2 のベース特徴量 + m1予測確率 (201特徴量)
        f_idx2 = model_feature_indices[m2]
        X_tr_b = data_tr["race_3d"][:, :, f_idx2]
        X_va_b = data_va["race_3d"][:, :, f_idx2]
        X_te_b = data_te["race_3d"][:, :, f_idx2]

        X_tr_in = np.concatenate([X_tr_b, m1_tr_p[:, :, np.newaxis]], axis=-1)
        X_va_in = np.concatenate([X_va_b, m1_va_p[:, :, np.newaxis]], axis=-1)
        X_te_in = np.concatenate([X_te_b, m1_te_p[:, :, np.newaxis]], axis=-1)

        t0 = time.time()
        res2 = train_and_eval_step(
            model_idx=m2,
            X_tr_3d=X_tr_in,
            X_va_3d=X_va_in,
            X_te_3d=X_te_in,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            device=device,
            epochs=4
        )
        dt = time.time() - t0

        elapsed_so_far = time.time() - t_start
        avg_time = elapsed_so_far / i
        remaining_secs = avg_time * (total_to_run - i)

        base_val_roi = origin_base_metrics[m1]["val_roi"]
        base_test_roi = origin_base_metrics[m1]["test_roi"]
        delta_val = res2["val_roi"] - base_val_roi
        delta_test = res2["test_roi"] - base_test_roi

        record = {
            "origin_model": m1,
            "second_model": m2,
            "pair_name": f"M{m1}->M{m2}",
            "num_features": res2["num_features"],
            # テスト成績 (Test: 後半751R)
            "test_roi": res2["test_roi"],
            "test_hit_rate": res2["test_hit_rate"],
            "test_hits": res2["test_hits"],
            "test_mean_odds": res2["test_mean_odds"],
            "test_payout": res2["test_payout"],
            "delta_test_roi": delta_test,
            # 検証成績 (Val: 前半751R)
            "val_roi": res2["val_roi"],
            "val_hit_rate": res2["val_hit_rate"],
            "val_hits": res2["val_hits"],
            "val_mean_odds": res2["val_mean_odds"],
            "val_payout": res2["val_payout"],
            "delta_val_roi": delta_val,
            "time_sec": dt,
            "source": "new_executed"
        }
        results_list.append(record)

        # ログ表示
        val_flag = "★VAL>=100%" if res2["val_roi"] >= 1.00 else ""
        test_flag = "★TEST>=100%" if res2["test_roi"] >= 1.00 else ""
        flags = f"{val_flag} {test_flag}".strip()
        flags_str = f" | >>> {flags} <<<" if flags else ""

        logger.info(
            f"[{i:3d}/{total_to_run}] ({(i/total_to_run)*100:5.1f}%) M{m1:2d}->M{m2:2d}: "
            f"Test ROI={res2['test_roi']*100:6.2f}% ({delta_test*100:+5.2f}pt, {res2['test_hits']}R, {res2['test_mean_odds']:4.2f}x) | "
            f"Val ROI={res2['val_roi']*100:6.2f}% ({delta_val*100:+5.2f}pt) | "
            f"ETA: {remaining_secs/60:4.1f}m{flags_str}"
        )

        # 100% 超えモデルの保存
        if res2["val_roi"] >= 1.00:
            val_over_100_list.append(record)
            w_path = os.path.join(artifacts_dir, f"val100_M{m1}_to_M{m2}.pth")
            torch.save(res2["best_model_state"], w_path)

        if res2["test_roi"] >= 1.00:
            test_over_100_list.append(record)
            w_path = os.path.join(artifacts_dir, f"test100_M{m1}_to_M{m2}.pth")
            torch.save(res2["best_model_state"], w_path)

        # 5ペアごとに中間保存
        if i % 5 == 0 or i == total_to_run:
            df_out = pd.DataFrame(results_list)
            df_out.to_csv(os.path.join(artifacts_dir, "exhaustive_196_pairs_summary.csv"), index=False)
            with open(os.path.join(artifacts_dir, "val_over_100_pairs.json"), "w", encoding="utf-8") as f:
                json.dump(val_over_100_list, f, ensure_ascii=False, indent=2)
            with open(os.path.join(artifacts_dir, "test_over_100_pairs.json"), "w", encoding="utf-8") as f:
                json.dump(test_over_100_list, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    logger.info(f"\n==================================================================")
    logger.info(f"  Exhaustive Pair Search Finished in {elapsed_total/60:.1f} mins!")
    logger.info(f"  Total Pairs in Dataset: {len(results_list)} / {len(all_target_pairs)}")
    logger.info(f"  Val ROI >= 100% Pairs: {len(val_over_100_list)}")
    logger.info(f"  Test ROI >= 100% Pairs: {len(test_over_100_list)}")
    logger.info(f"==================================================================\n")

if __name__ == "__main__":
    main()
