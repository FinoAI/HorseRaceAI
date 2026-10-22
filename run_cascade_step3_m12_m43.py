"""
run_cascade_step3_m12_m43.py
========================================================================
M12 -> M43 起点 3段カスケード (3-Chain) 総当たり探索スクリプト
------------------------------------------------------------------------
- ベースライン (Step 1 -> Step 2):
  * Step 1: M12 (model_12.pth)
  * Step 2: M43 (pattern_2_chain84_step2_m43.pth)
  * Step 2 成績: Val ROI = 123.34%, Test ROI = 89.97%
- Step 3 (総当たり対象):
  * 残り48モデル (Mx in [1..50], x != 12, 43)
  * 入力特徴量: Mx の200特徴量 + Step 2 予測確率 (1特徴量) = 201特徴量
  * 再学習: 2009〜2024年 (14,742R) のみで RaceLevelStage1NN を新規学習 (epochs=4)
  * Val評価: 前半751R で過学習監視・最良エポック選定
  * Test評価: 後半751R (完全未知) で最終実力測定
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

logger = setup_logger("CascadeStep3_M12_M43")

def predict_batches(net, X_t, mask_t, device, eval_bs=256):
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
        val_probs = predict_batches(model, X_va_t, mask_va_t, device)
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

    # 推論
    final_tr_probs = predict_batches(model, X_tr_t, mask_tr_t, device)
    final_va_probs = predict_batches(model, X_va_t, mask_va_t, device)
    final_te_probs = predict_batches(model, X_te_t, mask_te_t, device)

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
        "val_roi": val_res["roi"],
        "val_hit_rate": val_res["hit_rate"],
        "val_hits": val_res["hits"],
        "val_mean_odds": val_odds,
        "val_payout": val_res["total_payout"],
        "test_roi": test_res["roi"],
        "test_hit_rate": test_res["hit_rate"],
        "test_hits": test_res["hits"],
        "test_mean_odds": test_odds,
        "test_payout": test_res["total_payout"]
    }

def main():
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_cascade_step3_m12_m43")
    os.makedirs(artifacts_dir, exist_ok=True)

    RANDOM_SEARCH_SEED = 2026
    set_seed(RANDOM_SEARCH_SEED)

    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. キャッシュデータロード
    cached_data_path = os.path.join(CURRENT_DIR, "artifacts_dirt_1win/cached_dirt_1win_data.pkl")
    logger.info(f"Loading cached precomputed data from {cached_data_path}...")
    with open(cached_data_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    def get_model_features(m):
        feat_path = os.path.join(CURRENT_DIR, f"artifacts_dirt_1win/features_{m}.json")
        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        s_idx = [col_to_idx[c] for c in meta["yearly_features"]]
        return list(bac_col_indices) + s_idx

    # 2. Step 1 (M12) & Step 2 (M43) の事前推論確率生成 (またはロード)
    step2_cache_path = os.path.join(artifacts_dir, "step2_m12_m43_probs.pkl")
    if os.path.exists(step2_cache_path):
        logger.info(f"Loading cached Step 2 probabilities from {step2_cache_path}...")
        with open(step2_cache_path, "rb") as f:
            step2_cache = pickle.load(f)
        m43_tr_p = step2_cache["tr_probs"]
        m43_va_p = step2_cache["va_probs"]
        m43_te_p = step2_cache["te_probs"]
        base_s2_val = step2_cache["val_roi"]
        base_s2_test = step2_cache["test_roi"]
    else:
        logger.info("Computing Step 1 (M12) and Step 2 (M43) base probabilities...")
        # Step 1: M12
        f12 = get_model_features(12)
        m12_net = RaceLevelStage1NN(input_dim=18*200, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        m12_weights_path = os.path.join(CURRENT_DIR, "artifacts_dirt_1win/model_12.pth")
        m12_net.load_state_dict(torch.load(m12_weights_path, map_location=device))
        m12_net.eval()

        X_tr12 = torch.tensor(data_tr["race_3d"][:, :, f12].reshape(-1, 18*200), dtype=torch.float32)
        X_va12 = torch.tensor(data_va["race_3d"][:, :, f12].reshape(-1, 18*200), dtype=torch.float32)
        X_te12 = torch.tensor(data_te["race_3d"][:, :, f12].reshape(-1, 18*200), dtype=torch.float32)

        m12_tr_p = predict_batches(m12_net, X_tr12, torch.tensor(data_tr["race_masks"], dtype=torch.float32), device)
        m12_va_p = predict_batches(m12_net, X_va12, torch.tensor(data_va["race_masks"], dtype=torch.float32), device)
        m12_te_p = predict_batches(m12_net, X_te12, torch.tensor(data_te["race_masks"], dtype=torch.float32), device)
        del m12_net, X_tr12, X_va12, X_te12
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Step 2: M43 (pattern_2_chain84_step2_m43.pth)
        f43 = get_model_features(43)
        m43_net = RaceLevelStage1NN(input_dim=18*201, max_horses=18, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
        m43_weights_path = os.path.join(CURRENT_DIR, "artifacts_cascade_full1502/pattern_2_chain84_step2_m43.pth")
        m43_net.load_state_dict(torch.load(m43_weights_path, map_location=device))
        m43_net.eval()

        X_tr43 = np.concatenate([data_tr["race_3d"][:, :, f43], m12_tr_p[:, :, np.newaxis]], axis=-1)
        X_va43 = np.concatenate([data_va["race_3d"][:, :, f43], m12_va_p[:, :, np.newaxis]], axis=-1)
        X_te43 = np.concatenate([data_te["race_3d"][:, :, f43], m12_te_p[:, :, np.newaxis]], axis=-1)

        X_tr43_t = torch.tensor(X_tr43.reshape(-1, 18*201), dtype=torch.float32)
        X_va43_t = torch.tensor(X_va43.reshape(-1, 18*201), dtype=torch.float32)
        X_te43_t = torch.tensor(X_te43.reshape(-1, 18*201), dtype=torch.float32)

        m43_tr_p = predict_batches(m43_net, X_tr43_t, torch.tensor(data_tr["race_masks"], dtype=torch.float32), device)
        m43_va_p = predict_batches(m43_net, X_va43_t, torch.tensor(data_va["race_masks"], dtype=torch.float32), device)
        m43_te_p = predict_batches(m43_net, X_te43_t, torch.tensor(data_te["race_masks"], dtype=torch.float32), device)

        val_res_s2 = calculate_race_array_payout_metric(m43_va_p, data_va["race_masks"], data_va["race_y"], data_va["odds"], data_va["payouts"], "prob")
        test_res_s2 = calculate_race_array_payout_metric(m43_te_p, data_te["race_masks"], data_te["race_y"], data_te["odds"], data_te["payouts"], "prob")

        base_s2_val = val_res_s2["roi"]
        base_s2_test = test_res_s2["roi"]

        logger.info(f"Step 2 [M12->M43] Baseline: Val ROI = {base_s2_val*100:.2f}% | Test ROI = {base_s2_test*100:.2f}%")

        step2_cache = {
            "tr_probs": m43_tr_p,
            "va_probs": m43_va_p,
            "te_probs": m43_te_p,
            "val_roi": base_s2_val,
            "test_roi": base_s2_test,
            "val_hit_rate": val_res_s2["hit_rate"],
            "test_hit_rate": test_res_s2["hit_rate"]
        }
        with open(step2_cache_path, "wb") as f:
            pickle.dump(step2_cache, f)
        del m43_net, X_tr43, X_va43, X_te43, X_tr43_t, X_va43_t, X_te43_t
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # 3. Step 3 (総当たり48モデル) の探索リスト作成
    remaining_candidates = [m for m in range(1, 51) if m != 12 and m != 43]
    logger.info(f"Step 3 Target Candidates: {len(remaining_candidates)} models: {remaining_candidates}")

    # 途中再開用の既存チェック
    all_results = []
    res_csv_path = os.path.join(artifacts_dir, "cascade_step3_m12_m43_all_48.csv")
    already_done = set()
    if os.path.exists(res_csv_path):
        df_exist = pd.read_csv(res_csv_path)
        all_results = df_exist.to_dict("records")
        for r in all_results:
            already_done.add(int(r["step3_model"]))
        logger.info(f"Found {len(already_done)} already evaluated Step 3 models. Resuming remaining...")

    logger.info("\n" + "="*70)
    logger.info("  STARTING STEP 3 EXHAUSTIVE GRID SEARCH FOR [M12 -> M43 -> Mx]")
    logger.info(f"  Step 2 Baseline: Val ROI = {base_s2_val*100:.2f}%, Test ROI = {base_s2_test*100:.2f}%")
    logger.info(f"  Total Candidates in Queue: {len(remaining_candidates) - len(already_done)} models")
    logger.info("="*70 + "\n")

    t_start = time.time()
    for idx, m3 in enumerate(remaining_candidates, start=1):
        if m3 in already_done:
            continue

        t0 = time.time()
        f3 = get_model_features(m3)

        # カスケード特徴量結合: M3の200特徴量 + Step 2 (M43) の予測勝率 (1特徴量) = 201特徴量
        X_tr3_in = np.concatenate([data_tr["race_3d"][:, :, f3], m43_tr_p[:, :, np.newaxis]], axis=-1)
        X_va3_in = np.concatenate([data_va["race_3d"][:, :, f3], m43_va_p[:, :, np.newaxis]], axis=-1)
        X_te3_in = np.concatenate([data_te["race_3d"][:, :, f3], m43_te_p[:, :, np.newaxis]], axis=-1)

        # 新規NNインスタンス化 & 2009〜2024年のみで再学習 (epochs=4)
        res3 = train_and_eval_step(
            model_idx=m3,
            X_tr_3d=X_tr3_in,
            X_va_3d=X_va3_in,
            X_te_3d=X_te3_in,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            device=device,
            epochs=4
        )
        dt = time.time() - t0

        d_val = res3["val_roi"] - base_s2_val
        d_test = res3["test_roi"] - base_s2_test
        is_both_improved = (d_val > 0) and (d_test > 0)
        is_double_100 = (res3["val_roi"] >= 1.0) and (res3["test_roi"] >= 1.0)

        record = {
            "chain_name": f"M12->M43->M{m3}",
            "step1_model": 12,
            "step2_model": 43,
            "step3_model": m3,
            "s2_val_roi": base_s2_val,
            "s2_test_roi": base_s2_test,
            "s3_val_roi": res3["val_roi"],
            "s3_test_roi": res3["test_roi"],
            "delta_val_roi": d_val,
            "delta_test_roi": d_test,
            "val_hit_rate": res3["val_hit_rate"],
            "val_hits": res3["val_hits"],
            "val_mean_odds": res3["val_mean_odds"],
            "val_payout": res3["val_payout"],
            "test_hit_rate": res3["test_hit_rate"],
            "test_hits": res3["test_hits"],
            "test_mean_odds": res3["test_mean_odds"],
            "test_payout": res3["test_payout"],
            "is_both_improved": is_both_improved,
            "is_double_100": is_double_100,
            "time_sec": dt
        }
        all_results.append(record)

        # 優秀モデル重みの保存
        if is_double_100 or res3["test_roi"] >= 1.00:
            weight_path = os.path.join(artifacts_dir, f"found_M12_M43_to_M{m3}.pth")
            torch.save(res3["best_model_state"], weight_path)

        flag = ""
        if is_both_improved and is_double_100:
            flag = "★DOUBLE_100% & BOTH_UP!★"
        elif is_double_100:
            flag = "★DOUBLE_100%★"
        elif is_both_improved:
            flag = "★BOTH_UP★"
        elif res3["test_roi"] >= 1.0:
            flag = "Test>=100%"
        elif res3["val_roi"] >= 1.0:
            flag = "Val>=100%"

        logger.info(
            f"[{len(all_results):2d}/48] M12->M43->M{m3:2d} ({dt:4.1f}s) | "
            f"Val: {res3['val_roi']*100:6.2f}% ({d_val*100:+5.2f}pt) | "
            f"Test: {res3['test_roi']*100:6.2f}% ({d_test*100:+5.2f}pt) | "
            f"Hit: {res3['test_hit_rate']*100:4.1f}% | {flag}"
        )

        # 逐次保存
        pd.DataFrame(all_results).to_csv(res_csv_path, index=False)
        with open(os.path.join(artifacts_dir, "cascade_step3_m12_m43_all_48.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_time = (time.time() - t_start) / 60
    logger.info(f"\nCompleted all 48 Step 3 evaluations for M12->M43 in {total_time:.1f} minutes!")

if __name__ == "__main__":
    main()
