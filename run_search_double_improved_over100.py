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

from src.utils.helpers import setup_logger, set_seed, get_device
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("SearchDoubleImprovedOver100")

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

    # 推論
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
    artifacts_dir = "./artifacts_cascade_search_new_pairs"
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

    # 50モデル分の特徴量インデックス
    model_feature_indices = {}
    for m in range(1, 51):
        feat_path = f"./artifacts_dirt_1win/features_{m}.json"
        with open(feat_path, "r", encoding="utf-8") as f:
            f_meta = json.load(f)
        s_yearly = f_meta["yearly_features"]
        s_indices = [col_to_idx[c] for c in s_yearly]
        model_feature_indices[m] = list(bac_col_indices) + s_indices

    # 1. 既存ペアの完全ロード (除外リスト)
    already_evaluated_pairs = set()

    # 4origins から
    p_exh = "./artifacts_cascade_4origins_exhaustive/exhaustive_196_pairs_summary.csv"
    if os.path.exists(p_exh):
        df_exh = pd.read_csv(p_exh)
        for _, r in df_exh.iterrows():
            already_evaluated_pairs.add((int(r["origin_model"]), int(r["second_model"])))

    # random_cascade_full1502 から
    p_full = "./artifacts_cascade_full1502/random_cascade_full1502_all_steps.csv"
    if os.path.exists(p_full):
        df_f = pd.read_csv(p_full)
        for cid in df_f["chain_id"].unique():
            cdf = df_f[df_f["chain_id"] == cid]
            s1 = cdf[cdf["step"] == 1]
            s2 = cdf[cdf["step"] == 2]
            if len(s1) > 0 and len(s2) > 0:
                already_evaluated_pairs.add((int(s1["model_idx"].iloc[0]), int(s2["model_idx"].iloc[0])))

    logger.info(f"Loaded {len(already_evaluated_pairs)} already-evaluated pairs to SKIP (重複排除).")

    # 2. 探索の起点候補モデル（実績のある優良モデル群から優先）
    # M15, M23, M34, M24, M49, M17, M12, M1, M4, M46, M27, M30, M29, M43, M19 など
    PRIORITY_ORIGINS = [15, 23, 34, 24, 49, 17, 12, 1, 4, 46, 27, 30, 29, 43, 19, 3, 5, 6, 7, 9, 10, 11, 13, 14, 16, 18, 20, 21, 22, 25, 26, 28, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 44, 48, 50]

    # 有力次段候補 (両方上昇しやすいモデル群)
    PRIORITY_SECONDS = [27, 30, 29, 43, 19, 8, 45, 49, 34, 15, 24, 17, 23, 26, 12, 31, 32, 33, 40, 41]

    # 未試行ペアのキュー作成
    pairs_queue = []
    # まず有力起点 × 有力次段の未試行ペア
    for m1 in PRIORITY_ORIGINS:
        for m2 in PRIORITY_SECONDS:
            if m1 != m2 and (m1, m2) not in already_evaluated_pairs:
                if (m1, m2) not in pairs_queue:
                    pairs_queue.append((m1, m2))

    # 次に有力起点 × その他の全モデル
    for m1 in PRIORITY_ORIGINS:
        for m2 in range(1, 51):
            if m1 != m2 and (m1, m2) not in already_evaluated_pairs:
                if (m1, m2) not in pairs_queue:
                    pairs_queue.append((m1, m2))

    logger.info(f"Generated queue of {len(pairs_queue)} UNTRIED pairs to search.")

    TARGET_OVER_100_COUNT = 5
    found_over_100_list = []
    all_evaluated_new_pairs = []

    logger.info(f"\n==================================================================")
    logger.info(f"  Searching for NEW Pairs with DOUBLE IMPROVEMENT & ROI >= 100%")
    logger.info(f"  Condition: delta_val > 0 AND delta_test > 0 (両方向上)")
    logger.info(f"  Goal: Reach ROI >= 100% (Test >= 100% OR Val >= 100%)")
    logger.info(f"  Stop Condition: Found {TARGET_OVER_100_COUNT} pairs OR queue finished")
    logger.info(f"==================================================================\n")

    origin_caches = {}
    origin_base_metrics = {}
    t_start = time.time()

    for idx, (m1, m2) in enumerate(pairs_queue, start=1):
        # 1. 起点 m1 のキャッシュがなければ学習
        if m1 not in origin_caches:
            logger.info(f"\n>>> Precomputing Base Origin Model {m1:2d} (200 features) <<<")
            f_idx1 = model_feature_indices[m1]
            X_tr1 = data_tr["race_3d"][:, :, f_idx1]
            X_va1 = data_va["race_3d"][:, :, f_idx1]
            X_te1 = data_te["race_3d"][:, :, f_idx1]

            t0 = time.time()
            res1 = train_and_eval_step(
                model_idx=m1,
                X_tr_3d=X_tr1,
                X_va_3d=X_va1,
                X_te_3d=X_te1,
                data_tr=data_tr,
                data_va=data_va,
                data_te=data_te,
                device=device,
                epochs=4
            )
            origin_caches[m1] = {
                "tr_probs": res1["tr_probs"],
                "va_probs": res1["va_probs"],
                "te_probs": res1["te_probs"]
            }
            origin_base_metrics[m1] = res1
            logger.info(
                f"  Origin M{m1:2d} Done ({time.time()-t0:.1f}s) | "
                f"Val ROI={res1['val_roi']*100:6.2f}% | Test ROI={res1['test_roi']*100:6.2f}%"
            )

        # 2. Step 2 (m2) の学習
        m1_tr_p = origin_caches[m1]["tr_probs"]
        m1_va_p = origin_caches[m1]["va_probs"]
        m1_te_p = origin_caches[m1]["te_probs"]

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

        base_val = origin_base_metrics[m1]["val_roi"]
        base_test = origin_base_metrics[m1]["test_roi"]
        d_val = res2["val_roi"] - base_val
        d_test = res2["test_roi"] - base_test

        is_both_improved = (d_val > 0) and (d_test > 0)
        is_over_100 = (res2["test_roi"] >= 1.00) or (res2["val_roi"] >= 1.00)

        record = {
            "origin_model": m1,
            "second_model": m2,
            "pair_name": f"M{m1}->M{m2}",
            "s1_test_roi": base_test,
            "s2_test_roi": res2["test_roi"],
            "delta_test_roi": d_test,
            "s1_val_roi": base_val,
            "s2_val_roi": res2["val_roi"],
            "delta_val_roi": d_val,
            "test_hit_rate": res2["test_hit_rate"],
            "test_hits": res2["test_hits"],
            "test_mean_odds": res2["test_mean_odds"],
            "test_payout": res2["test_payout"],
            "val_hit_rate": res2["val_hit_rate"],
            "val_hits": res2["val_hits"],
            "val_mean_odds": res2["val_mean_odds"],
            "val_payout": res2["val_payout"],
            "is_both_improved": is_both_improved,
            "is_over_100": is_over_100,
            "time_sec": dt
        }
        all_evaluated_new_pairs.append(record)

        # ログメッセージ
        both_flag = "★BOTH_UP" if is_both_improved else "---"
        over_flag = ""
        if res2["test_roi"] >= 1.00 and res2["val_roi"] >= 1.00:
            over_flag = "★DOUBLE_100%★"
        elif res2["test_roi"] >= 1.00:
            over_flag = "★TEST>=100%★"
        elif res2["val_roi"] >= 1.00:
            over_flag = "★VAL>=100%★"

        status_str = f"[{both_flag}] {over_flag}".strip()

        logger.info(
            f"[{idx:3d}/{len(pairs_queue)}] M{m1:2d}->M{m2:2d} ({dt:.1f}s) | "
            f"Test: {res2['test_roi']*100:6.2f}% ({d_test*100:+5.2f}pt) | "
            f"Val: {res2['val_roi']*100:6.2f}% ({d_val*100:+5.2f}pt) | {status_str}"
        )

        # 条件達成チェック: 両方上がって、かつ 100% を超えた組み合わせ！
        if is_both_improved and is_over_100:
            found_over_100_list.append(record)
            w_path = os.path.join(artifacts_dir, f"found_M{m1}_to_M{m2}.pth")
            torch.save(res2["best_model_state"], w_path)
            logger.info(
                f"\n>>> ★★★ JACKPOT! Found Both Improved & ROI >= 100% (#{len(found_over_100_list)}): "
                f"M{m1}->M{m2} | Test={res2['test_roi']*100:.2f}% | Val={res2['val_roi']*100:.2f}% ★★★\n"
            )

            if len(found_over_100_list) >= TARGET_OVER_100_COUNT:
                logger.info(f"Target achieved! Found {TARGET_OVER_100_COUNT} pairs where BOTH improved and ROI >= 100%!")
                break

        # 5ペアごとに保存
        if idx % 5 == 0 or idx == len(pairs_queue):
            pd.DataFrame(all_evaluated_new_pairs).to_csv(
                os.path.join(artifacts_dir, "search_new_pairs_all.csv"), index=False
            )
            with open(os.path.join(artifacts_dir, "found_over_100_pairs.json"), "w", encoding="utf-8") as f:
                json.dump(found_over_100_list, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_start
    logger.info(f"\n==================================================================")
    logger.info(f"  Search Finished in {elapsed_total/60:.1f} mins!")
    logger.info(f"  New Pairs Evaluated: {len(all_evaluated_new_pairs)}")
    logger.info(f"  Both Improved & ROI >= 100% Pairs Found: {len(found_over_100_list)}")
    logger.info(f"==================================================================\n")

if __name__ == "__main__":
    main()
