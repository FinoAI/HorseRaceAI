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
from run_pipeline_dirt_1win import evaluate_subset_with_odds

logger = setup_logger("CascadeDirt1Win")

def train_cascade_step(
    step: int,
    model_idx: int,
    original_rank: int,
    X_tr_3d: np.ndarray,
    X_va_3d: np.ndarray,
    X_te_3d: np.ndarray,
    data_tr: dict,
    data_va: dict,
    data_te: dict,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    artifacts_dir: str = "./artifacts_cascade_dirt"
):
    num_horses = 18
    num_features = X_tr_3d.shape[2]
    input_dim = num_horses * num_features

    X_tr_flat = X_tr_3d.reshape(-1, input_dim)
    X_va_flat = X_va_3d.reshape(-1, input_dim)
    X_te_flat = X_te_3d.reshape(-1, input_dim)

    # CPU テンソルとして保持（MPS スライスクラッシュ対策）
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
    best_val_probs = None
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_races)
        epoch_loss = 0.0

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
            epoch_loss += loss.item()

        scheduler.step()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Val 評価
        val_probs = predict_batches(model, X_va_t, mask_va_t)
        metrics = calculate_race_array_payout_metric(
            probs=val_probs,
            masks=data_va["race_masks"],
            y_true=data_va["race_y"],
            odds=data_va["odds"],
            payouts=data_va["payouts"],
            selection_mode="prob"
        )
        hit_rate = metrics["hit_rate"]
        roi = metrics["roi"]
        val_score = roi * min(1.0, hit_rate / 0.20) + hit_rate

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_probs = val_probs
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 最良モデルを復元
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # モデル保存
    model_save_path = os.path.join(artifacts_dir, f"cascade_model_step_{step}_m{model_idx}.pth")
    torch.save(best_model_state, model_save_path)

    # Train, Val, Test に対する全推論確率を出力（次ステップへの引き渡し用）
    final_tr_probs = predict_batches(model, X_tr_t, mask_tr_t)
    final_va_probs = predict_batches(model, X_va_t, mask_va_t)
    final_te_probs = predict_batches(model, X_te_t, mask_te_t)

    # 評価
    test_res = evaluate_subset_with_odds([0], final_te_probs[np.newaxis, ...], data_te)
    val_res = evaluate_subset_with_odds([0], final_va_probs[np.newaxis, ...], data_va)

    del model, optimizer, scheduler, X_tr_t, mask_tr_t, y_tr_t, X_va_t, mask_va_t, X_te_t, mask_te_t
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "step": step,
        "model_idx": model_idx,
        "original_rank": original_rank,
        "num_features": num_features,
        "tr_probs": final_tr_probs,
        "va_probs": final_va_probs,
        "te_probs": final_te_probs,
        "test_hit_rate": test_res["hit_rate"],
        "test_hits": test_res["hits"],
        "test_mean_odds": test_res["mean_hit_odds"],
        "test_roi": test_res["roi"],
        "test_payout": test_res["total_payout"],
        "val_hit_rate": val_res["hit_rate"],
        "val_roi": val_res["roi"],
        "val_mean_odds": val_res["mean_hit_odds"],
        "val_score": best_val_score
    }

def main():
    artifacts_dir = "./artifacts_cascade_dirt"
    os.makedirs(artifacts_dir, exist_ok=True)

    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. キャッシュ済みデータのロード
    cached_data_path = "./artifacts_dirt_1win/cached_dirt_1win_data.pkl"
    logger.info(f"Loading cached precomputed data from {cached_data_path}...")
    with open(cached_data_path, "rb") as f:
        data_dict = pickle.load(f)
    logger.info("Successfully loaded precomputed dirt dataset!")

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    # 2. 単体モデルランキング取得
    perf_path = "./artifacts_dirt_1win/single_models_performance.csv"
    df_perf = pd.read_csv(perf_path)
    # Val 総合スコア順にソート (1位〜50位)
    df_ranked = df_perf.sort_values("val_score", ascending=False).reset_index(drop=True)
    
    # Top 25 モデルの選出
    top_25_models = df_ranked.iloc[:25].copy()
    top_25_models["rank"] = np.arange(1, 26)

    # ユーザー指示: 「最初は２５位からだんだん使うモデルの順位を上げてください」
    # 25位 -> 24位 -> ... -> 1位 へと逆順に並び替え
    execution_order = top_25_models.iloc[::-1].reset_index(drop=True)

    logger.info(f"\n==================================================================")
    logger.info(f"  Starting Sequential Cascade Stacking Pipeline (Top 25 Models)")
    logger.info(f"  Order: 25th -> 24th -> ... -> 1st Place Model")
    logger.info(f"  Step 1: Rank 25 (Model {execution_order.iloc[0]['model_idx']})")
    logger.info(f"  Step 25: Rank 1 (Model {execution_order.iloc[-1]['model_idx']})")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races)")
    logger.info(f"  Val: 2025-2026 ({len(data_va['race_y']):,d} races)")
    logger.info(f"  Test: 2025-2026 ({len(data_te['race_y']):,d} races)")
    logger.info(f"==================================================================\n")

    steps_results = []
    prev_tr_prob = None
    prev_va_prob = None
    prev_te_prob = None

    all_step_te_probs = []
    t_start = time.time()

    for step_num in range(1, 26):
        row = execution_order.iloc[step_num - 1]
        m_idx = int(row["model_idx"])
        orig_rank = int(row["rank"])

        # 当該モデルの JSON からベース 200 特徴量をロード
        feat_json_path = f"./artifacts_dirt_1win/features_{m_idx}.json"
        with open(feat_json_path, "r", encoding="utf-8") as f:
            feat_meta = json.load(f)

        sampled_yearly = feat_meta["yearly_features"]
        sampled_yearly_indices = [col_to_idx[c] for c in sampled_yearly]
        selected_indices = list(bac_col_indices) + sampled_yearly_indices # 200列

        # ベース 3D テンソル (N, 18, 200)
        X_tr_base = data_tr["race_3d"][:, :, selected_indices]
        X_va_base = data_va["race_3d"][:, :, selected_indices]
        X_te_base = data_te["race_3d"][:, :, selected_indices]

        # 前段モデルの予測確率がある場合、新たな特徴量として結合
        if prev_tr_prob is not None:
            # (N, 18, 1) として拡張
            tr_extra = prev_tr_prob[:, :, np.newaxis]
            va_extra = prev_va_prob[:, :, np.newaxis]
            te_extra = prev_te_prob[:, :, np.newaxis]

            X_tr_in = np.concatenate([X_tr_base, tr_extra], axis=-1) # (N, 18, 201)
            X_va_in = np.concatenate([X_va_base, va_extra], axis=-1)
            X_te_in = np.concatenate([X_te_base, te_extra], axis=-1)
            feat_note = f"Base 200 + Prev Prob (201 feats)"
        else:
            X_tr_in = X_tr_base # (N, 18, 200)
            X_va_in = X_va_base
            X_te_in = X_te_base
            feat_note = f"Base 200 feats (Initial Step)"

        logger.info(
            f">>> [Step {step_num:2d}/25] Training Model {m_idx:2d} (Original Rank: #{orig_rank:2d}) | {feat_note} <<<"
        )
        t0 = time.time()
        res = train_cascade_step(
            step=step_num,
            model_idx=m_idx,
            original_rank=orig_rank,
            X_tr_3d=X_tr_in,
            X_va_3d=X_va_in,
            X_te_3d=X_te_in,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            device=device,
            epochs=5,
            batch_size=512,
            lr=0.001,
            weight_decay=0.0001,
            artifacts_dir=artifacts_dir
        )
        dt = time.time() - t0

        # 次ステップ用に出力を保持
        prev_tr_prob = res["tr_probs"]
        prev_va_prob = res["va_probs"]
        prev_te_prob = res["te_probs"]
        all_step_te_probs.append(res["te_probs"])

        # 単体時の元々の成績を取得して比較
        orig_test_hit = row["test_hit_rate"]
        orig_test_roi = row["test_roi"]

        steps_results.append({
            "step": step_num,
            "rank": orig_rank,
            "model_idx": m_idx,
            "num_features": res["num_features"],
            "cascade_test_hit": res["test_hit_rate"],
            "cascade_test_hits": res["test_hits"],
            "cascade_test_odds": res["test_mean_odds"],
            "cascade_test_roi": res["test_roi"],
            "cascade_test_payout": res["test_payout"],
            "orig_test_hit": orig_test_hit,
            "orig_test_roi": orig_test_roi,
            "val_hit_rate": res["val_hit_rate"],
            "val_roi": res["val_roi"],
            "time_sec": dt
        })

        logger.info(
            f"    Result Step {step_num:2d} (Rank #{orig_rank:2d}, M{m_idx:2d}): "
            f"Test Hit={res['test_hit_rate']*100:.2f}% ({res['test_hits']}/751R) | "
            f"Test ROI={res['test_roi']*100:.2f}% | Mean Odds={res['test_mean_odds']:.2f}x | "
            f"[Orig Hit: {orig_test_hit*100:.2f}%, Orig ROI: {orig_test_roi*100:.2f}%] ({dt:.1f}s)"
        )

    # 結果保存
    df_steps = pd.DataFrame(steps_results)
    csv_out = os.path.join(artifacts_dir, "cascade_steps_summary.csv")
    df_steps.to_csv(csv_out, index=False)
    np.save(os.path.join(artifacts_dir, "cascade_test_probs.npy"), np.array(all_step_te_probs))

    # これまでのステップの累積アンサンブルも評価
    cum_results = []
    for s in range(1, 26):
        sub_probs = np.array(all_step_te_probs[:s]) # (s, 751, 18)
        ens_p = np.mean(sub_probs, axis=0)
        eval_res = evaluate_subset_with_odds([0], ens_p[np.newaxis, ...], data_te)
        cum_results.append({
            "step": s,
            "cum_models": s,
            "cum_test_hit": eval_res["hit_rate"],
            "cum_test_hits": eval_res["hits"],
            "cum_test_odds": eval_res["mean_hit_odds"],
            "cum_test_roi": eval_res["roi"],
            "cum_test_payout": eval_res["total_payout"]
        })
    df_cum = pd.DataFrame(cum_results)
    df_cum.to_csv(os.path.join(artifacts_dir, "cascade_cumulative_ensemble.csv"), index=False)

    print("\n" + "="*125)
    print("      ★ 逐次カスケード・スタッキング（25位 -> 1位）各ステップ推移結果 ★")
    print("="*125)
    formatters = {
        "cascade_test_hit": lambda x: f"{x*100:5.2f}%",
        "cascade_test_roi": lambda x: f"{x*100:6.2f}%",
        "cascade_test_odds": lambda x: f"{x:5.2f}倍",
        "cascade_test_payout": lambda x: f"¥{x:,.0f}",
        "orig_test_hit": lambda x: f"{x*100:5.2f}%",
        "orig_test_roi": lambda x: f"{x*100:6.2f}%",
    }
    disp_df = df_steps[["step", "rank", "model_idx", "cascade_test_hit", "cascade_test_hits", "cascade_test_odds", "cascade_test_roi", "cascade_test_payout", "orig_test_hit", "orig_test_roi"]].copy()
    for col, fmt in formatters.items():
        if col in disp_df.columns:
            disp_df[col] = disp_df[col].apply(fmt)
    print(disp_df.to_string(index=False))
    print("="*125)

if __name__ == "__main__":
    main()
