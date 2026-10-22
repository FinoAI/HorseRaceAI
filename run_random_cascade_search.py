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

logger = setup_logger("RandomCascadeSearch")

def train_and_eval_step(
    step: int,
    model_idx: int,
    X_tr_3d: np.ndarray,
    X_va_3d: np.ndarray,
    X_te_3d: np.ndarray,
    data_tr: dict,
    data_va: dict,
    data_te: dict,
    device: torch.device,
    epochs: int = 4, # 高速探索のため4エポック
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
        val_score = metrics["roi"] * min(1.0, metrics["hit_rate"] / 0.20) + metrics["hit_rate"]

        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 最良モデル復元
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Train, Val, Test に対する全推論確率を出力
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
        "val_mean_odds": val_res["mean_hit_odds"]
    }

def main():
    artifacts_dir = "./artifacts_cascade_random"
    os.makedirs(artifacts_dir, exist_ok=True)

    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    RANDOM_SEARCH_SEED = 2026
    set_seed(RANDOM_SEARCH_SEED)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)

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

    MAX_CHAINS = 100
    TARGET_ROI_OVER_100_COUNT = 5

    logger.info(f"\n==================================================================")
    logger.info(f"  Random Cascade Stacking Search")
    logger.info(f"  Condition: Random model order up to 50 models")
    logger.info(f"  Stop Condition: Stop chain immediately when ROI drops")
    logger.info(f"  Goal: 100 Chains OR 5 Patterns with ROI >= 100%")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races)")
    logger.info(f"  Test: 2025-2026 ({len(data_te['race_y']):,d} races)")
    logger.info(f"==================================================================\n")

    all_chains_history = []
    roi_over_100_hits = []
    total_models_trained = 0
    t_start = time.time()

    for chain_idx in range(1, MAX_CHAINS + 1):
        # 1〜50 のモデル順序をランダムシャッフル
        shuffled_models = rng.permutation(np.arange(1, 51)).tolist()

        logger.info(f"\n--- [Chain {chain_idx:3d}/{MAX_CHAINS}] Starting Random Cascade Chain ---")
        chain_steps = []
        prev_tr_p = None
        prev_va_p = None
        prev_te_p = None
        prev_roi = -1.0

        for step in range(1, 51):
            m_idx = shuffled_models[step - 1]
            feat_idx = model_feature_indices[m_idx]

            # ベーステンソル
            X_tr_b = data_tr["race_3d"][:, :, feat_idx]
            X_va_b = data_va["race_3d"][:, :, feat_idx]
            X_te_b = data_te["race_3d"][:, :, feat_idx]

            if prev_tr_p is not None:
                X_tr_in = np.concatenate([X_tr_b, prev_tr_p[:, :, np.newaxis]], axis=-1)
                X_va_in = np.concatenate([X_va_b, prev_va_p[:, :, np.newaxis]], axis=-1)
                X_te_in = np.concatenate([X_te_b, prev_te_p[:, :, np.newaxis]], axis=-1)
            else:
                X_tr_in = X_tr_b
                X_va_in = X_va_b
                X_te_in = X_te_b

            t0 = time.time()
            res = train_and_eval_step(
                step=step,
                model_idx=m_idx,
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
            total_models_trained += 1

            curr_roi = res["test_roi"]
            chain_steps.append({
                "chain_id": chain_idx,
                "step": step,
                "model_idx": m_idx,
                "test_roi": curr_roi,
                "test_hit_rate": res["test_hit_rate"],
                "test_hits": res["test_hits"],
                "test_mean_odds": res["test_mean_odds"],
                "test_payout": res["test_payout"],
                "val_roi": res["val_roi"],
                "val_hit_rate": res["val_hit_rate"]
            })

            logger.info(
                f"  Chain {chain_idx:3d} | Step {step:2d} (Model {m_idx:2d}): "
                f"Test ROI={curr_roi*100:6.2f}% | Hit={res['test_hit_rate']*100:5.2f}% ({res['test_hits']}R) | "
                f"Odds={res['test_mean_odds']:4.2f}x | (Prev ROI={prev_roi*100:5.2f}%) ({dt:.1f}s)"
            )

            # ROI 100% 超えのチェック
            if curr_roi >= 1.00:
                pattern_info = {
                    "chain_id": chain_idx,
                    "step": step,
                    "chain_models": [cs["model_idx"] for cs in chain_steps],
                    "final_model_idx": m_idx,
                    "test_roi": curr_roi,
                    "test_hit_rate": res["test_hit_rate"],
                    "test_hits": res["test_hits"],
                    "test_mean_odds": res["test_mean_odds"],
                    "test_payout": res["test_payout"],
                    "val_roi": res["val_roi"],
                    "val_hit_rate": res["val_hit_rate"]
                }
                roi_over_100_hits.append(pattern_info)
                logger.info(f"  >>> ★ FOUND ROI >= 100% pattern (#{len(roi_over_100_hits)}): ROI={curr_roi*100:.2f}%! <<<")

                # 目標件数 (5件) 達成チェック
                if len(roi_over_100_hits) >= TARGET_ROI_OVER_100_COUNT:
                    logger.info(f"\nTarget achieved! Found {TARGET_ROI_OVER_100_COUNT} patterns with ROI >= 100%!")
                    break

            # ユーザー指示: 「また下がったらそこでやめてください」
            if step > 1 and curr_roi < prev_roi:
                logger.info(f"  --> ROI dropped from {prev_roi*100:.2f}% to {curr_roi*100:.2f}%. Stopping Chain {chain_idx} at Step {step}.")
                break

            prev_roi = curr_roi
            prev_tr_p = res["tr_probs"]
            prev_va_p = res["va_probs"]
            prev_te_p = res["te_probs"]

        all_chains_history.extend(chain_steps)

        # 保存 (各チェーン終了ごとに最新状態を追記保存)
        pd.DataFrame(all_chains_history).to_csv(os.path.join(artifacts_dir, "random_cascade_all_steps.csv"), index=False)
        with open(os.path.join(artifacts_dir, "roi_over_100_cascade_patterns.json"), "w", encoding="utf-8") as f:
            json.dump(roi_over_100_hits, f, ensure_ascii=False, indent=2)

        if len(roi_over_100_hits) >= TARGET_ROI_OVER_100_COUNT:
            break

    elapsed_total = time.time() - t_start
    logger.info(f"\n==================================================================")
    logger.info(f"  Search Finished in {elapsed_total/60:.1f} mins!")
    logger.info(f"  Total Chains: {chain_idx} | Total Models Trained: {total_models_trained}")
    logger.info(f"  ROI >= 100% Patterns Found: {len(roi_over_100_hits)}")
    logger.info(f"==================================================================\n")

    # サマリー表示
    if len(roi_over_100_hits) > 0:
        df_hits = pd.DataFrame(roi_over_100_hits)
        print("\n" + "="*125)
        print(f"      ★ 発掘された ROI 100% 超えカスケードパターン (全 {len(df_hits)} 件) ★")
        print("="*125)
        formatters = {
            "test_roi": lambda x: f"{x*100:6.2f}%",
            "test_hit_rate": lambda x: f"{x*100:5.2f}%",
            "test_mean_odds": lambda x: f"{x:5.2f}倍",
            "test_payout": lambda x: f"¥{x:,.0f}",
            "val_roi": lambda x: f"{x*100:6.2f}%",
            "val_hit_rate": lambda x: f"{x*100:5.2f}%",
        }
        disp_df = df_hits[["chain_id", "step", "final_model_idx", "test_roi", "test_hit_rate", "test_hits", "test_mean_odds", "test_payout", "val_roi", "chain_models"]].copy()
        for col, fmt in formatters.items():
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(fmt)
        print(disp_df.to_string(index=False))
        print("="*125)

if __name__ == "__main__":
    main()
