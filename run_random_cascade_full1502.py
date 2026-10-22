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

logger = setup_logger("RandomCascadeFull1502")

def train_and_eval_step(
    step: int,
    model_idx: int,
    X_tr_3d: np.ndarray,
    X_va_3d: np.ndarray,
    X_full_3d: np.ndarray,
    data_tr: dict,
    data_va: dict,
    data_full: dict,
    device: torch.device,
    epochs: int = 4, # 高速探索用
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0001
):
    num_horses = 18
    num_features = X_tr_3d.shape[2]
    input_dim = num_horses * num_features

    X_tr_flat = X_tr_3d.reshape(-1, input_dim)
    X_va_flat = X_va_3d.reshape(-1, input_dim)
    X_full_flat = X_full_3d.reshape(-1, input_dim)

    # CPU テンソル保持 (MPS スライスクラッシュ対策)
    X_tr_t = torch.tensor(X_tr_flat, dtype=torch.float32)
    mask_tr_t = torch.tensor(data_tr["race_masks"], dtype=torch.float32)
    y_tr_t = torch.tensor(data_tr["race_y"], dtype=torch.int64)

    X_va_t = torch.tensor(X_va_flat, dtype=torch.float32)
    mask_va_t = torch.tensor(data_va["race_masks"], dtype=torch.float32)

    X_full_t = torch.tensor(X_full_flat, dtype=torch.float32)
    mask_full_t = torch.tensor(data_full["race_masks"], dtype=torch.float32)

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

    # Train, Val, Full Test に対する推論
    final_tr_probs = predict_batches(model, X_tr_t, mask_tr_t)
    final_va_probs = predict_batches(model, X_va_t, mask_va_t)
    final_full_probs = predict_batches(model, X_full_t, mask_full_t)

    # 評価: 全量1,502レースでのテスト成績
    full_res = calculate_race_array_payout_metric(
        probs=final_full_probs,
        masks=data_full["race_masks"],
        y_true=data_full["race_y"],
        odds=data_full["odds"],
        payouts=data_full["payouts"],
        selection_mode="prob"
    )
    val_res = calculate_race_array_payout_metric(
        probs=final_va_probs,
        masks=data_va["race_masks"],
        y_true=data_va["race_y"],
        odds=data_va["odds"],
        payouts=data_va["payouts"],
        selection_mode="prob"
    )

    full_odds = full_res["total_payout"] / (full_res["hits"] * 100) if full_res["hits"] > 0 else 0
    val_odds = val_res["total_payout"] / (val_res["hits"] * 100) if val_res["hits"] > 0 else 0

    del model, optimizer, scheduler, X_tr_t, mask_tr_t, y_tr_t, X_va_t, mask_va_t, X_full_t, mask_full_t
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "step": step,
        "model_idx": model_idx,
        "best_model_state": best_model_state,
        "tr_probs": final_tr_probs,
        "va_probs": final_va_probs,
        "full_probs": final_full_probs,
        "full_test_hit_rate": full_res["hit_rate"],
        "full_test_hits": full_res["hits"],
        "full_test_mean_odds": full_odds,
        "full_test_roi": full_res["roi"],
        "full_test_payout": full_res["total_payout"],
        "val_hit_rate": val_res["hit_rate"],
        "val_roi": val_res["roi"],
        "val_mean_odds": val_odds
    }

def main():
    artifacts_dir = "./artifacts_cascade_full1502"
    os.makedirs(artifacts_dir, exist_ok=True)

    RANDOM_SEARCH_SEED = 2026
    set_seed(RANDOM_SEARCH_SEED)
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)

    device = get_device()
    logger.info(f"Compute Device: {device}")

    # キャッシュデータロード (Train, Val)
    cached_data_path = "./artifacts_dirt_1win/cached_dirt_1win_data.pkl"
    logger.info(f"Loading cached precomputed data from {cached_data_path}...")
    with open(cached_data_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    # 全量1,502レースキャッシュロード
    full_cache_path = "./artifacts_dirt_1win/cached_test_full_1502.pkl"
    logger.info(f"Loading full 1,502 test data from {full_cache_path}...")
    with open(full_cache_path, "rb") as f:
        data_full = pickle.load(f)

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
    logger.info(f"  Random Cascade Stacking Search on FULL 1,502 Races Benchmark")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races, 2024年包含)")
    logger.info(f"  Test:  2025-2026 Shuffled Full Benchmark ({len(data_full['race_y']):,d} races)")
    logger.info(f"  Stop Condition: Stop chain immediately when Full Test ROI drops")
    logger.info(f"  Goal: 100 Chains OR 5 Patterns with Full Test ROI >= 100%")
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
        prev_full_p = None
        prev_roi = -1.0

        for step in range(1, 51):
            m_idx = shuffled_models[step - 1]
            feat_idx = model_feature_indices[m_idx]

            # ベーステンソル
            X_tr_b = data_tr["race_3d"][:, :, feat_idx]
            X_va_b = data_va["race_3d"][:, :, feat_idx]
            X_full_b = data_full["race_3d"][:, :, feat_idx]

            if prev_tr_p is not None:
                X_tr_in = np.concatenate([X_tr_b, prev_tr_p[:, :, np.newaxis]], axis=-1)
                X_va_in = np.concatenate([X_va_b, prev_va_p[:, :, np.newaxis]], axis=-1)
                X_full_in = np.concatenate([X_full_b, prev_full_p[:, :, np.newaxis]], axis=-1)
            else:
                X_tr_in = X_tr_b
                X_va_in = X_va_b
                X_full_in = X_full_b

            t0 = time.time()
            res = train_and_eval_step(
                step=step,
                model_idx=m_idx,
                X_tr_3d=X_tr_in,
                X_va_3d=X_va_in,
                X_full_3d=X_full_in,
                data_tr=data_tr,
                data_va=data_va,
                data_full=data_full,
                device=device,
                epochs=4
            )
            dt = time.time() - t0
            total_models_trained += 1

            curr_roi = res["full_test_roi"]
            chain_steps.append({
                "chain_id": chain_idx,
                "step": step,
                "model_idx": m_idx,
                "num_features": X_tr_in.shape[2],
                "full_test_roi": curr_roi,
                "full_test_hit_rate": res["full_test_hit_rate"],
                "full_test_hits": res["full_test_hits"],
                "full_test_mean_odds": res["full_test_mean_odds"],
                "full_test_payout": res["full_test_payout"],
                "val_roi": res["val_roi"],
                "val_hit_rate": res["val_hit_rate"],
                "time_sec": dt
            })

            logger.info(
                f"  Chain {chain_idx:3d} | Step {step:2d} (Model {m_idx:2d}): "
                f"Full Test ROI={curr_roi*100:6.2f}% | Hit={res['full_test_hit_rate']*100:5.2f}% ({res['full_test_hits']}/{len(data_full['race_y'])}R) | "
                f"Odds={res['full_test_mean_odds']:4.2f}x | (Prev ROI={prev_roi*100:5.2f}%) ({dt:.1f}s)"
            )

            # ROI 100% 超えのチェック
            if curr_roi >= 1.00:
                pattern_info = {
                    "pattern_id": len(roi_over_100_hits) + 1,
                    "chain_id": chain_idx,
                    "step": step,
                    "chain_models": [cs["model_idx"] for cs in chain_steps],
                    "final_model_idx": m_idx,
                    "full_test_roi": curr_roi,
                    "full_test_hit_rate": res["full_test_hit_rate"],
                    "full_test_hits": res["full_test_hits"],
                    "full_test_mean_odds": res["full_test_mean_odds"],
                    "full_test_payout": res["full_test_payout"],
                    "val_roi": res["val_roi"],
                    "val_hit_rate": res["val_hit_rate"]
                }
                roi_over_100_hits.append(pattern_info)
                logger.info(f"  >>> ★ FOUND FULL TEST ROI >= 100% pattern (#{len(roi_over_100_hits)}): ROI={curr_roi*100:.2f}%! <<<")

                # モデル重み保存
                model_save_path = os.path.join(artifacts_dir, f"pattern_{len(roi_over_100_hits)}_chain{chain_idx}_step{step}_m{m_idx}.pth")
                torch.save(res["best_model_state"], model_save_path)
                logger.info(f"      Saved model weights to {model_save_path}")

                # 目標件数 (5件) 達成チェック
                if len(roi_over_100_hits) >= TARGET_ROI_OVER_100_COUNT:
                    logger.info(f"\nTarget achieved! Found {TARGET_ROI_OVER_100_COUNT} patterns with Full Test ROI >= 100%!")
                    break

            # ユーザー指示: 「また下がったらそこでやめてください」
            if step > 1 and curr_roi < prev_roi:
                logger.info(f"  --> ROI dropped from {prev_roi*100:.2f}% to {curr_roi*100:.2f}%. Stopping Chain {chain_idx} at Step {step}.")
                break

            prev_roi = curr_roi
            prev_tr_p = res["tr_probs"]
            prev_va_p = res["va_probs"]
            prev_full_p = res["full_probs"]

        all_chains_history.extend(chain_steps)

        # 保存 (各チェーン終了ごとに最新状態を追記保存)
        pd.DataFrame(all_chains_history).to_csv(os.path.join(artifacts_dir, "random_cascade_full1502_all_steps.csv"), index=False)
        with open(os.path.join(artifacts_dir, "roi_over_100_full1502_patterns.json"), "w", encoding="utf-8") as f:
            json.dump(roi_over_100_hits, f, ensure_ascii=False, indent=2)

        if len(roi_over_100_hits) >= TARGET_ROI_OVER_100_COUNT:
            break

    elapsed_total = time.time() - t_start
    logger.info(f"\n==================================================================")
    logger.info(f"  Search Finished in {elapsed_total/60:.1f} mins!")
    logger.info(f"  Total Chains Evaluated: {chain_idx} | Total Models Trained: {total_models_trained}")
    logger.info(f"  ROI >= 100% Patterns Found: {len(roi_over_100_hits)}")
    logger.info(f"==================================================================\n")

if __name__ == "__main__":
    main()
