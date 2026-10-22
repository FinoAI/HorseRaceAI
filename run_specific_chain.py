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

logger = setup_logger("SpecificChain")

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
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    artifacts_dir: str = "./artifacts_specific_chain"
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

    # モデル保存
    model_save_path = os.path.join(artifacts_dir, f"chain_step_{step}_m{model_idx}.pth")
    torch.save(best_model_state, model_save_path)

    # 全データに対する推論
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
        "val_mean_odds": val_res["mean_hit_odds"]
    }

def main():
    artifacts_dir = "./artifacts_specific_chain"
    os.makedirs(artifacts_dir, exist_ok=True)

    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
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

    # ユーザー指定チェーン: M17 -> M50 -> M40 -> M42
    chain_models = [17, 50, 40, 42]

    # 特徴量ロード
    model_feature_indices = {}
    for m in chain_models:
        feat_path = f"./artifacts_dirt_1win/features_{m}.json"
        with open(feat_path, "r", encoding="utf-8") as f:
            f_meta = json.load(f)
        s_yearly = f_meta["yearly_features"]
        s_indices = [col_to_idx[c] for c in s_yearly]
        model_feature_indices[m] = list(bac_col_indices) + s_indices

    logger.info(f"\n==================================================================")
    logger.info(f"  Executing Specific Cascade Chain: M17 -> M50 -> M40 -> M42")
    logger.info(f"  Step 1: Model 17 (Base 200 features)")
    logger.info(f"  Step 2: Model 50 (Base 200 + M17 Pred Prob = 201 features)")
    logger.info(f"  Step 3: Model 40 (Base 200 + M50 Pred Prob = 201 features)")
    logger.info(f"  Step 4: Model 42 (Base 200 + M40 Pred Prob = 201 features)")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races)")
    logger.info(f"  Test: 2025-2026 ({len(data_te['race_y']):,d} races)")
    logger.info(f"==================================================================\n")

    steps_results = []
    all_te_probs = []
    prev_tr_p = None
    prev_va_p = None
    prev_te_p = None

    t_start = time.time()

    for step_num, m_idx in enumerate(chain_models, start=1):
        feat_idx = model_feature_indices[m_idx]

        X_tr_b = data_tr["race_3d"][:, :, feat_idx]
        X_va_b = data_va["race_3d"][:, :, feat_idx]
        X_te_b = data_te["race_3d"][:, :, feat_idx]

        if prev_tr_p is not None:
            X_tr_in = np.concatenate([X_tr_b, prev_tr_p[:, :, np.newaxis]], axis=-1)
            X_va_in = np.concatenate([X_va_b, prev_va_p[:, :, np.newaxis]], axis=-1)
            X_te_in = np.concatenate([X_te_b, prev_te_p[:, :, np.newaxis]], axis=-1)
            feat_note = f"Base 200 + Prev Prob (201 features)"
        else:
            X_tr_in = X_tr_b
            X_va_in = X_va_b
            X_te_in = X_te_base = X_te_b
            feat_note = f"Base 200 features (Initial Step)"

        logger.info(f">>> [Step {step_num}/4] Training Model {m_idx:2d} | {feat_note} <<<")
        t0 = time.time()
        res = train_and_eval_step(
            step=step_num,
            model_idx=m_idx,
            X_tr_3d=X_tr_in,
            X_va_3d=X_va_in,
            X_te_3d=X_te_in,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            device=device,
            epochs=5,
            artifacts_dir=artifacts_dir
        )
        dt = time.time() - t0

        prev_tr_p = res["tr_probs"]
        prev_va_p = res["va_probs"]
        prev_te_p = res["te_probs"]
        all_te_probs.append(res["te_probs"])

        steps_results.append({
            "step": step_num,
            "model_idx": m_idx,
            "num_features": res["num_features"],
            "cascade_test_hit": res["test_hit_rate"],
            "cascade_test_hits": res["test_hits"],
            "cascade_test_odds": res["test_mean_odds"],
            "cascade_test_roi": res["test_roi"],
            "cascade_test_payout": res["test_payout"],
            "val_hit_rate": res["val_hit_rate"],
            "val_roi": res["val_roi"],
            "time_sec": dt
        })

        logger.info(
            f"    Result Step {step_num} (Model {m_idx:2d}): "
            f"Test Hit={res['test_hit_rate']*100:.2f}% ({res['test_hits']}/751R) | "
            f"Odds={res['test_mean_odds']:.2f}x | Test ROI={res['test_roi']*100:.2f}% (¥{res['test_payout']:,.0f}) | "
            f"Val Hit={res['val_hit_rate']*100:.2f}%, Val ROI={res['val_roi']*100:.2f}% ({dt:.1f}s)"
        )

    # 結果保存
    df_steps = pd.DataFrame(steps_results)
    df_steps.to_csv(os.path.join(artifacts_dir, "specific_chain_summary.csv"), index=False)
    np.save(os.path.join(artifacts_dir, "specific_chain_test_probs.npy"), np.array(all_te_probs))

    # 累積アンサンブル評価 (Step 1〜4)
    cum_results = []
    for s in range(1, 5):
        sub_probs = np.array(all_te_probs[:s])
        ens_p = np.mean(sub_probs, axis=0)
        eval_res = evaluate_subset_with_odds([0], ens_p[np.newaxis, ...], data_te)
        val_eval_res = evaluate_subset_with_odds([0], ens_p[np.newaxis, ...], data_va)
        cum_results.append({
            "step": s,
            "models_included": chain_models[:s],
            "cum_test_hit": eval_res["hit_rate"],
            "cum_test_hits": eval_res["hits"],
            "cum_test_odds": eval_res["mean_hit_odds"],
            "cum_test_roi": eval_res["roi"],
            "cum_test_payout": eval_res["total_payout"],
            "cum_val_hit": val_eval_res["hit_rate"],
            "cum_val_roi": val_eval_res["roi"]
        })
    df_cum = pd.DataFrame(cum_results)
    df_cum.to_csv(os.path.join(artifacts_dir, "specific_chain_cumulative.csv"), index=False)

    print("\n" + "="*125)
    print("      ★ 指定チェーン M17 -> M50 -> M40 -> M42 カスケード推移結果 ★")
    print("="*125)
    formatters = {
        "cascade_test_hit": lambda x: f"{x*100:5.2f}%",
        "cascade_test_roi": lambda x: f"{x*100:6.2f}%",
        "cascade_test_odds": lambda x: f"{x:5.2f}倍",
        "cascade_test_payout": lambda x: f"¥{x:,.0f}",
        "val_hit_rate": lambda x: f"{x*100:5.2f}%",
        "val_roi": lambda x: f"{x*100:6.2f}%",
    }
    disp_df = df_steps[["step", "model_idx", "cascade_test_hit", "cascade_test_hits", "cascade_test_odds", "cascade_test_roi", "cascade_test_payout", "val_hit_rate", "val_roi"]].copy()
    for col, fmt in formatters.items():
        if col in disp_df.columns:
            disp_df[col] = disp_df[col].apply(fmt)
    print(disp_df.to_string(index=False))

    print("\n" + "-"*125)
    print("【★ チェーンモデル群の累積平均アンサンブル成績 (Step 1〜s)】")
    cum_formatters = {
        "cum_test_hit": lambda x: f"{x*100:5.2f}%",
        "cum_test_roi": lambda x: f"{x*100:6.2f}%",
        "cum_test_odds": lambda x: f"{x:5.2f}倍",
        "cum_test_payout": lambda x: f"¥{x:,.0f}",
        "cum_val_hit": lambda x: f"{x*100:5.2f}%",
        "cum_val_roi": lambda x: f"{x*100:6.2f}%",
    }
    disp_cum = df_cum[["step", "models_included", "cum_test_hit", "cum_test_hits", "cum_test_odds", "cum_test_roi", "cum_test_payout", "cum_val_hit", "cum_val_roi"]].copy()
    for col, fmt in cum_formatters.items():
        if col in disp_cum.columns:
            disp_cum[col] = disp_cum[col].apply(fmt)
    print(disp_cum.to_string(index=False))
    print("="*125)

if __name__ == "__main__":
    main()
