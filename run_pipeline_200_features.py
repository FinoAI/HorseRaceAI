import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from run_search_ensemble import prepare_precomputed_data
from run_comparison_studies import evaluate_subset_with_odds, greedy_optimize_subset
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("Pipeline200")

def train_and_eval_single_model_200(
    model_idx: int,
    data_tr: dict,
    data_va: dict,
    data_te: dict,
    selected_col_indices: list,
    device: torch.device,
    multiplier: int = 2,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    artifacts_dir: str = "./artifacts_models_200"
):
    max_horses = 18
    num_features = len(selected_col_indices) # 200
    input_dim = max_horses * num_features   # 3600

    X_tr = data_tr["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_tr = data_tr["race_masks"]
    y_tr = data_tr["race_y"]

    X_va = data_va["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_va = data_va["race_masks"]
    y_va = data_va["race_y"]

    X_te = data_te["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_te = data_te["race_masks"]

    X_tr_gpu = torch.tensor(X_tr, dtype=torch.float32, device=device)
    mask_tr_gpu = torch.tensor(mask_tr, dtype=torch.float32, device=device)
    y_tr_gpu = torch.tensor(y_tr, dtype=torch.int64, device=device)

    X_va_gpu = torch.tensor(X_va, dtype=torch.float32, device=device)
    mask_va_gpu = torch.tensor(mask_va, dtype=torch.float32, device=device)

    X_te_gpu = torch.tensor(X_te, dtype=torch.float32, device=device)
    mask_te_gpu = torch.tensor(mask_te, dtype=torch.float32, device=device)

    model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=multiplier,
        reduction_ratio=0.5,
        dropout=0.15
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_score = -1.0
    best_val_probs = None
    best_model_state = None
    num_races = len(X_tr)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_races, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, num_races, batch_size):
            idx = perm[i:i + batch_size]
            bx = X_tr_gpu[idx]
            bmask = mask_tr_gpu[idx]
            by = y_tr_gpu[idx]

            optimizer.zero_grad()
            logits = model(bx, mask=bmask)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Val 評価
        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_gpu, mask=mask_va_gpu)
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()

        metrics = calculate_race_array_payout_metric(
            probs=val_probs,
            masks=mask_va,
            y_true=y_va,
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

    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")
    torch.save(best_model_state, best_model_path)

    # Test 推論
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_gpu, mask=mask_te_gpu)
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()

    val_res = evaluate_subset_with_odds([0], best_val_probs[np.newaxis, ...], data_va)
    test_res = evaluate_subset_with_odds([0], test_probs[np.newaxis, ...], data_te)

    del model, optimizer, scheduler, X_tr_gpu, mask_tr_gpu, y_tr_gpu, X_va_gpu, mask_va_gpu, X_te_gpu, mask_te_gpu
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_idx": model_idx,
        "val_probs": best_val_probs,
        "test_probs": test_probs,
        "val_hit_rate": val_res["hit_rate"],
        "val_roi": val_res["roi"],
        "val_mean_odds": val_res["mean_hit_odds"],
        "val_score": best_val_score,
        "test_hit_rate": test_res["hit_rate"],
        "test_roi": test_res["roi"],
        "test_mean_odds": test_res["mean_hit_odds"],
        "test_hits": test_res["hits"],
        "test_payout": test_res["total_payout"]
    }

def main():
    artifacts_dir = "./artifacts_models_200"
    os.makedirs(artifacts_dir, exist_ok=True)

    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. データ準備 & 一括正規化
    data_dict = prepare_precomputed_data(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=cfg["data"]["train_years"],
        val_test_years=cfg["data"]["val_test_years"],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        prohibited_cols=cfg["feature_selection"]["prohibited_columns"],
        id_cols=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        max_horses=18
    )

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    yearly_candidates = data_dict["yearly_candidate_cols"]
    col_to_idx = data_dict["col_to_idx"]

    TOTAL_MODELS = 50
    FEATURE_COUNT = 200
    YEARLY_SAMPLE_COUNT = FEATURE_COUNT - len(bac_col_indices) # 200 - 36 = 164

    logger.info(f"\n==================================================================")
    logger.info(f"  Starting Training: {TOTAL_MODELS} Models × {FEATURE_COUNT} Features per Horse")
    logger.info(f"  Input Dimension: 18 horses × {FEATURE_COUNT} features = 3,600 dimensions")
    logger.info(f"  Features per model: BAC_KAB ({len(bac_col_indices)}) + Yearly Random ({YEARLY_SAMPLE_COUNT})")
    logger.info(f"==================================================================\n")

    all_val_probs = []
    all_test_probs = []
    models_summary = []

    t_start = time.time()

    for m in range(1, TOTAL_MODELS + 1):
        rng = random.Random(cfg["data"]["random_seed"] + m * 31)
        sampled_yearly = rng.sample(yearly_candidates, YEARLY_SAMPLE_COUNT)
        sampled_yearly_indices = [col_to_idx[c] for c in sampled_yearly]
        selected_indices = list(bac_col_indices) + sampled_yearly_indices

        # 特徴量リスト保存
        feat_meta = {
            "model_idx": m,
            "num_features": len(selected_indices),
            "bac_features_count": len(bac_col_indices),
            "yearly_features_count": len(sampled_yearly),
            "yearly_features": sampled_yearly
        }
        with open(os.path.join(artifacts_dir, f"features_{m}.json"), "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, ensure_ascii=False, indent=2)

        t0 = time.time()
        res = train_and_eval_single_model_200(
            model_idx=m,
            data_tr=data_tr,
            data_va=data_va,
            data_te=data_te,
            selected_col_indices=selected_indices,
            device=device,
            multiplier=2,
            epochs=5,
            batch_size=512,
            lr=0.001,
            weight_decay=0.0001,
            artifacts_dir=artifacts_dir
        )
        dt = time.time() - t0

        all_val_probs.append(res["val_probs"])
        all_test_probs.append(res["test_probs"])
        models_summary.append({
            "model_idx": m,
            "val_hit_rate": res["val_hit_rate"],
            "val_roi": res["val_roi"],
            "val_mean_odds": res["val_mean_odds"],
            "val_score": res["val_score"],
            "test_hit_rate": res["test_hit_rate"],
            "test_roi": res["test_roi"],
            "test_mean_odds": res["test_mean_odds"],
            "test_hits": res["test_hits"],
            "test_payout": res["test_payout"],
            "time_sec": dt
        })

        if m % 5 == 0 or m == TOTAL_MODELS:
            elapsed = time.time() - t_start
            eta = (elapsed / m) * (TOTAL_MODELS - m)
            logger.info(
                f"[Progress: {m:2d}/{TOTAL_MODELS}] Model {m:2d}: Val Hit={res['val_hit_rate']*100:.2f}% | "
                f"Val ROI={res['val_roi']*100:.2f}% | Mean Odds={res['val_mean_odds']:.2f}x | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m"
            )

    all_val_probs = np.array(all_val_probs)   # (50, N_val, 18)
    all_test_probs = np.array(all_test_probs) # (50, N_test, 18)

    # 予測キャッシュ保存
    np.save(os.path.join(artifacts_dir, "all_val_probs.npy"), all_val_probs)
    np.save(os.path.join(artifacts_dir, "all_test_probs.npy"), all_test_probs)
    np.savez_compressed(
        os.path.join(artifacts_dir, "eval_meta.npz"),
        va_race_masks=data_va["race_masks"],
        va_race_y=data_va["race_y"],
        va_odds=data_va["odds"],
        va_payouts=data_va["payouts"],
        va_ranks=data_va["ranks"],
        va_race_ids=data_va["race_ids"],
        te_race_masks=data_te["race_masks"],
        te_race_y=data_te["race_y"],
        te_odds=data_te["odds"],
        te_payouts=data_te["payouts"],
        te_ranks=data_te["ranks"],
        te_race_ids=data_te["race_ids"]
    )

    df_models = pd.DataFrame(models_summary)
    df_models.to_csv(os.path.join(artifacts_dir, "single_models_performance.csv"), index=False)

    # =========================================================================
    # 2. アンサンブル組み合わせ構築 ＆ 的中平均オッズ評価
    # =========================================================================
    logger.info("\n=== Building and Evaluating Ensemble Patterns (50 Models × 200 Features) ===")

    top_by_score = [int(m) - 1 for m in df_models.sort_values("val_score", ascending=False)["model_idx"].values]
    upset_candidates = [
        int(m) - 1 for m in df_models[df_models["val_hit_rate"] < 0.20].sort_values("val_roi", ascending=False)["model_idx"].values
    ]
    if len(upset_candidates) < 5:
        upset_candidates = [int(m) - 1 for m in df_models.sort_values("val_mean_odds", ascending=False)["model_idx"].values[:10]]

    mid_candidates = top_by_score[15:35]

    comparison_results = []

    def log_pattern(cat, name, subset):
        v_res = evaluate_subset_with_odds(subset, all_val_probs, data_va)
        t_res = evaluate_subset_with_odds(subset, all_test_probs, data_te)
        comparison_results.append({
            "category": cat,
            "pattern": name,
            "k": len(subset),
            "val_hit_rate": v_res["hit_rate"],
            "val_roi": v_res["roi"],
            "val_mean_odds": v_res["mean_hit_odds"],
            "test_hit_rate": t_res["hit_rate"],
            "test_roi": t_res["roi"],
            "test_mean_odds": t_res["mean_hit_odds"],
            "test_hits": t_res["hits"],
            "test_total_races": t_res["total_races"],
            "test_payout": t_res["total_payout"],
            "models_used": [m + 1 for m in subset]
        })

    # (A) 単体ベストモデル
    log_pattern("Baseline", f"Single Best Model (M{top_by_score[0]+1})", [top_by_score[0]])

    # (B) 全50モデル単純平均
    log_pattern("Full Ensemble", "All 50 Models Average", list(range(TOTAL_MODELS)))

    # (C) 単体スコア上位選抜 (Top Models)
    for k in [8, 10, 12, 15, 20]:
        log_pattern("Top Selection", f"Top {k} Models", top_by_score[:k])

    # (D) 上位＋中位穴馬ブレンド (Blend)
    log_pattern("Top + Upset Blend", "Top 6 + Upset 4 (10 Models)", top_by_score[:6] + upset_candidates[:4])
    log_pattern("Top + Upset Blend", "Top 8 + Upset 4 (12 Models)", top_by_score[:8] + upset_candidates[:4])
    log_pattern("Top + Upset Blend", "Top 10 + Upset 5 (15 Models)", top_by_score[:10] + upset_candidates[:5])
    log_pattern("Top + Mid Blend", "Top 8 + Mid 6 (14 Models)", top_by_score[:8] + mid_candidates[:6])

    # (E) 最適 Greedy 選抜 (k=8..15)
    all_models = list(range(TOTAL_MODELS))
    for k in [8, 9, 10, 11, 12, 14, 15]:
        best_greedy_sub = greedy_optimize_subset(all_models, target_k=k, all_val_probs=all_val_probs, data_va=data_va)
        log_pattern("Greedy Optimized", f"Optimized Ensemble k={k}", best_greedy_sub)

    # (F) 排他独立アンサンブル (系統A vs 系統B: 重複0個)
    sys_A = greedy_optimize_subset(all_models, target_k=10, all_val_probs=all_val_probs, data_va=data_va)
    pool_B = [m for m in all_models if m not in sys_A]
    sys_B = greedy_optimize_subset(pool_B, target_k=10, all_val_probs=all_val_probs, data_va=data_va)
    pool_C = [m for m in pool_B if m not in sys_B]
    sys_C = greedy_optimize_subset(pool_C, target_k=10, all_val_probs=all_val_probs, data_va=data_va)

    assert len(set(sys_A) & set(sys_B)) == 0, "A and B overlap!"
    assert len(set(sys_A) & set(sys_C)) == 0, "A and C overlap!"
    assert len(set(sys_B) & set(sys_C)) == 0, "B and C overlap!"

    log_pattern("Exclusive Disjoint", "Exclusive System A (10 Models)", sys_A)
    log_pattern("Exclusive Disjoint", "Exclusive System B (10 Models - No Overlap)", sys_B)
    log_pattern("Exclusive Disjoint", "Exclusive System C (10 Models - No Overlap)", sys_C)
    log_pattern("Exclusive Disjoint", "Combined Dual System A+B (20 Models)", sys_A + sys_B)

    df_comp = pd.DataFrame(comparison_results)
    csv_out = os.path.join(artifacts_dir, "ensemble_200_features_summary.csv")
    df_comp.to_csv(csv_out, index=False)

    print("\n" + "="*125)
    print("      ★ 200特徴量 × 50モデル構築 ＆ 多様性アンサンブル総合評価 (的中平均オッズ付き) ★")
    print("="*125)
    print("未見テストデータ: 3,455 レース (2024-2025年 | selection_mode: prob (EV考慮なし))")
    print("-"*125)
    formatters = {
        "val_hit_rate": lambda x: f"{x*100:5.2f}%",
        "val_roi": lambda x: f"{x*100:6.2f}%",
        "val_mean_odds": lambda x: f"{x:5.2f}倍",
        "test_hit_rate": lambda x: f"{x*100:5.2f}%",
        "test_roi": lambda x: f"{x*100:6.2f}%",
        "test_mean_odds": lambda x: f"{x:5.2f}倍",
        "test_payout": lambda x: f"¥{x:,.0f}"
    }

    display_df = df_comp[["category", "pattern", "k", "test_hit_rate", "test_hits", "test_mean_odds", "test_roi", "test_payout", "val_hit_rate", "val_roi", "val_mean_odds"]].copy()
    for col, fmt in formatters.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)

    print(display_df.to_string(index=False))
    print("="*125)
    print(f"\n[Artifacts Saved in {artifacts_dir}]")
    print(f"- ensemble_200_features_summary.csv")
    print(f"- single_models_performance.csv")
    print(f"- all_val_probs.npy, all_test_probs.npy\n")

if __name__ == "__main__":
    main()
