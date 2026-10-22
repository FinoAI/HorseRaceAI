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
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import extract_target_and_meta
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("EnsembleSearch")

def prepare_precomputed_data(
    data_dir: str,
    bac_filename: str,
    train_years: list,
    val_test_years: list,
    val_ratio: float,
    random_seed: int,
    prohibited_cols: list,
    id_cols: list,
    max_horses: int = 18,
    sample_races: int = None
):
    """
    全データをロードし、全特徴量を一括正規化して 3D レース単位配列に配置。
    以後のモデル学習でスライシングのみで高速に特徴量を切り出せるようにする。
    """
    logger.info("=== Loading Datasets ===")
    df_train, df_val, df_test, bac_cols = get_train_val_test_datasets(
        data_dir=data_dir,
        bac_filename=bac_filename,
        train_years=train_years,
        val_test_years=val_test_years,
        val_ratio=val_ratio,
        random_seed=random_seed,
        sample_n_races_per_year=sample_races
    )

    y_tr, meta_tr = extract_target_and_meta(df_train)
    y_va, meta_va = extract_target_and_meta(df_val)
    y_te, meta_te = extract_target_and_meta(df_test)

    # 候補カラムの特定
    prohibited_set = set(prohibited_cols) | set(id_cols) | {"RACE_ID", "KYI_RACE_KEY", "YEAR"}
    bac_feature_cols = [c for c in bac_cols if c not in prohibited_set and not c.startswith("Target_")]
    
    all_cols = df_train.columns.tolist()
    yearly_candidate_cols = []
    for c in all_cols:
        if c.startswith("Target_") or c in prohibited_set or c in bac_feature_cols:
            continue
        yearly_candidate_cols.append(c)

    logger.info(f"Total BAC features: {len(bac_feature_cols)} | Yearly candidate features: {len(yearly_candidate_cols)}")

    # 全特徴量リスト: BAC_KAB (先頭) + 開催年候補列
    all_feature_cols = list(bac_feature_cols) + yearly_candidate_cols
    col_to_idx = {c: i for i, c in enumerate(all_feature_cols)}
    bac_col_indices = [col_to_idx[c] for c in bac_feature_cols]

    logger.info("Converting all features to float32 matrices...")
    
    def df_to_numeric_matrix(df, cols):
        mat = np.zeros((len(df), len(cols)), dtype=np.float32)
        for i, c in enumerate(cols):
            mat[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(np.nan).values
        return mat

    mat_train = df_to_numeric_matrix(df_train, all_feature_cols)
    mat_val = df_to_numeric_matrix(df_val, all_feature_cols)
    mat_test = df_to_numeric_matrix(df_test, all_feature_cols)

    # Train の統計量で一括正規化
    logger.info("Normalizing features across all candidates...")
    mat_train = np.nan_to_num(mat_train, nan=np.nan, posinf=1e6, neginf=-1e6)
    with np.errstate(all='ignore'):
        medians = np.nanmedian(mat_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

    # NaN を median で埋める
    for col_i in range(mat_train.shape[1]):
        m_val = medians[col_i]
        nan_tr = np.isnan(mat_train[:, col_i])
        if np.any(nan_tr):
            mat_train[nan_tr, col_i] = m_val

        nan_va = np.isnan(mat_val[:, col_i])
        if np.any(nan_va):
            mat_val[nan_va, col_i] = m_val

        nan_te = np.isnan(mat_test[:, col_i])
        if np.any(nan_te):
            mat_test[nan_te, col_i] = m_val

    with np.errstate(all='ignore'):
        means = np.mean(mat_train, axis=0)
        stds = np.std(mat_train, axis=0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds <= 1e-5] = 1.0
    means = np.nan_to_num(means, nan=0.0)

    mat_train = np.clip((mat_train - means) / stds, -10.0, 10.0)
    mat_val = np.clip((mat_val - means) / stds, -10.0, 10.0)
    mat_test = np.clip((mat_test - means) / stds, -10.0, 10.0)

    mat_train = np.nan_to_num(mat_train, nan=0.0)
    mat_val = np.nan_to_num(mat_val, nan=0.0)
    mat_test = np.nan_to_num(mat_test, nan=0.0)

    # 3D テンソル配列の事前構築
    def build_3d_tensor(mat, meta_df):
        race_ids_ordered = []
        seen = set()
        for r in meta_df["RACE_ID"].values:
            if r not in seen:
                seen.add(r)
                race_ids_ordered.append(r)
        
        num_races = len(race_ids_ordered)
        race_id_to_idx = {r: i for i, r in enumerate(race_ids_ordered)}
        
        race_3d = np.zeros((num_races, max_horses, mat.shape[1]), dtype=np.float32)
        race_masks = np.zeros((num_races, max_horses), dtype=np.float32)
        race_y = np.zeros(num_races, dtype=np.int64)
        race_odds = np.zeros((num_races, max_horses), dtype=np.float32)
        race_payouts = np.zeros((num_races, max_horses), dtype=np.float32)
        race_ranks = np.full((num_races, max_horses), 99.0, dtype=np.float32)

        r_ids = meta_df["RACE_ID"].values
        horse_nums = meta_df["KYI_馬番"].values
        ranks = meta_df["Target_着順"].values
        odds = meta_df["Target_確定単勝オッズ"].values
        payouts = meta_df["Target_単勝"].values

        for i in range(len(meta_df)):
            r_idx = race_id_to_idx[r_ids[i]]
            slot = int(horse_nums[i]) - 1
            if 0 <= slot < max_horses:
                race_3d[r_idx, slot, :] = mat[i]
                race_masks[r_idx, slot] = 1.0
                rank = ranks[i]
                race_ranks[r_idx, slot] = rank
                if rank == 1:
                    race_y[r_idx] = slot
                race_odds[r_idx, slot] = odds[i]
                p = payouts[i]
                race_payouts[r_idx, slot] = p if not np.isnan(p) else 0.0

        return {
            "race_3d": race_3d,
            "race_masks": race_masks,
            "race_y": race_y,
            "race_ids": np.array(race_ids_ordered),
            "odds": race_odds,
            "payouts": race_payouts,
            "ranks": race_ranks
        }

    logger.info("Building 3D race-level tensors...")
    data_tr = build_3d_tensor(mat_train, meta_tr)
    data_va = build_3d_tensor(mat_val, meta_va)
    data_te = build_3d_tensor(mat_test, meta_te)

    logger.info(f"Train races: {len(data_tr['race_y'])} | Val races: {len(data_va['race_y'])} | Test races: {len(data_te['race_y'])}")

    return {
        "data_tr": data_tr,
        "data_va": data_va,
        "data_te": data_te,
        "all_feature_cols": all_feature_cols,
        "bac_col_indices": bac_col_indices,
        "yearly_candidate_cols": yearly_candidate_cols,
        "col_to_idx": col_to_idx
    }

def train_and_eval_single_model(
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
    artifacts_dir: str = "./artifacts_models_100"
):
    """
    1つのモデルを指定された100特徴量で学習し、ValおよびTestでの予測確率テンソルを返す。
    """
    max_horses = 18
    num_features = len(selected_col_indices) # 100
    input_dim = max_horses * num_features   # 1800

    # 特徴量スライス & 2Dフラット化 (高速)
    X_tr = data_tr["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_tr = data_tr["race_masks"]
    y_tr = data_tr["race_y"]

    X_va = data_va["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_va = data_va["race_masks"]
    y_va = data_va["race_y"]

    X_te = data_te["race_3d"][:, :, selected_col_indices].reshape(-1, input_dim)
    mask_te = data_te["race_masks"]

    # GPU テンソル (GPU常駐で高速化)
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

        # Val 評価 (EV考慮なし: selection_mode="prob")
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
        # 評価スコア: 的中率と回収率の双方を評価
        val_score = roi * min(1.0, hit_rate / 0.20) + hit_rate

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_probs = val_probs
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 最良モデルを保存
    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")
    torch.save(best_model_state, best_model_path)

    # Test 推論
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_gpu, mask=mask_te_gpu)
        test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()

    # 単体成績計算
    val_metrics = calculate_race_array_payout_metric(
        probs=best_val_probs,
        masks=mask_va,
        y_true=y_va,
        odds=data_va["odds"],
        payouts=data_va["payouts"],
        selection_mode="prob"
    )

    test_metrics = calculate_race_array_payout_metric(
        probs=test_probs,
        masks=data_te["race_masks"],
        y_true=data_te["race_y"],
        odds=data_te["odds"],
        payouts=data_te["payouts"],
        selection_mode="prob"
    )

    # メモリ解放
    del model, optimizer, scheduler, X_tr_gpu, mask_tr_gpu, y_tr_gpu, X_va_gpu, mask_va_gpu, X_te_gpu, mask_te_gpu
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "model_idx": model_idx,
        "val_probs": best_val_probs,
        "test_probs": test_probs,
        "val_hit_rate": val_metrics["hit_rate"],
        "val_roi": val_metrics["roi"],
        "val_score": best_val_score,
        "test_hit_rate": test_metrics["hit_rate"],
        "test_roi": test_metrics["roi"]
    }

def evaluate_ensemble(
    subset_indices: list,
    all_val_probs: np.ndarray, # shape (M, N_val, 18)
    data_va: dict,
    selection_mode: str = "prob"
) -> dict:
    """
    指定したモデルサブセットの平均予測確率による評価
    """
    sub_probs = all_val_probs[subset_indices] # (K, N_val, 18)
    ens_probs = np.mean(sub_probs, axis=0)     # (N_val, 18)

    metrics = calculate_race_array_payout_metric(
        probs=ens_probs,
        masks=data_va["race_masks"],
        y_true=data_va["race_y"],
        odds=data_va["odds"],
        payouts=data_va["payouts"],
        selection_mode=selection_mode
    )
    hit_rate = metrics["hit_rate"]
    roi = metrics["roi"]
    score = roi * min(1.0, hit_rate / 0.20) + hit_rate
    return {
        "subset": list(subset_indices),
        "k": len(subset_indices),
        "hit_rate": hit_rate,
        "roi": roi,
        "score": score,
        "total_payout": metrics["total_payout"],
        "total_investment": metrics["total_investment"],
        "hits": metrics["hits"],
        "total_races": metrics["total_races"]
    }

def search_best_ensembles(
    all_val_probs: np.ndarray, # shape (100, N_val, 18)
    data_va: dict,
    min_k: int = 8,
    max_k: int = 15,
    ga_iterations: int = 4000
) -> dict:
    """
    100モデルの中から、k ∈ [8, 15] の最良アンサンブル組み合わせを探索。
    1. Greedy Forward Selection
    2. Local Search / GA による近傍探索
    """
    num_models = len(all_val_probs)
    logger.info(f"=== Starting Ensemble Combination Search (k={min_k}~{max_k} from {num_models} models) ===")

    # 1. Greedy Forward Selection
    selected = []
    candidates = list(range(num_models))
    greedy_history = {}

    for step in range(1, max_k + 1):
        best_cand = None
        best_cand_score = -1.0
        best_cand_metrics = None

        for c in candidates:
            trial = selected + [c]
            res = evaluate_ensemble(trial, all_val_probs, data_va)
            if res["score"] > best_cand_score:
                best_cand_score = res["score"]
                best_cand = c
                best_cand_metrics = res

        selected.append(best_cand)
        candidates.remove(best_cand)
        greedy_history[step] = best_cand_metrics
        logger.info(f"[Greedy k={step:2d}] Added Model {best_cand+1:3d} | Val Hit: {best_cand_metrics['hit_rate']*100:.2f}% | Val ROI: {best_cand_metrics['roi']*100:.2f}% | Score: {best_cand_metrics['score']:.4f}")

    # 2. Local Search / 遺伝的探索 (k=8..15 の各サイズをさらに最適化)
    best_per_k = {}
    for k in range(min_k, max_k + 1):
        init_subset = greedy_history[k]["subset"]
        best_res = greedy_history[k]
        current_subset = list(init_subset)

        rng = random.Random(42 + k)
        # 近傍探索 (1モデルを入れ替える変異)
        for gen in range(ga_iterations):
            neighbor = list(current_subset)
            avail = [m for m in range(num_models) if m not in neighbor]
            if not avail:
                break
            remove_idx = rng.randrange(k)
            add_m = rng.choice(avail)
            neighbor[remove_idx] = add_m

            res = evaluate_ensemble(neighbor, all_val_probs, data_va)
            if res["score"] > best_res["score"]:
                best_res = res
                current_subset = list(neighbor)
            elif rng.random() < 0.05: # Simulated Annealing風のランダム遷移
                current_subset = list(neighbor)

        best_per_k[k] = best_res
        logger.info(
            f"★ Best Ensemble (k={k:2d}): Models {[m+1 for m in best_per_k[k]['subset']]} | "
            f"Val Hit Rate: {best_per_k[k]['hit_rate']*100:.2f}% | Val ROI: {best_per_k[k]['roi']*100:.2f}% | "
            f"Score: {best_per_k[k]['score']:.4f}"
        )

    # 全体最良の k を選定
    overall_best_k = max(best_per_k.keys(), key=lambda k: best_per_k[k]["score"])
    logger.info(f"\n>>> OVERALL BEST ENSEMBLE: k={overall_best_k} models (Score: {best_per_k[overall_best_k]['score']:.4f}) <<<")

    return {
        "best_per_k": best_per_k,
        "overall_best": best_per_k[overall_best_k],
        "overall_best_k": overall_best_k
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="100モデル学習 & 最適アンサンブル組み合わせ探索")
    parser.add_argument("--total_models", type=int, default=100, help="作成するモデル総数 (デフォルト: 100)")
    parser.add_argument("--feature_count", type=int, default=100, help="各馬の特徴量数 (デフォルト: 100)")
    parser.add_argument("--sample_races", type=int, default=None, help="疎通確認用の年あたりサンプリングレース数")
    parser.add_argument("--min_k", type=int, default=8, help="アンサンブル最小モデル数 (デフォルト: 8)")
    parser.add_argument("--max_k", type=int, default=15, help="アンサンブル最大モデル数 (デフォルト: 15)")
    args = parser.parse_args()

    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Compute Device: {device}")

    artifacts_dir = "./artifacts_models_100"
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. データ準備 & 事前正規化
    data_dict = prepare_precomputed_data(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=cfg["data"]["train_years"],
        val_test_years=cfg["data"]["val_test_years"],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        prohibited_cols=cfg["feature_selection"]["prohibited_columns"],
        id_cols=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        max_horses=18,
        sample_races=args.sample_races
    )

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    yearly_candidates = data_dict["yearly_candidate_cols"]
    col_to_idx = data_dict["col_to_idx"]

    # 2. モデル学習
    TOTAL_MODELS = args.total_models
    FEATURE_COUNT = args.feature_count
    YEARLY_SAMPLE_COUNT = max(2, FEATURE_COUNT - len(bac_col_indices))
    logger.info(f"\n=== Training {TOTAL_MODELS} Diverse Models (Features per horse: {FEATURE_COUNT} = BAC {len(bac_col_indices)} + Yearly {YEARLY_SAMPLE_COUNT}) ===")

    all_val_probs = []
    all_test_probs = []
    models_summary = []

    t_start = time.time()

    for m in range(1, TOTAL_MODELS + 1):
        rng = random.Random(cfg["data"]["random_seed"] + m * 7)
        sampled_yearly = rng.sample(yearly_candidates, YEARLY_SAMPLE_COUNT)
        sampled_yearly_indices = [col_to_idx[c] for c in sampled_yearly]
        selected_indices = list(bac_col_indices) + sampled_yearly_indices

        # 特徴量リストを JSON 保存
        feat_meta = {
            "model_idx": m,
            "num_features": len(selected_indices),
            "bac_features_count": len(bac_col_indices),
            "yearly_features_count": len(sampled_yearly),
            "yearly_features": sampled_yearly
        }
        with open(os.path.join(artifacts_dir, f"features_{m}.json"), "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, ensure_ascii=False, indent=2)

        # モデル学習 & 予測
        t0 = time.time()
        res = train_and_eval_single_model(
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
            "val_score": res["val_score"],
            "test_hit_rate": res["test_hit_rate"],
            "test_roi": res["test_roi"],
            "time_sec": dt
        })

        if m % 10 == 0 or m == TOTAL_MODELS:
            elapsed = time.time() - t_start
            eta = (elapsed / m) * (TOTAL_MODELS - m)
            logger.info(
                f"[Progress: {m:3d}/{TOTAL_MODELS}] Model {m:3d}: Val Hit={res['val_hit_rate']*100:.2f}% | Val ROI={res['val_roi']*100:.2f}% | "
                f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m"
            )

    df_models = pd.DataFrame(models_summary)
    df_models.to_csv(os.path.join(artifacts_dir, "single_models_performance.csv"), index=False)
    logger.info(f"Saved single models performance to {os.path.join(artifacts_dir, 'single_models_performance.csv')}")

    # 3. 最良アンサンブル組み合わせ探索 (k=8..15)
    all_val_probs = np.array(all_val_probs)   # (TOTAL_MODELS, N_val, 18)
    all_test_probs = np.array(all_test_probs) # (TOTAL_MODELS, N_test, 18)

    actual_min_k = max(2, min(args.min_k, TOTAL_MODELS))
    actual_max_k = max(actual_min_k, min(args.max_k, TOTAL_MODELS))

    search_res = search_best_ensembles(
        all_val_probs=all_val_probs,
        data_va=data_va,
        min_k=actual_min_k,
        max_k=actual_max_k,
        ga_iterations=3000
    )

    best_per_k = search_res["best_per_k"]
    overall_best = search_res["overall_best"]
    overall_best_k = search_res["overall_best_k"]

    # 4. 未見テストデータ (2024-2025 Test) での最終評価
    logger.info("\n=== Step 4: Final Test Evaluation on Unseen 2024-2025 Data ===")

    test_eval_summary = []

    # (A) 単体ベストモデル (Val基準)
    best_single_m = int(df_models.sort_values("val_score", ascending=False).iloc[0]["model_idx"]) - 1
    single_test_metrics = calculate_race_array_payout_metric(
        probs=all_test_probs[best_single_m],
        masks=data_te["race_masks"],
        y_true=data_te["race_y"],
        odds=data_te["odds"],
        payouts=data_te["payouts"],
        selection_mode="prob"
    )
    test_eval_summary.append({
        "pattern": f"Single Best Model (M{best_single_m+1})",
        "num_models": 1,
        "val_hit_rate": df_models.loc[best_single_m, "val_hit_rate"],
        "val_roi": df_models.loc[best_single_m, "val_roi"],
        "test_hit_rate": single_test_metrics["hit_rate"],
        "test_roi": single_test_metrics["roi"],
        "test_hits": single_test_metrics["hits"],
        "test_total_races": single_test_metrics["total_races"],
        "test_payout": single_test_metrics["total_payout"],
        "models_used": [best_single_m + 1]
    })

    # (B) 全モデル単純平均
    full_ens_probs = np.mean(all_test_probs, axis=0)
    full_ens_metrics = calculate_race_array_payout_metric(
        probs=full_ens_probs,
        masks=data_te["race_masks"],
        y_true=data_te["race_y"],
        odds=data_te["odds"],
        payouts=data_te["payouts"],
        selection_mode="prob"
    )
    test_eval_summary.append({
        "pattern": f"All {TOTAL_MODELS} Models Average",
        "num_models": TOTAL_MODELS,
        "val_hit_rate": float(np.nan),
        "val_roi": float(np.nan),
        "test_hit_rate": full_ens_metrics["hit_rate"],
        "test_roi": full_ens_metrics["roi"],
        "test_hits": full_ens_metrics["hits"],
        "test_total_races": full_ens_metrics["total_races"],
        "test_payout": full_ens_metrics["total_payout"],
        "models_used": f"All 1-{TOTAL_MODELS}"
    })

    # (C) 最適アンサンブル k の各パターン
    for k in range(actual_min_k, actual_max_k + 1):
        pattern_res = best_per_k[k]
        subset = pattern_res["subset"]
        sub_test_probs = np.mean(all_test_probs[subset], axis=0)
        
        te_metrics = calculate_race_array_payout_metric(
            probs=sub_test_probs,
            masks=data_te["race_masks"],
            y_true=data_te["race_y"],
            odds=data_te["odds"],
            payouts=data_te["payouts"],
            selection_mode="prob"
        )
        is_overall = (k == overall_best_k)
        prefix = "★ OVERALL BEST: " if is_overall else ""
        test_eval_summary.append({
            "pattern": f"{prefix}Ensemble k={k}",
            "num_models": k,
            "val_hit_rate": pattern_res["hit_rate"],
            "val_roi": pattern_res["roi"],
            "test_hit_rate": te_metrics["hit_rate"],
            "test_roi": te_metrics["roi"],
            "test_hits": te_metrics["hits"],
            "test_total_races": te_metrics["total_races"],
            "test_payout": te_metrics["total_payout"],
            "models_used": [m + 1 for m in subset]
        })

    df_test_summary = pd.DataFrame(test_eval_summary)
    df_test_summary.to_csv(os.path.join(artifacts_dir, "ensemble_patterns_summary.csv"), index=False)

    # 最良アンサンブルのテスト予測を保存
    best_subset = overall_best["subset"]
    best_test_probs = np.mean(all_test_probs[best_subset], axis=0)

    eval_rows = []
    for r_idx, r_id in enumerate(data_te["race_ids"]):
        mask = data_te["race_masks"][r_idx]
        probs = best_test_probs[r_idx]
        odds = data_te["odds"][r_idx]
        payouts = data_te["payouts"][r_idx]
        ranks = data_te["ranks"][r_idx]

        for slot in range(18):
            if mask[slot] > 0:
                horse_num = slot + 1
                eval_rows.append({
                    "RACE_ID": r_id,
                    "KYI_馬番": horse_num,
                    "prob": float(probs[slot]),
                    "Target_確定単勝オッズ": float(odds[slot]),
                    "Target_単勝": float(payouts[slot]),
                    "Target_着順": float(ranks[slot])
                })

    df_best_preds = pd.DataFrame(eval_rows)
    df_best_preds.to_csv(os.path.join(artifacts_dir, "best_ensemble_predictions.csv"), index=False)

    # ベストアンサンブル設定を JSON 保存
    best_config = {
        "best_k": overall_best_k,
        "selected_model_indices": [m + 1 for m in best_subset],
        "val_hit_rate": overall_best["hit_rate"],
        "val_roi": overall_best["roi"],
        "test_hit_rate": float(df_test_summary[df_test_summary["pattern"].str.contains("OVERALL BEST")]["test_hit_rate"].iloc[0]),
        "test_roi": float(df_test_summary[df_test_summary["pattern"].str.contains("OVERALL BEST")]["test_roi"].iloc[0]),
    }
    with open(os.path.join(artifacts_dir, "best_ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    print("\n" + "="*90)
    print("        ★ 100モデル学習 ＆ 最適アンサンブル組み合わせ探索 (k=8~15) 最終結果サマリー ★")
    print("="*90)
    print(f"テスト対象レース数: {data_te['race_masks'].shape[0]} レース (2024-2025年未見データ | selection_mode: prob (EV考慮なし))")
    print("-"*90)
    display_cols = ["pattern", "num_models", "val_hit_rate", "val_roi", "test_hit_rate", "test_roi", "test_hits", "test_payout"]
    print(df_test_summary[display_cols].to_string(index=False))
    print("="*90)
    print(f"\n[Artifacts Saved in {artifacts_dir}]")
    print(f"- best_ensemble_config.json")
    print(f"- ensemble_patterns_summary.csv")
    print(f"- single_models_performance.csv")
    print(f"- best_ensemble_predictions.csv\n")

if __name__ == "__main__":
    main()
