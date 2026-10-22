"""
run_pipeline_turf_1win.py
========================================================================
JRA 芝・1勝クラス以上専用 AIモデル パイプライン
【200特徴量 × 50モデル ランダムサンプリング＆学習】
------------------------------------------------------------------------
■ 概要・仕様:
1. 対象データ: JRA 芝 1勝クラス以上 (1勝/2勝/3勝/オープン/重賞)
   - 芝ダ障害コード == 1 (芝)
   - KYI_条件クラス in [1, 2, 3, 9] (1勝クラス以上)
2. データ期間:
   - 学習データ (Train): 2009年〜2024年 (16年間)
   - 検証データ (Val):   2025年〜2026年 (50% シャッフル分割)
   - テストデータ (Test): 2025年〜2026年 (50% シャッフル分割・完全ブラインド)
3. 特徴量構成 (各モデル 200次元):
   - BAC_KAB 共通特徴量 (36列)
   - 開催年候補特徴量からランダムサンプリング (164列)
   - 合計 200列 (18頭フルゲート展開時: 3,600次元入力)
4. モデル数: 50モデル (Model 1 〜 Model 50)
   - 各モデルのアーキテクチャ: RaceLevelStage1NN (multiplier=2, reduction=0.5, dropout=0.15)
   - エポック数: 5, バッチサイズ: 512, オプティマイザ: AdamW (lr=0.001)
5. 成果物 (artifacts_turf_1win/):
   - cached_turf_1win_data.pkl (前処理済みテンソルキャッシュ)
   - turf_1win_preprocessor_stats.pkl (推論用正規化統計量)
   - features_1.json 〜 features_50.json (各モデル特徴量定義)
   - model_1.pth 〜 model_50.pth (学習済みモデル重み)
   - single_models_performance.csv (全50モデル個別成績一覧)
   - ensemble_turf_1win_summary.csv (アンサンブル選抜成績一覧)
========================================================================
"""

import os
import sys
import json
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.data.preprocessor import extract_target_and_meta
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("PipelineTurf1Win")

def load_bac_kab(data_dir: str, bac_filename: str = "BAC_KAB.csv") -> pd.DataFrame:
    bac_path = os.path.join(data_dir, bac_filename)
    if not os.path.exists(bac_path):
        raise FileNotFoundError(f"BAC_KAB file not found at: {bac_path}")
    logger.info(f"Loading BAC_KAB from {bac_path}...")
    df_bac = pd.read_csv(bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)
    logger.info(f"Loaded BAC_KAB: shape={df_bac.shape}")
    return df_bac

def load_filtered_year_data(
    data_dir: str,
    year: int,
    turf_race_ids: set,
    df_bac: pd.DataFrame
) -> pd.DataFrame:
    year_path = os.path.join(data_dir, f"{year}.csv")
    if not os.path.exists(year_path):
        raise FileNotFoundError(f"Year file not found at: {year_path}")
    
    t0 = time.time()
    df_year = pd.read_csv(year_path, dtype={"KYI_RACE_KEY": str}, low_memory=False)
    race_keys = df_year["KYI_RACE_KEY"].astype(str).str.zfill(10)
    race_ids = race_keys.str[:8]
    
    df_year = df_year.assign(
        KYI_RACE_KEY=race_keys,
        RACE_ID=race_ids
    )
    
    # フィルタ: 芝 (芝ダ障害コード==1) かつ 1勝以上クラス (KYI_条件クラス in [1, 2, 3, 9])
    cond_mask = df_year["RACE_ID"].isin(turf_race_ids) & df_year["KYI_条件クラス"].isin([1, 2, 3, 9])
    df_filtered = df_year[cond_mask].copy().reset_index(drop=True)
    
    # BAC_KAB とマージ
    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_filtered.columns]
    merged = pd.merge(df_filtered, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
    merged = merged.assign(YEAR=year)
    dt = time.time() - t0
    logger.info(f"Year {year}: {len(df_filtered):,d} rows ({df_filtered['RACE_ID'].nunique()} races) loaded & filtered in {dt:.1f}s")
    return merged

def prepare_turf_1win_data(
    data_dir: str,
    bac_filename: str = "BAC_KAB.csv",
    train_years: List[int] = list(range(2009, 2025)),
    val_test_years: List[int] = [2025, 2026],
    val_ratio: float = 0.5,
    random_seed: int = 42,
    prohibited_cols: List[str] = None,
    id_cols: List[str] = None,
    max_horses: int = 18,
    cache_path: str = "./artifacts_turf_1win/cached_turf_1win_data.pkl",
    stats_path: str = "./artifacts_turf_1win/turf_1win_preprocessor_stats.pkl"
) -> Dict:
    """
    芝・1勝以上クラス専用データを読み込み、前処理・テンソル化してキャッシュ保存/読み込み
    """
    if os.path.exists(cache_path):
        logger.info(f"Found cached precomputed data at {cache_path}. Loading...")
        with open(cache_path, "rb") as f:
            data_dict = pickle.load(f)
        logger.info(f"Successfully loaded cached data from {cache_path}!")
        return data_dict

    logger.info("=== Loading and Filtering Datasets for Turf 1-Win+ Class ===")
    df_bac = load_bac_kab(data_dir, bac_filename)
    # 芝 (芝ダ障害コード == 1)
    turf_race_ids = set(df_bac[df_bac["芝ダ障害コード"] == 1]["RACE_ID"])
    bac_feature_cols = [c for c in df_bac.columns if c != "RACE_ID"]
    logger.info(f"Total Turf Races in BAC_KAB: {len(turf_race_ids)}")

    # 1. 学習データのロード (2009〜2024)
    logger.info(f"Loading Train Datasets: {train_years[0]} - {train_years[-1]} ({len(train_years)} Years)...")
    train_dfs = []
    for y in train_years:
        df_y = load_filtered_year_data(data_dir, y, turf_race_ids, df_bac)
        train_dfs.append(df_y)
    df_train = pd.concat(train_dfs, ignore_index=True)
    logger.info(f"Combined Train DataFrame: shape={df_train.shape}, races={df_train['RACE_ID'].nunique()}")

    # 2. 検証・テストデータのロード (2025 & 2026)
    logger.info(f"Loading Val/Test Datasets: {val_test_years}...")
    val_test_dfs = []
    for y in val_test_years:
        df_y = load_filtered_year_data(data_dir, y, turf_race_ids, df_bac)
        val_test_dfs.append(df_y)
    df_val_test = pd.concat(val_test_dfs, ignore_index=True)
    logger.info(f"Combined 2025-2026 DataFrame: shape={df_val_test.shape}, races={df_val_test['RACE_ID'].nunique()}")

    # レース単位でシャッフル分割 (50% Val, 50% Test)
    unique_races = np.array(df_val_test["RACE_ID"].drop_duplicates().tolist())
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_races)

    n_val_races = int(len(unique_races) * val_ratio)
    val_race_ids = set(unique_races[:n_val_races])
    test_race_ids = set(unique_races[n_val_races:])

    df_val = df_val_test[df_val_test["RACE_ID"].isin(val_race_ids)].reset_index(drop=True)
    df_test = df_val_test[df_val_test["RACE_ID"].isin(test_race_ids)].reset_index(drop=True)

    logger.info(f"Val DataFrame (shuffled 2025-2026): shape={df_val.shape}, races={len(val_race_ids)}")
    logger.info(f"Test DataFrame (shuffled 2025-2026): shape={df_test.shape}, races={len(test_race_ids)}")

    # ターゲット & メタデータ抽出
    y_tr, meta_tr = extract_target_and_meta(df_train)
    y_va, meta_va = extract_target_and_meta(df_val)
    y_te, meta_te = extract_target_and_meta(df_test)

    # 候補カラムの特定
    prohibited_set = set(prohibited_cols or []) | set(id_cols or []) | {"RACE_ID", "KYI_RACE_KEY", "YEAR"}
    bac_feature_cols = [c for c in bac_feature_cols if c not in prohibited_set and not c.startswith("Target_")]
    
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
    logger.info("Normalizing features across all candidates with Train statistics...")
    mat_train = np.nan_to_num(mat_train, nan=np.nan, posinf=1e6, neginf=-1e6)
    with np.errstate(all='ignore'):
        medians = np.nanmedian(mat_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

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

    # 推論用正規化統計量を保存
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_data = {
        "all_feature_cols": all_feature_cols,
        "col_to_idx": col_to_idx,
        "bac_feature_cols": bac_feature_cols,
        "medians": medians,
        "means": means,
        "stds": stds
    }
    with open(stats_path, "wb") as f:
        pickle.dump(stats_data, f)
    logger.info(f"Saved preprocessing stats to {stats_path}!")

    # 3D テンソル配列の構築
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
            "odds": race_odds,
            "payouts": race_payouts,
            "ranks": race_ranks,
            "race_ids": np.array(race_ids_ordered)
        }

    logger.info("Building 3D race-level tensors...")
    data_tr = build_3d_tensor(mat_train, meta_tr)
    data_va = build_3d_tensor(mat_val, meta_va)
    data_te = build_3d_tensor(mat_test, meta_te)

    logger.info(f"Train Races: {len(data_tr['race_y'])}, Val Races: {len(data_va['race_y'])}, Test Races: {len(data_te['race_y'])}")

    result_dict = {
        "data_tr": data_tr,
        "data_va": data_va,
        "data_te": data_te,
        "bac_col_indices": bac_col_indices,
        "bac_feature_cols": bac_feature_cols,
        "yearly_candidate_cols": yearly_candidate_cols,
        "all_feature_cols": all_feature_cols,
        "col_to_idx": col_to_idx
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    logger.info(f"Saving precomputed data cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached successfully!")

    return result_dict

def evaluate_subset_with_odds(subset: list, probs_array: np.ndarray, meta_dict: dict):
    sub_probs = probs_array[subset]
    ens_probs = np.mean(sub_probs, axis=0)

    metrics = calculate_race_array_payout_metric(
        probs=ens_probs,
        masks=meta_dict["race_masks"],
        y_true=meta_dict["race_y"],
        odds=meta_dict["odds"],
        payouts=meta_dict["payouts"],
        selection_mode="prob",
        unit_bet=100
    )

    hits = metrics["hits"]
    total_races = metrics["total_races"]
    total_payout = metrics["total_payout"]
    hit_rate = metrics["hit_rate"]
    roi = metrics["roi"]
    mean_hit_odds = (total_payout / 100.0) / hits if hits > 0 else 0.0

    return {
        "subset": list(subset),
        "k": len(subset),
        "hit_rate": hit_rate,
        "roi": roi,
        "mean_hit_odds": mean_hit_odds,
        "hits": hits,
        "total_races": total_races,
        "total_payout": total_payout,
        "score": roi * min(1.0, hit_rate / 0.20) + hit_rate
    }

def greedy_optimize_subset(candidates: list, target_k: int, all_val_probs: np.ndarray, data_va: dict):
    selected = []
    pool = list(candidates)

    for step in range(target_k):
        best_cand = None
        best_score = -1e9
        for c in pool:
            trial = selected + [c]
            res = evaluate_subset_with_odds(trial, all_val_probs, data_va)
            if res["score"] > best_score:
                best_score = res["score"]
                best_cand = c
        selected.append(best_cand)
        pool.remove(best_cand)

    return selected

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
    artifacts_dir: str = "./artifacts_turf_1win"
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

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    mask_tr_t = torch.tensor(mask_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64)

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    mask_va_t = torch.tensor(mask_va, dtype=torch.float32)

    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    mask_te_t = torch.tensor(mask_te, dtype=torch.float32)

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
            epoch_loss += loss.item() * len(idx)

        scheduler.step()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # Val 評価 (最良エポック選定)
        val_probs = predict_batches(model, X_va_t, mask_va_t)
        val_res = evaluate_subset_with_odds([0], val_probs[np.newaxis, ...], data_va)
        val_score = val_res["score"]

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_probs = val_probs
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    best_model_path = os.path.join(artifacts_dir, f"model_{model_idx}.pth")
    torch.save(best_model_state, best_model_path)

    # Test 推論 (完全ブラインド)
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    test_probs = predict_batches(model, X_te_t, mask_te_t)

    val_res = evaluate_subset_with_odds([0], best_val_probs[np.newaxis, ...], data_va)
    test_res = evaluate_subset_with_odds([0], test_probs[np.newaxis, ...], data_te)

    del model, optimizer, scheduler, X_tr_t, mask_tr_t, y_tr_t, X_va_t, mask_va_t, X_te_t, mask_te_t
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
    artifacts_dir = os.path.join(CURRENT_DIR, "artifacts_turf_1win")
    os.makedirs(artifacts_dir, exist_ok=True)

    config_path = os.path.join(CURRENT_DIR, "configs/config.yaml")
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. データ準備 & キャッシュ機構 (芝・1勝クラス以上)
    cache_path = os.path.join(artifacts_dir, "cached_turf_1win_data.pkl")
    stats_path = os.path.join(artifacts_dir, "turf_1win_preprocessor_stats.pkl")

    data_dict = prepare_turf_1win_data(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=list(range(2009, 2025)),   # 2009〜2024 (16年間)
        val_test_years=[2025, 2026],          # 2025〜2026
        val_ratio=0.5,
        random_seed=cfg["data"]["random_seed"],
        prohibited_cols=cfg["feature_selection"]["prohibited_columns"],
        id_cols=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        max_horses=18,
        cache_path=cache_path,
        stats_path=stats_path
    )

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    yearly_candidates = data_dict["yearly_candidate_cols"]
    col_to_idx = data_dict["col_to_idx"]

    TOTAL_MODELS = 50
    FEATURE_COUNT = 200
    YEARLY_SAMPLE_COUNT = FEATURE_COUNT - len(bac_col_indices)

    logger.info(f"\n==================================================================")
    logger.info(f"  Turf 1-Win+ Pipeline: {TOTAL_MODELS} Models × {FEATURE_COUNT} Features")
    logger.info(f"  Train: 2009-2024 ({len(data_tr['race_y']):,d} races)")
    logger.info(f"  Val (50% of 2025-2026): {len(data_va['race_y']):,d} races")
    logger.info(f"  Test (50% of 2025-2026): {len(data_te['race_y']):,d} races")
    logger.info(f"  Input Dimension: 18 horses × {FEATURE_COUNT} features = 3,600 dimensions")
    logger.info(f"==================================================================\n")

    all_val_probs = []
    all_test_probs = []
    models_summary = []

    t_start = time.time()

    for m in range(1, TOTAL_MODELS + 1):
        # 途中再開チェック
        w_path = os.path.join(artifacts_dir, f"model_{m}.pth")
        f_path = os.path.join(artifacts_dir, f"features_{m}.json")

        rng = random.Random(cfg["data"]["random_seed"] + m * 37)
        sampled_yearly = rng.sample(yearly_candidates, YEARLY_SAMPLE_COUNT)
        sampled_yearly_indices = [col_to_idx[c] for c in sampled_yearly]
        selected_indices = list(bac_col_indices) + sampled_yearly_indices

        feat_meta = {
            "model_idx": m,
            "num_features": len(selected_indices),
            "bac_features_count": len(bac_col_indices),
            "yearly_features_count": len(sampled_yearly),
            "yearly_features": sampled_yearly
        }
        with open(f_path, "w", encoding="utf-8") as f:
            json.dump(feat_meta, f, ensure_ascii=False, indent=2)

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
                f"Val ROI={res['val_roi']*100:.2f}% | Test ROI={res['test_roi']*100:.2f}% | "
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
    # 2. アンサンブル組み合わせ構築 ＆ 評価
    # =========================================================================
    logger.info("\n=== Building and Evaluating Ensemble Patterns (Turf 1-Win+ 50 Models) ===")

    top_by_score = [int(m) - 1 for m in df_models.sort_values("val_score", ascending=False)["model_idx"].values]

    upset_candidates = [
        int(m) - 1 for m in df_models[df_models["val_hit_rate"] < 0.22].sort_values("val_roi", ascending=False)["model_idx"].values
    ]
    if len(upset_candidates) < 5:
        upset_candidates = [int(m) - 1 for m in df_models.sort_values("val_mean_odds", ascending=False)["model_idx"].values[:10]]

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

    # (C) 単体スコア上位選抜
    for k in [5, 8, 10, 12, 15, 20]:
        log_pattern("Top Selection", f"Top {k} Models", top_by_score[:k])

    # (D) 上位＋穴馬ブレンド
    log_pattern("Top + Upset Blend", "Top 6 + Upset 4 (10 Models)", top_by_score[:6] + upset_candidates[:4])
    log_pattern("Top + Upset Blend", "Top 8 + Upset 4 (12 Models)", top_by_score[:8] + upset_candidates[:4])
    log_pattern("Top + Upset Blend", "Top 10 + Upset 5 (15 Models)", top_by_score[:10] + upset_candidates[:5])

    # (E) 最適 Greedy 選抜
    all_models = list(range(TOTAL_MODELS))
    for k in [5, 8, 9, 10, 11, 12, 14, 15]:
        best_greedy_sub = greedy_optimize_subset(all_models, target_k=k, all_val_probs=all_val_probs, data_va=data_va)
        log_pattern("Greedy Optimized", f"Optimized Ensemble k={k}", best_greedy_sub)

    # (F) 排他独立アンサンブル (系統A vs 系統B: 重複0個)
    sys_A = greedy_optimize_subset(all_models, target_k=10, all_val_probs=all_val_probs, data_va=data_va)
    pool_B = [m for m in all_models if m not in sys_A]
    sys_B = greedy_optimize_subset(pool_B, target_k=10, all_val_probs=all_val_probs, data_va=data_va)
    pool_C = [m for m in pool_B if m not in sys_B]
    sys_C = greedy_optimize_subset(pool_C, target_k=10, all_val_probs=all_val_probs, data_va=data_va)

    log_pattern("Exclusive Disjoint", "Exclusive System A (10 Models)", sys_A)
    log_pattern("Exclusive Disjoint", "Exclusive System B (10 Models - No Overlap)", sys_B)
    log_pattern("Exclusive Disjoint", "Exclusive System C (10 Models - No Overlap)", sys_C)
    log_pattern("Exclusive Disjoint", "Combined Dual System A+B (20 Models)", sys_A + sys_B)

    df_comp = pd.DataFrame(comparison_results)
    csv_out = os.path.join(artifacts_dir, "ensemble_turf_1win_summary.csv")
    df_comp.to_csv(csv_out, index=False)

    print("\n" + "="*125)
    print("      ★ 芝・1勝以上クラス専用モデル (2009-2024学習 / 2025-2026検証・テスト) 総合結果 ★")
    print("="*125)
    print(f"未見テストデータ: {len(data_te['race_y'])} レース (2025-2026年シャッフル | selection_mode: prob)")
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
    print(f"- ensemble_turf_1win_summary.csv")
    print(f"- single_models_performance.csv")
    print(f"- all_val_probs.npy, all_test_probs.npy\n")

if __name__ == "__main__":
    main()
