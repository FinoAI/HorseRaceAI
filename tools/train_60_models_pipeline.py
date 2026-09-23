import os
import sys
import struct
import json
import random
import torch
import numpy as np
import pandas as pd
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import setup_logger, set_seed, load_config, get_device, save_json
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta, build_race_level_dataset
from src.models.base_nn import RaceLevelStage1NN
from src.training.train_stage1 import train_single_stage1_model, predict_race_stage1

logger = setup_logger("Train60Models")

def save_meta_bin(race_dataset, out_path):
    N = len(race_dataset["race_ids"])
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", N, 18))
        for i in range(N):
            y = int(race_dataset["race_y"][i])
            f.write(struct.pack("<i", y))
            f.write(race_dataset["race_masks"][i].astype(np.float32).tobytes())
            f.write(race_dataset["odds"][i].astype(np.float32).tobytes())
            f.write(race_dataset["payouts"][i].astype(np.float32).tobytes())
    logger.info(f"Saved metadata to {out_path} ({N} races)")

def sample_diverse_features(
    model_idx: int,
    bac_cols: List[str],
    yearly_cols: List[str],
    bac_count: int = 6,
    yearly_count: int = 4,
    random_seed: int = 42
) -> List[str]:
    """
    アンサンブルの多様性を最大化するため、BACカラムおよび開催年カラムからそれぞれ指定数をサンプリング
    選ばれた特徴量リスト (計 D=10) を返す (全馬で共通使用)
    """
    rng = random.Random(random_seed + model_idx * 17)
    
    k_bac = min(bac_count, len(bac_cols))
    k_yr = min(yearly_count, len(yearly_cols))
    
    sel_bac = rng.sample(bac_cols, k_bac)
    sel_yr = rng.sample(yearly_cols, k_yr)
    
    return sel_bac + sel_yr

def main():
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Using compute device: {device}")

    data_c_dir = "data_c"
    os.makedirs(data_c_dir, exist_ok=True)
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. データのロード (Train: 2022-2023, Val/Test: 2024-2025)
    logger.info("=== Loading Datasets (Train: 2022-2023, Val/Test: 2024-2025) ===")
    df_train, df_val, df_test, bac_feature_cols = get_train_val_test_datasets(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=[2022, 2023],
        val_test_years=[2024, 2025],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        sample_n_races_per_year=100 # 評価に十分なレース数
    )

    y_train_raw, meta_train = extract_target_and_meta(df_train)
    y_val_raw, meta_val = extract_target_and_meta(df_val)
    y_test_raw, meta_test = extract_target_and_meta(df_test)

    max_horses = 18

    # 2. メタデータバイナリの保存
    val_meta_path = os.path.join(data_c_dir, "val_meta.bin")
    test_meta_path = os.path.join(data_c_dir, "test_meta.bin")
    
    dummy_feat = np.zeros((len(meta_val), 1), dtype=np.float32)
    sample_val = build_race_level_dataset(dummy_feat, meta_val, max_horses=max_horses)
    save_meta_bin(sample_val, val_meta_path)
    
    dummy_feat_te = np.zeros((len(meta_test), 1), dtype=np.float32)
    sample_test = build_race_level_dataset(dummy_feat_te, meta_test, max_horses=max_horses)
    save_meta_bin(sample_test, test_meta_path)

    N_val_races = len(sample_val["race_ids"])
    N_test_races = len(sample_test["race_ids"])
    logger.info(f"Target Evaluation Races: Val={N_val_races} races, Test={N_test_races} races")

    # 3. 前処理と特徴量候補の抽出
    preprocessor = FeaturePreprocessor(
        prohibited_columns=cfg["feature_selection"]["prohibited_columns"],
        id_and_str_columns=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        bac_feature_cols=bac_feature_cols,
        artifacts_dir=artifacts_dir
    )
    yearly_candidates = preprocessor.get_candidate_yearly_columns(df_train.columns.tolist())
    clean_bac_cols = preprocessor.bac_feature_cols
    logger.info(f"Available candidates: BAC={len(clean_bac_cols)}, Yearly={len(yearly_candidates)}")

    # 4. 60個のモデルを高速に順次学習 & Val/Test 予測を蓄積
    num_total_models = 60
    logger.info(f"\n=== Training {num_total_models} Diverse Models (D=10, 1st layer=5760 units, 8 layers) ===")

    all_val_preds = []  # shape: (60, N_val, 18)
    all_test_preds = [] # shape: (60, N_test, 18)

    for m_idx in range(1, num_total_models + 1):
        selected_features = sample_diverse_features(
            model_idx=m_idx,
            bac_cols=clean_bac_cols,
            yearly_cols=yearly_candidates,
            bac_count=6,
            yearly_count=4,
            random_seed=cfg["data"]["random_seed"]
        )

        json_path = os.path.join(artifacts_dir, f"features_model_{m_idx}.json")
        save_json({"model_idx": m_idx, "features": selected_features}, json_path)

        # 特徴量正規化
        X_tr = preprocessor.fit_transform_features(df_train, selected_features, model_idx=m_idx)
        X_va = preprocessor.transform_features(df_val, selected_features, model_idx=m_idx)
        X_te = preprocessor.transform_features(df_test, selected_features, model_idx=m_idx)

        race_tr = build_race_level_dataset(X_tr, meta_train, max_horses=max_horses)
        race_va = build_race_level_dataset(X_va, meta_val, max_horses=max_horses)
        race_te = build_race_level_dataset(X_te, meta_test, max_horses=max_horses)

        # モデル学習 (高速ピラミッド構造)
        model, score = train_single_stage1_model(
            model_idx=m_idx,
            race_train=race_tr,
            race_val=race_va,
            max_horses=max_horses,
            multiplier=32,
            reduction_ratio=0.5,
            dropout=0.2,
            learning_rate=0.001,
            weight_decay=0.0001,
            epochs=2,
            early_stopping_patience=1,
            eval_metric="payout",
            selection_mode="ev_filtered",
            min_prob_for_ev=0.10,
            loss_weighting="payout",
            artifacts_dir=artifacts_dir,
            device=device
        )

        val_pred = predict_race_stage1(model, race_va["race_X"], race_va["race_masks"], device=device)
        test_pred = predict_race_stage1(model, race_te["race_X"], race_te["race_masks"], device=device)

        all_val_preds.append(val_pred.astype(np.float32))
        all_test_preds.append(test_pred.astype(np.float32))

        if m_idx % 10 == 0 or m_idx == 1:
            logger.info(f"[Progress] Completed Model {m_idx}/{num_total_models} | Val Payout Score: {score:.4f}")

    # 5. C言語用の予測テンソルバイナリの保存
    val_preds_arr = np.stack(all_val_preds, axis=0) # (60, N_val, 18)
    test_preds_arr = np.stack(all_test_preds, axis=0) # (60, N_test, 18)

    val_preds_bin_path = os.path.join(data_c_dir, "val_preds_60.bin")
    test_preds_bin_path = os.path.join(data_c_dir, "test_preds_60.bin")

    with open(val_preds_bin_path, "wb") as f:
        f.write(struct.pack("<III", num_total_models, N_val_races, max_horses))
        f.write(val_preds_arr.tobytes())

    with open(test_preds_bin_path, "wb") as f:
        f.write(struct.pack("<III", num_total_models, N_test_races, max_horses))
        f.write(test_preds_arr.tobytes())

    logger.info(f"Successfully generated C-ready predictions: {val_preds_bin_path} and {test_preds_bin_path}")
    logger.info(f"Val shape: {val_preds_arr.shape}, Test shape: {test_preds_arr.shape}")

if __name__ == "__main__":
    main()
