import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta, build_race_level_dataset
from src.training.train_stage1 import train_single_stage1_model
from tools.export_weights_to_bin import export_model_to_bin

logger = setup_logger("TrainExportModel1")

def main():
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Using compute device: {device}")

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. 実データのロード (Model 1 用にサンプリングして高速学習)
    logger.info("Loading real dataset for Model 1...")
    df_train, df_val, df_test, bac_feature_cols = get_train_val_test_datasets(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=[2023], # 直近の開催年
        val_test_years=[2024, 2025],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        sample_n_races_per_year=10 # 迅速にモデル1の重みを確立
    )

    y_train_raw, meta_train = extract_target_and_meta(df_train)
    y_val_raw, meta_val = extract_target_and_meta(df_val)

    # 2. 特徴量選定 (Model 1)
    preprocessor = FeaturePreprocessor(
        prohibited_columns=cfg["feature_selection"]["prohibited_columns"],
        id_and_str_columns=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        bac_feature_cols=bac_feature_cols,
        artifacts_dir=artifacts_dir
    )
    yearly_candidates = preprocessor.get_candidate_yearly_columns(df_train.columns.tolist())
    sample_count = cfg["feature_selection"].get("num_random_yearly_features", 14)

    # Model 1 の特徴量選定 & JSON保存
    selected_features = preprocessor.sample_features_for_model(
        model_idx=1,
        yearly_candidate_cols=yearly_candidates,
        sample_count=sample_count,
        random_seed=cfg["data"]["random_seed"]
    )

    # 特徴量正規化
    X_tr_norm = preprocessor.fit_transform_features(df_train, selected_features, model_idx=1)
    X_va_norm = preprocessor.transform_features(df_val, selected_features, model_idx=1)

    max_horses = cfg["data"].get("max_horses_per_race", 18)
    race_train = build_race_level_dataset(X_tr_norm, meta_train, max_horses=max_horses)
    race_val = build_race_level_dataset(X_va_norm, meta_val, max_horses=max_horses)

    # 3. Model 1 の学習
    logger.info("Training Stage 1 Model 1...")
    model, best_score = train_single_stage1_model(
        model_idx=1,
        race_train=race_train,
        race_val=race_val,
        max_horses=max_horses,
        multiplier=cfg["stage1"]["multiplier"],
        reduction_ratio=cfg["stage1"].get("reduction_ratio", 0.5),
        dropout=cfg["stage1"]["dropout"],
        learning_rate=cfg["stage1"]["learning_rate"],
        weight_decay=cfg["stage1"]["weight_decay"],
        epochs=2,
        early_stopping_patience=2,
        eval_metric=cfg.get("evaluation", {}).get("eval_metric", "payout"),
        selection_mode=cfg.get("evaluation", {}).get("selection_mode", "ev_filtered"),
        min_prob_for_ev=cfg.get("evaluation", {}).get("min_prob_for_ev", 0.10),
        loss_weighting=cfg.get("evaluation", {}).get("loss_weighting", "none"),
        artifacts_dir=artifacts_dir,
        device=device
    )

    # 4. C言語/Metal向けバイナリへのエクスポート
    bin_output_path = os.path.join(artifacts_dir, "model_1.bin")
    logger.info(f"Exporting Model 1 to C/Metal binary: {bin_output_path}...")
    export_model_to_bin(model, bin_output_path)
    logger.info("Model 1 training and C/Metal binary export completed successfully!")

if __name__ == "__main__":
    main()
