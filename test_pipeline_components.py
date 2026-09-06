import os
import json
import numpy as np
import pandas as pd
from src.utils.helpers import setup_logger, set_seed, load_json
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta
from src.evaluation.metrics import calculate_top1_hit_rate
from src.evaluation.simulation import BetSimulator

logger = setup_logger("TestPipeline")

def test_components():
    logger.info(">>> Starting Pipeline Components Test (No-GPU / Pure Python & Pandas) <<<")
    
    # 1. データローダーのテスト (実データから各年10レース分をサンプリング)
    data_dir = "/Users/hinomasafumi/Data/main"
    logger.info("Testing Data Loader with sample_n_races_per_year=10...")
    
    df_train, df_val, df_test, bac_cols = get_train_val_test_datasets(
        data_dir=data_dir,
        bac_filename="BAC_KAB.csv",
        train_years=[2022, 2023],
        val_test_years=[2024, 2025],
        val_ratio=0.5,
        random_seed=42,
        sample_n_races_per_year=10
    )
    
    assert len(df_train) > 0, "Train DataFrame is empty"
    assert len(df_val) > 0, "Val DataFrame is empty"
    assert len(df_test) > 0, "Test DataFrame is empty"
    assert "RACE_ID" in df_train.columns, "RACE_ID not in df_train"
    assert "距離" in df_train.columns, "BAC_KAB column '距離' not merged"
    logger.info(f"Data Loader Test Passed: Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}")

    # 2. 目的変数とメタデータの抽出テスト
    y_train, meta_train = extract_target_and_meta(df_train)
    y_val, meta_val = extract_target_and_meta(df_val)
    y_test, meta_test = extract_target_and_meta(df_test)

    assert len(y_train) == len(df_train), "y_train length mismatch"
    assert "Target_着順" in meta_train.columns, "Target_着順 not in meta_train"
    logger.info(f"Target & Meta Extraction Passed. Total winners in Train: {int(y_train.sum())}")

    # 3. 前処理 & 禁止カラム排除 & 10モデル特徴量サンプリングテスト
    artifacts_dir = "./test_artifacts"
    prohibited = [
        "KYI_枠確定馬体重増減", "KYI_基準オッズ", "KYI_基準人気順位",
        "KYI_基準複勝オッズ", "KYI_基準複勝人気順位", "KYI_IDM", "KYI_ＩＤＭ"
    ]
    id_cols = [
        "KYI_馬名", "KYI_血統登録番号", "SUBV1_RACE_ID", "SUBV1_RACE_KEY"
    ]
    
    preprocessor = FeaturePreprocessor(
        prohibited_columns=prohibited,
        id_and_str_columns=id_cols,
        bac_feature_cols=bac_cols,
        artifacts_dir=artifacts_dir
    )
    yearly_candidates = preprocessor.get_candidate_yearly_columns(df_train.columns.tolist())
    
    for p in prohibited:
        assert p not in yearly_candidates, f"Prohibited column {p} leaked into yearly_candidates!"
    for c in yearly_candidates:
        assert not c.startswith("Target_"), f"Target column {c} leaked into yearly_candidates!"

    logger.info(f"Prohibited Columns Excluded Successfully! Candidate yearly count: {len(yearly_candidates)}")

    # 10モデル分のランダムサンプリングとJSON保存テスト
    for idx in range(1, 11):
        feats = preprocessor.sample_features_for_model(
            model_idx=idx,
            yearly_candidate_cols=yearly_candidates,
            sample_ratio=0.35,
            random_seed=42
        )
        json_path = os.path.join(artifacts_dir, f"features_model_{idx}.json")
        assert os.path.exists(json_path), f"JSON for model {idx} not created!"
        saved_data = load_json(json_path)
        assert saved_data["model_idx"] == idx, f"Model index mismatch in {json_path}"
        assert len(saved_data["features"]) == len(feats), f"Features length mismatch in {json_path}"
        
        # 特徴量変換テスト
        X_tr = preprocessor.fit_transform_features(df_train, feats, model_idx=idx)
        X_te = preprocessor.transform_features(df_test, feats, model_idx=idx)
        assert X_tr.shape[1] == len(feats), "X_tr column mismatch"
        assert not np.isnan(X_tr).any(), "NaN found in transformed X_tr"
        assert not np.isnan(X_te).any(), "NaN found in transformed X_te"

    logger.info("Random Subspace (10 models) & Feature JSON persistence Test Passed!")

    # 4. 評価指標 & ベッティングシミュレーションのテスト
    dummy_test = meta_test.copy()
    rng = np.random.default_rng(42)
    dummy_test["prob"] = rng.uniform(0.01, 0.40, size=len(dummy_test))
    
    top1_res = calculate_top1_hit_rate(dummy_test)
    assert "hit_rate" in top1_res
    logger.info(f"Hit Rate Metric Test Passed: {top1_res}")

    simulator = BetSimulator(unit_bet=100)
    flat_res = simulator.simulate_flat_bet(dummy_test)
    assert "roi" in flat_res
    logger.info(f"Flat Bet Simulation Passed: ROI={flat_res['roi']:.4f}, Hit Rate={flat_res['hit_rate']:.4f}")

    ev_res = simulator.simulate_ev_threshold(dummy_test, ev_threshold=1.0, min_prob=0.1)
    assert "roi" in ev_res
    logger.info(f"EV Threshold Simulation Passed: {ev_res}")

    grid_res = simulator.grid_search_strategies(dummy_test, ev_grid=[0.9, 1.0, 1.1], prob_grid=[0.1, 0.2])
    assert len(grid_res) > 0
    logger.info(f"Grid Search Simulation Passed! Generated {len(grid_res)} strategy evaluations.")

    logger.info("\n>>> ALL PIPELINE COMPONENT TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_components()
