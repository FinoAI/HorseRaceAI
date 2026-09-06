import os
import json
import numpy as np
import pandas as pd
from src.utils.helpers import setup_logger, set_seed, load_json
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta
from src.evaluation.metrics import calculate_top1_hit_rate, calculate_payout_metric
from src.evaluation.simulation import BetSimulator

logger = setup_logger("TestPipeline")

def test_components():
    logger.info(">>> Starting Pipeline Components Test (including Payout Maximization Metric) <<<")
    
    # 1. データローダーのテスト (実データから各年5レース分をサンプリング)
    data_dir = "/Users/hinomasafumi/Data/main"
    logger.info("Testing Data Loader with sample_n_races_per_year=5...")
    
    df_train, df_val, df_test, bac_cols = get_train_val_test_datasets(
        data_dir=data_dir,
        bac_filename="BAC_KAB.csv",
        train_years=[2023],
        val_test_years=[2024, 2025],
        val_ratio=0.5,
        random_seed=42,
        sample_n_races_per_year=5
    )
    
    assert len(df_train) > 0, "Train DataFrame is empty"
    assert len(df_val) > 0, "Val DataFrame is empty"
    assert len(df_test) > 0, "Test DataFrame is empty"
    logger.info(f"Data Loader Test Passed: Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}")

    # 2. 目的変数とメタデータの抽出テスト
    y_train, meta_train = extract_target_and_meta(df_train)
    y_val, meta_val = extract_target_and_meta(df_val)
    y_test, meta_test = extract_target_and_meta(df_test)

    assert "Target_単勝" in meta_test.columns, "Target_単勝 not in meta_test"
    assert "Target_確定単勝オッズ" in meta_test.columns, "Target_確定単勝オッズ not in meta_test"

    # 3. 払戻金評価関数 (Target_単勝 * (Target_着順 == 1)) のテスト
    dummy_test = meta_test.copy()
    rng = np.random.default_rng(42)
    dummy_test["prob"] = rng.uniform(0.01, 0.40, size=len(dummy_test))

    # モード1: 予測勝率最大 (prob)
    res_prob = calculate_payout_metric(dummy_test, selection_mode="prob")
    assert "total_payout" in res_prob
    assert "roi" in res_prob
    logger.info(f"Payout Metric (prob): {res_prob}")

    # モード2: 期待値最大 (ev)
    res_ev = calculate_payout_metric(dummy_test, selection_mode="ev")
    assert "total_payout" in res_ev
    logger.info(f"Payout Metric (ev): {res_ev}")

    # モード3: フィルタ付き期待値最大 (ev_filtered)
    res_ev_filt = calculate_payout_metric(dummy_test, selection_mode="ev_filtered", min_prob=0.10)
    assert "total_payout" in res_ev_filt
    logger.info(f"Payout Metric (ev_filtered): {res_ev_filt}")

    # 着順が1以外の馬の払戻金が0になっているかの厳密チェック
    # 全馬着順2にしてテスト
    test_zero = dummy_test.copy()
    test_zero["Target_着順"] = 2
    res_zero = calculate_payout_metric(test_zero, selection_mode="prob")
    assert res_zero["hits"] == 0, "Hits should be 0"
    assert res_zero["total_payout"] == 0.0, f"Payout should be 0, got {res_zero['total_payout']}"
    assert res_zero["roi"] == 0.0, "ROI should be 0"
    logger.info("Zero Payout for Non-Winners Verified: Correctly returns 0.0 payout when order != 1")

    logger.info("\n>>> ALL PIPELINE COMPONENT TESTS (INCLUDING PAYOUT EVALUATION) PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_components()
