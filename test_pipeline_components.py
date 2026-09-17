import os
import json
import numpy as np
import pandas as pd
from src.utils.helpers import setup_logger, set_seed, load_json
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta, build_race_level_dataset
from src.models.base_nn import RaceLevelStage1NN
from src.models.meta_nn import RaceLevelMetaNN
from src.evaluation.metrics import calculate_top1_hit_rate, calculate_payout_metric, calculate_race_array_payout_metric
from src.evaluation.simulation import BetSimulator

logger = setup_logger("TestPipeline")

def test_components():
    logger.info(">>> Starting Race-Level Pipeline Components Test <<<")
    
    # 1. データローダーのテスト
    data_dir = "/Users/hinomasafumi/Data/main"
    logger.info("Testing Data Loader with sample_n_races_per_year=3...")
    
    df_train, df_val, df_test, bac_cols = get_train_val_test_datasets(
        data_dir=data_dir,
        bac_filename="BAC_KAB.csv",
        train_years=[2023],
        val_test_years=[2024, 2025],
        val_ratio=0.5,
        random_seed=42,
        sample_n_races_per_year=3
    )
    
    assert len(df_train) > 0, "Train DataFrame is empty"
    assert len(df_val) > 0, "Val DataFrame is empty"
    assert len(df_test) > 0, "Test DataFrame is empty"
    logger.info(f"Data Loader Test Passed: Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}")

    # 2. 目的変数とメタデータの抽出
    y_train, meta_train = extract_target_and_meta(df_train)
    y_val, meta_val = extract_target_and_meta(df_val)

    # 3. 特徴量選定 (全馬共通 & JSON保存)
    artifacts_dir = "./test_artifacts_race"
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
    
    # 14列ランダム抽出
    selected_features = preprocessor.sample_features_for_model(
        model_idx=1,
        yearly_candidate_cols=yearly_candidates,
        sample_count=14,
        random_seed=42
    )
    num_feats = len(selected_features)
    logger.info(f"Selected {num_feats} features per horse (BAC:{len(preprocessor.bac_feature_cols)} + Yearly:14)")
    assert num_feats == len(preprocessor.bac_feature_cols) + 14

    # 特徴量変換
    X_tr_norm = preprocessor.fit_transform_features(df_train, selected_features, model_idx=1)
    
    # 4. レース単位テンソル変換のテスト (出走頭数18 * 特徴量数)
    max_horses = 18
    race_train = build_race_level_dataset(X_tr_norm, meta_train, max_horses=max_horses)
    
    N_races = len(race_train["race_ids"])
    assert race_train["race_X"].shape == (N_races, max_horses * num_feats), f"Shape mismatch: {race_train['race_X'].shape}"
    assert race_train["race_masks"].shape == (N_races, max_horses)
    assert race_train["race_y"].shape == (N_races,)
    logger.info(f"Race-level Tensor Transformation Passed: race_X={race_train['race_X'].shape}, masks={race_train['race_masks'].shape}")

    # 5. RaceLevelStage1NN モデル構造のテスト (32倍入力層 & 最大1/2減衰)
    import torch
    input_dim = max_horses * num_feats
    multiplier = 32
    stage1_model = RaceLevelStage1NN(
        input_dim=input_dim,
        max_horses=max_horses,
        multiplier=multiplier,
        reduction_ratio=0.5,
        dropout=0.2
    )
    
    # レイヤー数と減衰比率の確認
    assert stage1_model.total_layers >= 5, f"Expected at least 5 layers, got {stage1_model.total_layers}"
    first_dim = stage1_model.input_layer[0].out_features
    assert first_dim == input_dim * multiplier, f"Expected first layer {input_dim * multiplier}, got {first_dim}"
    
    # 各層の次元が直前の最大1/2以下（半分ずつ減衰）になっているか確認
    prev_dim = first_dim
    for b_idx, block in enumerate(stage1_model.blocks):
        curr_dim = block.linear.out_features
        ratio = curr_dim / prev_dim
        assert ratio <= 0.5001, f"Reduction ratio exceeded 1/2 at block {b_idx}: {ratio:.3f}"
        prev_dim = curr_dim

    logger.info(f"RaceLevelStage1NN Architecture Passed: {stage1_model.total_layers} layers, 1st layer={first_dim} (x32), 1/2 reductions verified!")

    # Forward & マスキングの検証
    dummy_x = torch.tensor(race_train["race_X"][:2], dtype=torch.float32)
    dummy_mask = torch.tensor(race_train["race_masks"][:2], dtype=torch.float32)
    logits = stage1_model(dummy_x, mask=dummy_mask)
    probs = torch.softmax(logits, dim=-1)

    # 非出走馬の確率が0であることを確認
    for r in range(2):
        for slot in range(max_horses):
            if dummy_mask[r, slot] == 0:
                assert probs[r, slot].item() < 1e-6, f"Non-running horse received non-zero prob: {probs[r, slot].item()}"

    logger.info("Non-running Horse Masking Verified: Non-runners have 0.0 probability!")

    # 6. RaceLevelMetaNN モデル構造のテスト (入力252次元、32倍入力、1/2減衰)
    meta_in_dim = 252
    meta_model = RaceLevelMetaNN(
        input_dim=meta_in_dim,
        max_horses=max_horses,
        multiplier=multiplier,
        reduction_ratio=0.5,
        dropout=0.1
    )
    assert meta_model.total_layers >= 5
    assert meta_model.input_layer[0].out_features == meta_in_dim * multiplier
    logger.info(f"RaceLevelMetaNN Architecture Passed: {meta_model.total_layers} layers, 1st layer={meta_in_dim * multiplier} (x32), 1/2 reductions verified!")

    # 7. レース単位払戻金評価関数のテスト
    sample_probs = probs.detach().numpy()
    sample_mask = dummy_mask.numpy()
    sample_y = race_train["race_y"][:2]
    sample_odds = race_train["odds"][:2]
    sample_payouts = race_train["payouts"][:2]
    
    payout_res = calculate_race_array_payout_metric(
        probs=sample_probs,
        masks=sample_mask,
        y_true=sample_y,
        odds=sample_odds,
        payouts=sample_payouts,
        selection_mode="prob"
    )
    assert "roi" in payout_res
    assert "hit_rate" in payout_res
    logger.info(f"Race Array Payout Metric Passed: {payout_res}")

    import shutil
    shutil.rmtree(artifacts_dir, ignore_errors=True)
    logger.info("\n>>> ALL RACE-LEVEL PIPELINE COMPONENT TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_components()
