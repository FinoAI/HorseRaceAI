import argparse
import os
import sys
import numpy as np
import pandas as pd

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta
from src.training.train_stage1 import train_single_stage1_model, predict_stage1_model
from src.training.train_stage2 import build_meta_features, train_meta_model, predict_meta_model
from src.evaluation.metrics import calculate_top1_hit_rate
from src.evaluation.simulation import BetSimulator

logger = setup_logger("MainPipeline")

def run(config_path: str, sample_races: int = None):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Using compute device: {device}")

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. データロード
    logger.info("=== Step 1: Loading Datasets (Train / Val / Test) ===")
    df_train, df_val, df_test, bac_feature_cols = get_train_val_test_datasets(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=cfg["data"]["train_years"],
        val_test_years=cfg["data"]["val_test_years"],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        sample_n_races_per_year=sample_races
    )

    # ターゲット・メタデータ抽出
    y_train, meta_train = extract_target_and_meta(df_train)
    y_val, meta_val = extract_target_and_meta(df_val)
    y_test, meta_test = extract_target_and_meta(df_test)

    # 前処理クラス初期化
    preprocessor = FeaturePreprocessor(
        prohibited_columns=cfg["feature_selection"]["prohibited_columns"],
        id_and_str_columns=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        bac_feature_cols=bac_feature_cols,
        artifacts_dir=artifacts_dir
    )
    yearly_candidates = preprocessor.get_candidate_yearly_columns(df_train.columns.tolist())
    logger.info(f"Yearly candidate features count: {len(yearly_candidates)}")
    logger.info(f"BAC features count: {len(preprocessor.bac_feature_cols)}")

    # 2. 前段10モデル (Stage 1) の学習 & 予測
    logger.info("=== Step 2: Training 10 Stage-1 Models (Random Subspace + 5-Layer Deep NN) ===")
    num_models = cfg["stage1"]["num_models"]
    val_preds_list = []
    test_preds_list = []

    for idx in range(1, num_models + 1):
        logger.info(f"\n>>> [Stage 1] Training Model {idx}/{num_models} <<<")
        # 特徴量サンプリング (BAC全量 + 開催年ランダム抽出) & JSON保存
        selected_features = preprocessor.sample_features_for_model(
            model_idx=idx,
            yearly_candidate_cols=yearly_candidates,
            sample_ratio=cfg["feature_selection"]["yearly_sample_ratio"],
            random_seed=cfg["data"]["random_seed"]
        )

        # 特徴量変換 (標準化)
        X_tr = preprocessor.fit_transform_features(df_train, selected_features, model_idx=idx)
        X_va = preprocessor.transform_features(df_val, selected_features, model_idx=idx)
        X_te = preprocessor.transform_features(df_test, selected_features, model_idx=idx)

        # モデル学習
        model, best_hit_rate = train_single_stage1_model(
            model_idx=idx,
            X_train=X_tr,
            y_train=y_train,
            race_ids_train=meta_train["RACE_ID"].values,
            X_val=X_va,
            y_val=y_val,
            race_ids_val=meta_val["RACE_ID"].values,
            multiplier=cfg["stage1"]["multiplier"],
            dropout=cfg["stage1"]["dropout"],
            learning_rate=cfg["stage1"]["learning_rate"],
            weight_decay=cfg["stage1"]["weight_decay"],
            epochs=cfg["stage1"]["epochs"],
            early_stopping_patience=cfg["stage1"]["early_stopping_patience"],
            artifacts_dir=artifacts_dir,
            device=device
        )

        # 予測確率の算出
        val_pred = predict_stage1_model(model, X_va, device=device)
        test_pred = predict_stage1_model(model, X_te, device=device)
        val_preds_list.append(val_pred)
        test_preds_list.append(test_pred)

    # 3. 後段メタNN (Stage 2) の学習 & 最終予測
    logger.info("\n=== Step 3: Training Stage-2 Meta NN (Ensemble LLM-like Stacking) ===")
    X_val_meta = build_meta_features(val_preds_list)
    X_test_meta = build_meta_features(test_preds_list)

    meta_model, meta_best_hit_rate = train_meta_model(
        X_val_meta=X_val_meta,
        y_val=y_val,
        race_ids_val=meta_val["RACE_ID"].values,
        multiplier=cfg["stage2"]["multiplier"],
        dropout=cfg["stage2"]["dropout"],
        learning_rate=cfg["stage2"]["learning_rate"],
        weight_decay=cfg["stage2"]["weight_decay"],
        epochs=cfg["stage2"]["epochs"],
        early_stopping_patience=cfg["stage2"]["early_stopping_patience"],
        artifacts_dir=artifacts_dir,
        device=device
    )

    # 4. テストデータでの最終予測とシミュレーション
    logger.info("\n=== Step 4: Final Test Evaluation & Betting Simulation ===")
    test_final_probs = predict_meta_model(
        meta_model,
        X_test_meta,
        meta_test["RACE_ID"].values,
        device=device
    )

    df_test_eval = meta_test.copy()
    df_test_eval["prob"] = test_final_probs

    # 予測結果CSVの保存
    preds_output_path = os.path.join(artifacts_dir, "test_predictions.csv")
    df_test_eval.to_csv(preds_output_path, index=False)
    logger.info(f"Saved test predictions to {preds_output_path}")

    # 的中率
    top1_metrics = calculate_top1_hit_rate(df_test_eval)
    logger.info(f"Test Top-1 Hit Rate: {top1_metrics['hit_rate']*100:.2f}% ({top1_metrics['hits']}/{top1_metrics['total_races']} races)")

    # ベッティングシミュレーション
    simulator = BetSimulator(unit_bet=cfg["simulation"]["unit_bet_amount"])
    flat_res = simulator.simulate_flat_bet(df_test_eval)
    logger.info(f"[Flat Bet] Hit Rate: {flat_res['hit_rate']*100:.2f}%, ROI: {flat_res['roi']*100:.2f}%")

    # グリッドサーチ評価
    df_sim_results = simulator.grid_search_strategies(
        df_test_eval,
        ev_grid=cfg["simulation"]["ev_threshold_grid"],
        prob_grid=[0.10, 0.15, 0.20, 0.25, 0.30]
    )
    sim_output_path = os.path.join(artifacts_dir, "simulation_results.csv")
    df_sim_results.to_csv(sim_output_path, index=False)
    logger.info(f"Saved simulation grid search results to {sim_output_path}")

    # 目標達成戦略（的中率 >= 20% かつ 回収率 >= 100%）の抽出
    target_hits = df_sim_results[
        (df_sim_results["hit_rate"] >= cfg["simulation"]["target_hit_rate"]) &
        (df_sim_results["roi"] >= cfg["simulation"]["target_roi"]) &
        (df_sim_results["num_bets"] >= 10) # 最低10レース以上
    ]

    print("\n" + "="*70)
    print("                ★ 競馬AI 最終評価結果サマリー ★")
    print("="*70)
    print(f"テスト対象レース数: {top1_metrics['total_races']} レース (2024-2025年シャッフル)")
    print(f"Top-1 単勝的中率:  {top1_metrics['hit_rate']*100:.2f}% (目標: 20%以上)")
    print(f"Top-1 ベタ買い回収率: {flat_res['roi']*100:.2f}%")
    print("-"*70)
    if len(target_hits) > 0:
        print("【目標達成 (的中率>=20% かつ 回収率>=100%) のベッティング戦略】:")
        print(target_hits[["strategy", "num_bets", "hits", "hit_rate", "roi"]].to_string(index=False))
    else:
        print("※指定条件（的中率20%超 & 回収率100%超）の上位戦略:")
        top_candidates = df_sim_results.sort_values("roi", ascending=False).head(5)
        print(top_candidates[["strategy", "num_bets", "hits", "hit_rate", "roi"]].to_string(index=False))
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬AI パイプライン")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--sample_races", type=int, default=None, help="疎通確認用の1年あたりサンプリングレース数")
    args = parser.parse_args()
    run(args.config, sample_races=args.sample_races)
