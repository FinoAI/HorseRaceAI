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
from src.evaluation.metrics import calculate_top1_hit_rate, calculate_payout_metric
from src.evaluation.simulation import BetSimulator

logger = setup_logger("MainPipeline")

def run(config_path: str, sample_races: int = None):
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"])
    device = get_device()
    logger.info(f"Using compute device: {device}")

    eval_cfg = cfg.get("evaluation", {})
    eval_metric = eval_cfg.get("eval_metric", "payout")
    selection_mode = eval_cfg.get("selection_mode", "ev_filtered")
    min_prob_for_ev = eval_cfg.get("min_prob_for_ev", 0.10)
    loss_weighting = eval_cfg.get("loss_weighting", "none")

    logger.info(f"Evaluation Config: metric={eval_metric}, selection_mode={selection_mode}, min_prob={min_prob_for_ev}, loss_weighting={loss_weighting}")

    artifacts_dir = cfg["paths"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. データロード (Train: 過去年, Val/Test: 2024+2025 シャッフル)
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

    # Stage 2 での過学習・データリークを防ぐため、Val データをレース単位でさらに 2 分割:
    # - df_val_tr (Stage 2 メタモデルの学習用)
    # - df_val_ev (Stage 2 メタモデルの Early Stopping 評価用)
    val_races = np.array(df_val["RACE_ID"].drop_duplicates().tolist())
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(val_races)
    n_half = max(1, len(val_races) // 2)
    val_tr_races = set(val_races[:n_half])
    val_ev_races = set(val_races[n_half:])

    df_val_tr = df_val[df_val["RACE_ID"].isin(val_tr_races)].reset_index(drop=True)
    df_val_ev = df_val[df_val["RACE_ID"].isin(val_ev_races)].reset_index(drop=True)

    logger.info(f"Val split for Stage 2: Stage2-Train={df_val_tr.shape} ({len(val_tr_races)} races), Stage2-Eval={df_val_ev.shape} ({len(val_ev_races)} races)")

    # ターゲット・メタデータ抽出
    y_train, meta_train = extract_target_and_meta(df_train)
    y_val_tr, meta_val_tr = extract_target_and_meta(df_val_tr)
    y_val_ev, meta_val_ev = extract_target_and_meta(df_val_ev)
    y_test, meta_test = extract_target_and_meta(df_test)

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
    val_tr_preds_list = []
    val_ev_preds_list = []
    test_preds_list = []

    for idx in range(1, num_models + 1):
        logger.info(f"\n>>> [Stage 1] Training Model {idx}/{num_models} <<<")
        selected_features = preprocessor.sample_features_for_model(
            model_idx=idx,
            yearly_candidate_cols=yearly_candidates,
            sample_ratio=cfg["feature_selection"]["yearly_sample_ratio"],
            random_seed=cfg["data"]["random_seed"]
        )

        X_tr = preprocessor.fit_transform_features(df_train, selected_features, model_idx=idx)
        X_va_tr = preprocessor.transform_features(df_val_tr, selected_features, model_idx=idx)
        X_va_ev = preprocessor.transform_features(df_val_ev, selected_features, model_idx=idx)
        X_te = preprocessor.transform_features(df_test, selected_features, model_idx=idx)

        # Stage 1 は df_train で学習し、df_val_ev で Early Stopping 判定
        model, best_score = train_single_stage1_model(
            model_idx=idx,
            X_train=X_tr,
            y_train=y_train,
            meta_train=meta_train,
            X_val=X_va_ev,
            y_val=y_val_ev,
            meta_val=meta_val_ev,
            multiplier=cfg["stage1"]["multiplier"],
            dropout=cfg["stage1"]["dropout"],
            learning_rate=cfg["stage1"]["learning_rate"],
            weight_decay=cfg["stage1"]["weight_decay"],
            epochs=cfg["stage1"]["epochs"],
            early_stopping_patience=cfg["stage1"]["early_stopping_patience"],
            eval_metric=eval_metric,
            selection_mode=selection_mode,
            min_prob_for_ev=min_prob_for_ev,
            loss_weighting=loss_weighting,
            artifacts_dir=artifacts_dir,
            device=device
        )

        val_tr_pred = predict_stage1_model(model, X_va_tr, device=device)
        val_ev_pred = predict_stage1_model(model, X_va_ev, device=device)
        test_pred = predict_stage1_model(model, X_te, device=device)

        val_tr_preds_list.append(val_tr_pred)
        val_ev_preds_list.append(val_ev_pred)
        test_preds_list.append(test_pred)

    # 3. 後段メタNN (Stage 2) の学習 & 最終予測
    logger.info("\n=== Step 3: Training Stage-2 Meta NN (Ensemble LLM-like Stacking) ===")
    X_val_tr_meta = build_meta_features(val_tr_preds_list)
    X_val_ev_meta = build_meta_features(val_ev_preds_list)
    X_test_meta = build_meta_features(test_preds_list)

    # Stage 2 は X_val_tr_meta で学習し、独立した X_val_ev_meta で Early Stopping (データリーク解消)
    meta_model, meta_best_score = train_meta_model(
        X_train_meta=X_val_tr_meta,
        y_train=y_val_tr,
        meta_train=meta_val_tr,
        X_val_meta=X_val_ev_meta,
        y_val=y_val_ev,
        meta_val=meta_val_ev,
        multiplier=cfg["stage2"]["multiplier"],
        dropout=cfg["stage2"]["dropout"],
        learning_rate=cfg["stage2"]["learning_rate"],
        weight_decay=cfg["stage2"]["weight_decay"],
        epochs=cfg["stage2"]["epochs"],
        early_stopping_patience=cfg["stage2"]["early_stopping_patience"],
        eval_metric=eval_metric,
        selection_mode=selection_mode,
        min_prob_for_ev=min_prob_for_ev,
        loss_weighting=loss_weighting,
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

    preds_output_path = os.path.join(artifacts_dir, "test_predictions.csv")
    df_test_eval.to_csv(preds_output_path, index=False)
    logger.info(f"Saved test predictions to {preds_output_path}")

    # 各種戦略での払戻金（Target_単勝 * 着順1）評価
    eval_prob = calculate_payout_metric(df_test_eval, selection_mode="prob")
    eval_ev = calculate_payout_metric(df_test_eval, selection_mode="ev")
    eval_ev_filt = calculate_payout_metric(df_test_eval, selection_mode="ev_filtered", min_prob=min_prob_for_ev)

    simulator = BetSimulator(unit_bet=cfg["simulation"]["unit_bet_amount"])
    df_sim_results = simulator.grid_search_strategies(
        df_test_eval,
        ev_grid=cfg["simulation"]["ev_threshold_grid"],
        prob_grid=[0.10, 0.15, 0.20, 0.25, 0.30]
    )
    sim_output_path = os.path.join(artifacts_dir, "simulation_results.csv")
    df_sim_results.to_csv(sim_output_path, index=False)

    print("\n" + "="*75)
    print("                ★ 競馬AI 払戻金最大化 & 評価結果サマリー ★")
    print("="*75)
    print(f"テスト対象レース数: {eval_prob['total_races']} レース (2024-2025年シャッフル)")
    print("-"*75)
    print("【各馬券選定方式における成績比較 (払戻金額 = Target_単勝 * (着順==1))】")
    print(f"1. 予測勝率最大 (prob)       : 的中率 {eval_prob['hit_rate']*100:5.2f}% | 回収率 {eval_prob['roi']*100:6.2f}% (払戻: ¥{eval_prob['total_payout']:,.0f})")
    print(f"2. 期待値最大 (ev - 穴狙い)   : 的中率 {eval_ev['hit_rate']*100:5.2f}% | 回収率 {eval_ev['roi']*100:6.2f}% (払戻: ¥{eval_ev['total_payout']:,.0f})")
    print(f"3. フィルタ付き期待値最大     : 的中率 {eval_ev_filt['hit_rate']*100:5.2f}% | 回収率 {eval_ev_filt['roi']*100:6.2f}% (払戻: ¥{eval_ev_filt['total_payout']:,.0f})")
    print(f"   (勝率 >= {min_prob_for_ev*100:.0f}% かつ EV最大 ★推奨)")
    print("-"*75)
    
    target_hits = df_sim_results[
        (df_sim_results["hit_rate"] >= cfg["simulation"]["target_hit_rate"]) &
        (df_sim_results["roi"] >= cfg["simulation"]["target_roi"]) &
        (df_sim_results["num_bets"] >= 10)
    ]
    if len(target_hits) > 0:
        print("【目標達成 (的中率>=20% かつ 回収率>=100%) のベッティング戦略】:")
        print(target_hits[["strategy", "num_bets", "hits", "hit_rate", "roi"]].to_string(index=False))
    else:
        print("※高回収率上位戦略 (EVフィルタリング):")
        top_candidates = df_sim_results.sort_values("roi", ascending=False).head(5)
        print(top_candidates[["strategy", "num_bets", "hits", "hit_rate", "roi"]].to_string(index=False))
    print("="*75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬AI パイプライン")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--sample_races", type=int, default=None, help="疎通確認用の1年あたりサンプリングレース数")
    args = parser.parse_args()
    run(args.config, sample_races=args.sample_races)
