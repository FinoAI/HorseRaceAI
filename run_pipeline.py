import argparse
import os
import sys
import numpy as np
import pandas as pd

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import FeaturePreprocessor, extract_target_and_meta, build_race_level_dataset
from src.training.train_stage1 import train_single_stage1_model, predict_race_stage1
from src.training.train_stage2 import build_race_meta_features, train_race_meta_model, predict_race_meta_model
from src.evaluation.metrics import calculate_top1_hit_rate, calculate_payout_metric, calculate_race_array_payout_metric
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
    max_horses = cfg["data"].get("max_horses_per_race", 18)

    logger.info(f"Evaluation Config: metric={eval_metric}, selection_mode={selection_mode}, min_prob={min_prob_for_ev}, loss_weighting={loss_weighting}")

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

    # Stage 2 での過学習・データリークを防ぐため、Val データをレース単位でさらに 2 分割
    val_races = np.array(df_val["RACE_ID"].drop_duplicates().tolist())
    rng = np.random.default_rng(cfg["data"]["random_seed"])
    rng.shuffle(val_races)
    n_half = max(1, len(val_races) // 2)
    val_tr_races = set(val_races[:n_half])
    val_ev_races = set(val_races[n_half:])

    df_val_tr = df_val[df_val["RACE_ID"].isin(val_tr_races)].reset_index(drop=True)
    df_val_ev = df_val[df_val["RACE_ID"].isin(val_ev_races)].reset_index(drop=True)

    logger.info(f"Val split for Stage 2: Stage2-Train={len(val_tr_races)} races, Stage2-Eval={len(val_ev_races)} races")

    # メタデータ抽出
    y_train_raw, meta_train = extract_target_and_meta(df_train)
    y_val_tr_raw, meta_val_tr = extract_target_and_meta(df_val_tr)
    y_val_ev_raw, meta_val_ev = extract_target_and_meta(df_val_ev)
    y_test_raw, meta_test = extract_target_and_meta(df_test)

    preprocessor = FeaturePreprocessor(
        prohibited_columns=cfg["feature_selection"]["prohibited_columns"],
        id_and_str_columns=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        bac_feature_cols=bac_feature_cols,
        artifacts_dir=artifacts_dir
    )
    yearly_candidates = preprocessor.get_candidate_yearly_columns(df_train.columns.tolist())
    sample_count = cfg["feature_selection"].get("num_random_yearly_features", 14)
    logger.info(f"Candidate yearly features: {len(yearly_candidates)} | BAC features: {len(preprocessor.bac_feature_cols)} | Random yearly per model: {sample_count}")

    # 2. 前段10モデル (Stage 1: レース単位入力) の学習 & 予測
    logger.info("\n=== Step 2: Training 10 Stage-1 Models (Race-Level Inputs, 32x Multiplier, 1/2 Reductions) ===")
    num_models = cfg["stage1"]["num_models"]
    val_tr_preds_list = []
    val_ev_preds_list = []
    test_preds_list = []
    last_race_test = None

    for idx in range(1, num_models + 1):
        logger.info(f"\n>>> [Stage 1] Training Model {idx}/{num_models} <<<")
        # 特徴量サンプリング (全馬で同一の特徴量セットを使用)
        selected_features = preprocessor.sample_features_for_model(
            model_idx=idx,
            yearly_candidate_cols=yearly_candidates,
            sample_count=sample_count,
            random_seed=cfg["data"]["random_seed"]
        )

        # 特徴量正規化
        X_tr_norm = preprocessor.fit_transform_features(df_train, selected_features, model_idx=idx)
        X_va_tr_norm = preprocessor.transform_features(df_val_tr, selected_features, model_idx=idx)
        X_va_ev_norm = preprocessor.transform_features(df_val_ev, selected_features, model_idx=idx)
        X_te_norm = preprocessor.transform_features(df_test, selected_features, model_idx=idx)

        # レース単位テンソル表現へ変換 (N_races, 18 * D)
        race_train = build_race_level_dataset(X_tr_norm, meta_train, max_horses=max_horses)
        race_val_tr = build_race_level_dataset(X_va_tr_norm, meta_val_tr, max_horses=max_horses)
        race_val_ev = build_race_level_dataset(X_va_ev_norm, meta_val_ev, max_horses=max_horses)
        race_test = build_race_level_dataset(X_te_norm, meta_test, max_horses=max_horses)
        last_race_test = race_test

        # モデル学習 (5段以上、第1層入力*32倍、最大1/2ずつ減衰)
        model, best_score = train_single_stage1_model(
            model_idx=idx,
            race_train=race_train,
            race_val=race_val_ev,
            max_horses=max_horses,
            multiplier=cfg["stage1"]["multiplier"],
            reduction_ratio=cfg["stage1"].get("reduction_ratio", 0.5),
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

        val_tr_pred = predict_race_stage1(model, race_val_tr["race_X"], race_val_tr["race_masks"], device=device)
        val_ev_pred = predict_race_stage1(model, race_val_ev["race_X"], race_val_ev["race_masks"], device=device)
        test_pred = predict_race_stage1(model, race_test["race_X"], race_test["race_masks"], device=device)

        val_tr_preds_list.append(val_tr_pred)
        val_ev_preds_list.append(val_ev_pred)
        test_preds_list.append(test_pred)

    # 3. 後段メタNN (Stage 2: レース単位アンサンブル) の学習 & 最終予測
    logger.info("\n=== Step 3: Training Stage-2 Meta NN (Race-Level Stacking Ensemble) ===")
    X_val_tr_meta = build_race_meta_features(val_tr_preds_list)
    X_val_ev_meta = build_race_meta_features(val_ev_preds_list)
    X_test_meta = build_race_meta_features(test_preds_list)

    meta_model, meta_best_score = train_race_meta_model(
        X_train_meta=X_val_tr_meta,
        race_train=race_val_tr,
        X_val_meta=X_val_ev_meta,
        race_val=race_val_ev,
        max_horses=max_horses,
        multiplier=cfg["stage2"]["multiplier"],
        reduction_ratio=cfg["stage2"].get("reduction_ratio", 0.5),
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
    test_final_probs = predict_race_meta_model(
        meta_model,
        X_test_meta,
        last_race_test["race_masks"],
        device=device
    )

    # レース単位の予測確率を行形式のDataFrameに展開
    eval_rows = []
    for r_idx, r_id in enumerate(last_race_test["race_ids"]):
        mask = last_race_test["race_masks"][r_idx]
        probs = test_final_probs[r_idx]
        odds = last_race_test["odds"][r_idx]
        payouts = last_race_test["payouts"][r_idx]
        ranks = last_race_test["ranks"][r_idx]
        
        for slot in range(max_horses):
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

    df_test_eval = pd.DataFrame(eval_rows)
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

    print("\n" + "="*80)
    print("                ★ 競馬AI レース単位モデル 最終評価結果サマリー ★")
    print("="*80)
    print(f"テスト対象レース数: {eval_prob['total_races']} レース (2024-2025年シャッフル)")
    print("-"*80)
    print("【各馬券選定方式における成績比較 (払戻金額 = Target_単勝 * (着順==1))】")
    print(f"1. 予測勝率最大 (prob)       : 的中率 {eval_prob['hit_rate']*100:5.2f}% | 回収率 {eval_prob['roi']*100:6.2f}% (払戻: ¥{eval_prob['total_payout']:,.0f})")
    print(f"2. 期待値最大 (ev - 穴狙い)   : 的中率 {eval_ev['hit_rate']*100:5.2f}% | 回収率 {eval_ev['roi']*100:6.2f}% (払戻: ¥{eval_ev['total_payout']:,.0f})")
    print(f"3. フィルタ付き期待値最大     : 的中率 {eval_ev_filt['hit_rate']*100:5.2f}% | 回収率 {eval_ev_filt['roi']*100:6.2f}% (払戻: ¥{eval_ev_filt['total_payout']:,.0f})")
    print(f"   (勝率 >= {min_prob_for_ev*100:.0f}% かつ EV最大 ★推奨)")
    print("-"*80)
    
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
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="競馬AI レース単位パイプライン")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="設定ファイルのパス")
    parser.add_argument("--sample_races", type=int, default=None, help="疎通確認用の1年あたりサンプリングレース数")
    args = parser.parse_args()
    run(args.config, sample_races=args.sample_races)
