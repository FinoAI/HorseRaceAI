import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from run_search_ensemble import prepare_precomputed_data, train_and_eval_single_model
from run_comparison_studies import evaluate_subset_with_odds

logger = setup_logger("Model101")

def main():
    artifacts_dir = "./artifacts_models_100"
    config_path = "configs/config.yaml"
    cfg = load_config(config_path)
    set_seed(cfg["data"]["random_seed"] + 101)
    device = get_device()
    logger.info(f"Compute Device: {device}")

    # 1. 未採用10特徴量の指定
    must_have_features = [
        "SUBV2_前３Ｆ先頭差",
        "SUBV2_天候コード",
        "SUBV2_季節成績_2",
        "SUBV2_騎手距離成績_2",
        "SUBV3_馬具コード2",
        "SUBV3_テンＦ",
        "SUBV4_蹄状態",
        "SUBV4_ローテ成績_4",
        "SUBV5_クラス",
        "main_芝ダ障害別距離成績_1"
    ]

    # 2. データ準備
    data_dict = prepare_precomputed_data(
        data_dir=cfg["data"]["data_dir"],
        bac_filename=cfg["data"]["bac_filename"],
        train_years=cfg["data"]["train_years"],
        val_test_years=cfg["data"]["val_test_years"],
        val_ratio=cfg["data"]["val_ratio"],
        random_seed=cfg["data"]["random_seed"],
        prohibited_cols=cfg["feature_selection"]["prohibited_columns"],
        id_cols=cfg["feature_selection"]["id_and_str_columns_to_drop"],
        max_horses=18
    )

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    yearly_candidates = data_dict["yearly_candidate_cols"]
    col_to_idx = data_dict["col_to_idx"]

    # must_have_features の存在確認
    for f in must_have_features:
        if f not in col_to_idx:
            raise ValueError(f"Feature '{f}' not found in candidate features!")

    must_have_indices = [col_to_idx[f] for f in must_have_features]

    # 3. 特徴量の構成: BAC 36列 + 未採用10列 + 残り54列 (計100列)
    TOTAL_FEATURES = 100
    REMAINING_COUNT = TOTAL_FEATURES - len(bac_col_indices) - len(must_have_indices) # 100 - 36 - 10 = 54
    available_yearly = [c for c in yearly_candidates if c not in must_have_features]

    rng = random.Random(cfg["data"]["random_seed"] + 101 * 13)
    sampled_remaining = rng.sample(available_yearly, REMAINING_COUNT)
    sampled_remaining_indices = [col_to_idx[c] for c in sampled_remaining]

    selected_yearly_features = must_have_features + sampled_remaining
    selected_indices = list(bac_col_indices) + must_have_indices + sampled_remaining_indices

    logger.info(f"Model 101 Feature Breakdown:")
    logger.info(f"- BAC_KAB Features: {len(bac_col_indices)} cols")
    logger.info(f"- Specific Unused Features: {len(must_have_indices)} cols: {must_have_features}")
    logger.info(f"- Random Sampled Yearly Features: {len(sampled_remaining_indices)} cols")
    logger.info(f"- Total Features: {len(selected_indices)} cols")

    # 特徴量リストを JSON 保存
    feat_meta = {
        "model_idx": 101,
        "num_features": len(selected_indices),
        "bac_features_count": len(bac_col_indices),
        "specific_unused_features": must_have_features,
        "specific_unused_count": len(must_have_indices),
        "random_yearly_count": len(sampled_remaining),
        "yearly_features_count": len(selected_yearly_features),
        "yearly_features": selected_yearly_features
    }
    feat_json_path = os.path.join(artifacts_dir, "features_101.json")
    with open(feat_json_path, "w", encoding="utf-8") as f:
        json.dump(feat_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved feature definition to {feat_json_path}")

    # 4. Model 101 の学習実行
    logger.info("=== Training Model 101 ===")
    t0 = time.time()
    res = train_and_eval_single_model(
        model_idx=101,
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
    logger.info(f"Model 101 Training Completed in {dt:.1f}s")

    val_probs_101 = res["val_probs"]   # (N_val, 18)
    test_probs_101 = res["test_probs"] # (N_test, 18)

    # 単体成績評価 (的中平均オッズ含む)
    val_eval = evaluate_subset_with_odds([0], val_probs_101[np.newaxis, ...], data_va)
    test_eval = evaluate_subset_with_odds([0], test_probs_101[np.newaxis, ...], data_te)

    # 5. 既存キャッシュの更新 (all_val_probs.npy, all_test_probs.npy を101モデルに拡張)
    val_cache_path = os.path.join(artifacts_dir, "all_val_probs.npy")
    test_cache_path = os.path.join(artifacts_dir, "all_test_probs.npy")

    if os.path.exists(val_cache_path) and os.path.exists(test_cache_path):
        all_val = np.load(val_cache_path)
        all_test = np.load(test_cache_path)

        if all_val.shape[0] == 100:
            all_val_101 = np.vstack([all_val, val_probs_101[np.newaxis, ...]])
            all_test_101 = np.vstack([all_test, test_probs_101[np.newaxis, ...]])
        else:
            all_val_101 = all_val
            all_val_101[100] = val_probs_101
            all_test_101 = all_test
            all_test_101[100] = test_probs_101

        np.save(val_cache_path, all_val_101)
        np.save(test_cache_path, all_test_101)
        logger.info(f"Updated predictions cache to 101 models: Val={all_val_101.shape}, Test={all_test_101.shape}")

    # single_models_performance.csv の更新
    single_perf_path = os.path.join(artifacts_dir, "single_models_performance.csv")
    df_perf = pd.read_csv(single_perf_path)
    new_row = {
        "model_idx": 101,
        "val_hit_rate": val_eval["hit_rate"],
        "val_roi": val_eval["roi"],
        "val_score": val_eval["score"],
        "test_hit_rate": test_eval["hit_rate"],
        "test_roi": test_eval["roi"],
        "time_sec": dt
    }
    df_perf = df_perf[df_perf["model_idx"] != 101]
    df_perf = pd.concat([df_perf, pd.DataFrame([new_row])], ignore_index=True)
    df_perf.to_csv(single_perf_path, index=False)

    # 6. Model 101 を組み込んだアンサンブル効果の検証
    # 例: 以前のベストアンサンブル (GA Best k=8, k=9, 系統A, 系統C) に Model 101 を追加した場合の比較
    study_csv_path = os.path.join(artifacts_dir, "advanced_ensemble_comparison.csv")
    df_adv = pd.read_csv(study_csv_path)

    # GA Best k=8: [11, 26, 27, 49, 65, 74, 95, 99] (0-indexed: [10, 25, 26, 48, 64, 73, 94, 98])
    base_ga8 = [10, 25, 26, 48, 64, 73, 94, 98]
    base_ga8_plus_101 = base_ga8 + [100]

    # GA Best k=9: [11, 16, 19, 32, 47, 59, 74, 78, 95]
    base_ga9 = [10, 15, 18, 31, 46, 58, 73, 77, 94]
    base_ga9_plus_101 = base_ga9 + [100]

    # 系統C (10モデル): [19, 62, 85, 84, 35, 86, 14, 54, 32, 74]
    sys_c = [18, 61, 84, 83, 34, 85, 13, 53, 31, 73]
    sys_c_plus_101 = sys_c + [100]

    eval_ga8_plus = evaluate_subset_with_odds(base_ga8_plus_101, all_test_101, data_te)
    eval_ga9_plus = evaluate_subset_with_odds(base_ga9_plus_101, all_test_101, data_te)
    eval_sys_c_plus = evaluate_subset_with_odds(sys_c_plus_101, all_test_101, data_te)

    print("\n" + "="*100)
    print("                 ★ Model 101 (未採用10特徴量含有モデル) 単体成績サマリー ★")
    print("="*100)
    print(f"モデルID: Model 101")
    print(f"特徴量数: 100列 (BAC_KAB: 36列 ＋ 未採用10列 ＋ 開催年ランダム: 54列)")
    print(f"含有未採用特徴量: {must_have_features}")
    print("-"*100)
    print(f"【検証データ (3,454レース)】:")
    print(f"- 的中率: {val_eval['hit_rate']*100:.2f}% | 回収率: {val_eval['roi']*100:.2f}% | 的中平均オッズ: {val_eval['mean_hit_odds']:.2f}倍 (的中数: {val_eval['hits']})")
    print(f"【テストデータ (3,455レース)】:")
    print(f"- 的中率: {test_eval['hit_rate']*100:.2f}% | 回収率: {test_eval['roi']*100:.2f}% | 的中平均オッズ: {test_eval['mean_hit_odds']:.2f}倍 (的中数: {test_eval['hits']}, 払戻: ¥{test_eval['total_payout']:,.0f})")
    print("="*100)

    print("\n" + "="*100)
    print("          ★ Model 101 を主要アンサンブルに追加・ブレンドした場合の比較 ★")
    print("="*100)
    print("未見テストデータ: 3,455レース (EV考慮なし)")
    print("-"*100)
    print(f"1. GA Best k=8 単独             : 的中率 28.48% | 回収率 80.31% | 平均オッズ 2.82倍 (的中 984, 払戻 ¥277,470)")
    print(f"   GA Best k=8 + Model 101 (9基) : 的中率 {eval_ga8_plus['hit_rate']*100:.2f}% | 回収率 {eval_ga8_plus['roi']*100:.2f}% | 平均オッズ {eval_ga8_plus['mean_hit_odds']:.2f}倍 (的中 {eval_ga8_plus['hits']}, 払戻 ¥{eval_ga8_plus['total_payout']:,.0f})")
    print("-"*100)
    print(f"2. GA Best k=9 (別系統) 単独    : 的中率 28.19% | 回収率 81.40% | 平均オッズ 2.89倍 (的中 974, 払戻 ¥281,230)")
    print(f"   GA Best k=9 + Model 101 (10基): 的中率 {eval_ga9_plus['hit_rate']*100:.2f}% | 回収率 {eval_ga9_plus['roi']*100:.2f}% | 平均オッズ {eval_ga9_plus['mean_hit_odds']:.2f}倍 (的中 {eval_ga9_plus['hits']}, 払戻 ¥{eval_ga9_plus['total_payout']:,.0f})")
    print("-"*100)
    print(f"3. 系統C (最高回収率) 単独       : 的中率 28.57% | 回収率 84.67% | 平均オッズ 2.96倍 (的中 987, 払戻 ¥292,530)")
    print(f"   系統C + Model 101 (11基)     : 的中率 {eval_sys_c_plus['hit_rate']*100:.2f}% | 回収率 {eval_sys_c_plus['roi']*100:.2f}% | 平均オッズ {eval_sys_c_plus['mean_hit_odds']:.2f}倍 (的中 {eval_sys_c_plus['hits']}, 払戻 ¥{eval_sys_c_plus['total_payout']:,.0f})")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
