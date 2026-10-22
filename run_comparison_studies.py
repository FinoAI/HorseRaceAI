import os
import sys
import json
import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_race_array_payout_metric

def evaluate_subset_with_odds(subset: list, probs_array: np.ndarray, meta_dict: dict):
    """
    指定モデルサブセットの平均勝率で評価し、的中率・回収率・的中平均オッズを算出
    """
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

    # 的中平均オッズ: (総払戻金 / 100円) / 的中数 = 的中レースの確定単勝オッズ平均
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
    """
    候補モデルリスト candidates の中から、Val スコアが最大となる target_k 個のモデルを Greedy 探索
    """
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

def main():
    artifacts_dir = "./artifacts_models_100"
    val_cache_path = os.path.join(artifacts_dir, "all_val_probs.npy")
    test_cache_path = os.path.join(artifacts_dir, "all_test_probs.npy")
    meta_cache_path = os.path.join(artifacts_dir, "eval_meta.npz")
    single_perf_path = os.path.join(artifacts_dir, "single_models_performance.csv")

    print("=== Loading cached predictions and single model performance ===")
    all_val_probs = np.load(val_cache_path)
    all_test_probs = np.load(test_cache_path)
    meta_data = np.load(meta_cache_path, allow_pickle=True)
    df_single = pd.read_csv(single_perf_path)

    data_va = {k: meta_data[f"va_{k}"] for k in ["race_masks", "race_y", "odds", "payouts", "ranks", "race_ids"]}
    data_te = {k: meta_data[f"te_{k}"] for k in ["race_masks", "race_y", "odds", "payouts", "ranks", "race_ids"]}

    print(f"Data shapes: Val={all_val_probs.shape}, Test={all_test_probs.shape}")

    # 0-indexed に変換したモデルインデックス
    # 単体スコア上位順
    top_by_score = [int(m) - 1 for m in df_single.sort_values("val_score", ascending=False)["model_idx"].values]

    # 単体穴馬型モデル: 的中率は20%未満だがROIが高いモデル (高配当・穴狙い型)
    upset_candidates = [
        int(m) - 1 for m in df_single[df_single["val_hit_rate"] < 0.20].sort_values("val_roi", ascending=False)["model_idx"].values
    ]
    # 中位モデル (順位 31〜70位)
    mid_candidates = top_by_score[30:70]

    results_table = []

    # =========================================================================
    # 実験 1: 単体上位選抜 vs 中位穴馬型モデルの組み合わせ
    # =========================================================================
    print("\n>>> Experiment 1: Top Models vs. Upset/Mid Blend <<<")

    patterns_exp1 = {
        # (1) 単体スコア上位のみで固めた組み合わせ
        "Top 8 Models": top_by_score[:8],
        "Top 10 Models": top_by_score[:10],
        "Top 12 Models": top_by_score[:12],
        "Top 15 Models": top_by_score[:15],

        # (2) 単体上位 ＋ 中位穴馬型モデルのブレンド
        "Top 6 + Upset 4 (10 Models)": top_by_score[:6] + upset_candidates[:4],
        "Top 8 + Upset 4 (12 Models)": top_by_score[:8] + upset_candidates[:4],
        "Top 8 + Upset 7 (15 Models)": top_by_score[:8] + upset_candidates[:7],
        "Top 6 + Mid 4 (10 Models)": top_by_score[:6] + mid_candidates[:4],
        "Top 8 + Mid 6 (14 Models)": top_by_score[:8] + mid_candidates[:6],

        # (3) 穴馬型モデルのみで構成
        "Upset Only 10 Models": upset_candidates[:10],
        "Upset Only 14 Models": upset_candidates[:14],
    }

    for name, subset in patterns_exp1.items():
        v_res = evaluate_subset_with_odds(subset, all_val_probs, data_va)
        t_res = evaluate_subset_with_odds(subset, all_test_probs, data_te)
        results_table.append({
            "category": "1. Top vs Upset Blend",
            "pattern": name,
            "k": len(subset),
            "val_hit_rate": v_res["hit_rate"],
            "val_roi": v_res["roi"],
            "val_mean_odds": v_res["mean_hit_odds"],
            "test_hit_rate": t_res["hit_rate"],
            "test_roi": t_res["roi"],
            "test_mean_odds": t_res["mean_hit_odds"],
            "test_hits": t_res["hits"],
            "test_payout": t_res["total_payout"],
            "models_used": [m + 1 for m in subset]
        })

    # =========================================================================
    # 実験 2: 互いに重複しない（排他的な）独立アンサンブル
    # =========================================================================
    print("\n>>> Experiment 2: Mutually Exclusive (Disjoint) Ensembles <<<")

    # 系統A: 100モデル全体から最良の10モデルを Greedy 最適化
    all_models = list(range(100))
    system_A_models = greedy_optimize_subset(all_models, target_k=10, all_val_probs=all_val_probs, data_va=data_va)

    # 系統B: 系統Aを完全に除外した残りの90モデルから最良の10モデルを Greedy 最適化
    pool_B = [m for m in all_models if m not in system_A_models]
    system_B_models = greedy_optimize_subset(pool_B, target_k=10, all_val_probs=all_val_probs, data_va=data_va)

    # 系統C: 系統Aおよび系統Bの20モデルを完全に除外した残りの80モデルから最良の10モデルを Greedy 最適化
    pool_C = [m for m in pool_B if m not in system_B_models]
    system_C_models = greedy_optimize_subset(pool_C, target_k=10, all_val_probs=all_val_probs, data_va=data_va)

    # 検証: モデルの重複が 0 であることを確認
    assert len(set(system_A_models) & set(system_B_models)) == 0, "System A and B overlap!"
    assert len(set(system_A_models) & set(system_C_models)) == 0, "System A and C overlap!"
    assert len(set(system_B_models) & set(system_C_models)) == 0, "System B and C overlap!"

    # デュアルアンサンブル: 系統A + 系統B (20モデル合同)
    system_AB_models = system_A_models + system_B_models

    patterns_exp2 = {
        "Exclusive System A (10 Models)": system_A_models,
        "Exclusive System B (10 Models - No Overlap)": system_B_models,
        "Exclusive System C (10 Models - No Overlap)": system_C_models,
        "Combined Dual System A+B (20 Models)": system_AB_models,
        "Combined Triple System A+B+C (30 Models)": system_A_models + system_B_models + system_C_models
    }

    for name, subset in patterns_exp2.items():
        v_res = evaluate_subset_with_odds(subset, all_val_probs, data_va)
        t_res = evaluate_subset_with_odds(subset, all_test_probs, data_te)
        results_table.append({
            "category": "2. Exclusive Disjoint Ensembles",
            "pattern": name,
            "k": len(subset),
            "val_hit_rate": v_res["hit_rate"],
            "val_roi": v_res["roi"],
            "val_mean_odds": v_res["mean_hit_odds"],
            "test_hit_rate": t_res["hit_rate"],
            "test_roi": t_res["roi"],
            "test_mean_odds": t_res["mean_hit_odds"],
            "test_hits": t_res["hits"],
            "test_payout": t_res["total_payout"],
            "models_used": [m + 1 for m in subset]
        })

    # 結果 DataFrame 化
    df_results = pd.DataFrame(results_table)
    out_csv = os.path.join(artifacts_dir, "study_top_vs_upset_and_exclusive.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

    # 結果の表示
    print("\n" + "="*125)
    print("      ★ 単体上位 vs 中位穴馬ブレンド ＆ 排他独立アンサンブル (的中平均オッズ付き) 比較結果 ★")
    print("="*125)
    print("未見テストデータ: 3,455 レース (2024-2025年 | selection_mode: prob (EV考慮なし))")
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

    display_df = df_results[["category", "pattern", "k", "test_hit_rate", "test_hits", "test_mean_odds", "test_roi", "test_payout", "val_hit_rate", "val_roi", "val_mean_odds"]].copy()
    for col, fmt in formatters.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)

    print(display_df.to_string(index=False))
    print("="*125)

    # 系統Aと系統Bのモデル詳細
    print("\n【排他独立アンサンブルの構成モデル】")
    print(f"- 系統A (10モデル): {[m+1 for m in system_A_models]}")
    print(f"- 系統B (10モデル・完全独立): {[m+1 for m in system_B_models]}")
    print(f"- 系統C (10モデル・完全独立): {[m+1 for m in system_C_models]}")
    print(f"- 共通モデル数: A∩B = {len(set(system_A_models) & set(system_B_models))}個, A∩C = {len(set(system_A_models) & set(system_C_models))}個, B∩C = {len(set(system_B_models) & set(system_C_models))}個\n")

if __name__ == "__main__":
    main()
