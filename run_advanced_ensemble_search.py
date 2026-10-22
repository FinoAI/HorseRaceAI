import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from src.utils.helpers import setup_logger, set_seed, load_config, get_device
from src.data.loader import get_train_val_test_datasets
from src.data.preprocessor import extract_target_and_meta
from src.models.base_nn import RaceLevelStage1NN
from src.evaluation.metrics import calculate_race_array_payout_metric

logger = setup_logger("AdvancedEnsembleSearch")

def load_or_cache_model_predictions(
    artifacts_dir: str = "./artifacts_models_100",
    config_path: str = "configs/config.yaml",
    device: torch.device = None
):
    """
    100モデルの予測勝率が保存されていなければ推論して .npy に保存し、
    保存済みであれば即座にロードして返す。
    """
    val_cache_path = os.path.join(artifacts_dir, "all_val_probs.npy")
    test_cache_path = os.path.join(artifacts_dir, "all_test_probs.npy")
    meta_cache_path = os.path.join(artifacts_dir, "eval_meta.npz")

    if os.path.exists(val_cache_path) and os.path.exists(test_cache_path) and os.path.exists(meta_cache_path):
        logger.info(f"Loading cached predictions from {artifacts_dir}...")
        all_val_probs = np.load(val_cache_path)
        all_test_probs = np.load(test_cache_path)
        meta_data = np.load(meta_cache_path, allow_pickle=True)
        data_va = {k: meta_data[f"va_{k}"] for k in ["race_masks", "race_y", "odds", "payouts", "ranks", "race_ids"]}
        data_te = {k: meta_data[f"te_{k}"] for k in ["race_masks", "race_y", "odds", "payouts", "ranks", "race_ids"]}
        logger.info(f"Loaded cached predictions: Val={all_val_probs.shape}, Test={all_test_probs.shape}")
        return all_val_probs, all_test_probs, data_va, data_te

    # キャッシュがない場合はデータをロードして100モデルの推論を実行
    logger.info("Cache not found. Generating predictions from 100 trained models...")
    cfg = load_config(config_path)
    if device is None:
        device = get_device()

    from run_search_ensemble import prepare_precomputed_data

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

    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]
    bac_col_indices = data_dict["bac_col_indices"]
    col_to_idx = data_dict["col_to_idx"]

    all_val_probs = []
    all_test_probs = []
    TOTAL_MODELS = 100
    max_horses = 18

    t0 = time.time()
    for m in range(1, TOTAL_MODELS + 1):
        feat_path = os.path.join(artifacts_dir, f"features_{m}.json")
        model_path = os.path.join(artifacts_dir, f"model_{m}.pth")

        with open(feat_path, "r", encoding="utf-8") as f:
            feat_info = json.load(f)

        yearly_features = feat_info["yearly_features"]
        selected_indices = list(bac_col_indices) + [col_to_idx[c] for c in yearly_features]
        num_features = len(selected_indices)
        input_dim = max_horses * num_features

        X_va = data_va["race_3d"][:, :, selected_indices].reshape(-1, input_dim)
        mask_va = data_va["race_masks"]
        X_te = data_te["race_3d"][:, :, selected_indices].reshape(-1, input_dim)
        mask_te = data_te["race_masks"]

        X_va_gpu = torch.tensor(X_va, dtype=torch.float32, device=device)
        mask_va_gpu = torch.tensor(mask_va, dtype=torch.float32, device=device)
        X_te_gpu = torch.tensor(X_te, dtype=torch.float32, device=device)
        mask_te_gpu = torch.tensor(mask_te, dtype=torch.float32, device=device)

        model = RaceLevelStage1NN(
            input_dim=input_dim,
            max_horses=max_horses,
            multiplier=2,
            reduction_ratio=0.5,
            dropout=0.15
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            val_logits = model(X_va_gpu, mask=mask_va_gpu)
            val_probs = torch.softmax(val_logits, dim=-1).cpu().numpy()

            test_logits = model(X_te_gpu, mask=mask_te_gpu)
            test_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()

        all_val_probs.append(val_probs)
        all_test_probs.append(test_probs)

        del model, state_dict, X_va_gpu, mask_va_gpu, X_te_gpu, mask_te_gpu
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if m % 20 == 0:
            logger.info(f"Inferred {m}/{TOTAL_MODELS} models ({time.time()-t0:.1f}s)")

    all_val_probs = np.array(all_val_probs)   # (100, N_val, 18)
    all_test_probs = np.array(all_test_probs) # (100, N_test, 18)

    np.save(val_cache_path, all_val_probs)
    np.save(test_cache_path, all_test_probs)
    np.savez_compressed(
        meta_cache_path,
        va_race_masks=data_va["race_masks"],
        va_race_y=data_va["race_y"],
        va_odds=data_va["odds"],
        va_payouts=data_va["payouts"],
        va_ranks=data_va["ranks"],
        va_race_ids=data_va["race_ids"],
        te_race_masks=data_te["race_masks"],
        te_race_y=data_te["race_y"],
        te_odds=data_te["odds"],
        te_payouts=data_te["payouts"],
        te_ranks=data_te["ranks"],
        te_race_ids=data_te["race_ids"]
    )
    logger.info(f"Saved inference cache to {artifacts_dir} (Val: {all_val_probs.shape}, Test: {all_test_probs.shape})")

    return all_val_probs, all_test_probs, data_va, data_te

def evaluate_subset(
    subset: list,
    all_val_probs: np.ndarray,
    data_va: dict
) -> dict:
    """
    アンサンブルの平均確率に基づき、EV考慮なし (selection_mode: prob) でスコア計算
    """
    sub_probs = all_val_probs[subset]
    ens_probs = np.mean(sub_probs, axis=0)

    metrics = calculate_race_array_payout_metric(
        probs=ens_probs,
        masks=data_va["race_masks"],
        y_true=data_va["race_y"],
        odds=data_va["odds"],
        payouts=data_va["payouts"],
        selection_mode="prob"
    )
    hit_rate = metrics["hit_rate"]
    roi = metrics["roi"]
    # 総合スコア: 的中率20%超を確保しつつ回収率を高める評価
    score = roi * min(1.0, hit_rate / 0.20) + hit_rate
    return {
        "subset": list(subset),
        "k": len(subset),
        "hit_rate": hit_rate,
        "roi": roi,
        "score": score,
        "hits": metrics["hits"],
        "total_payout": metrics["total_payout"],
        "total_investment": metrics["total_investment"],
        "total_races": metrics["total_races"]
    }

# =========================================================================
# アプローチ 1: 多重スタート遺伝的探索 (Multi-Start GA / Extensive Random Search)
# =========================================================================
def run_multi_start_ga_search(
    all_val_probs: np.ndarray,
    data_va: dict,
    min_k: int = 8,
    max_k: int = 15,
    num_runs: int = 15,
    generations_per_run: int = 250,
    pop_size: int = 40
) -> dict:
    """
    完全にランダムな独立した初期集団（Multi-Start）から遺伝的アルゴリズムを実行し、
    Greedy系列にとらわれない多様な組み合わせパターンを探索する。
    """
    num_models = len(all_val_probs)
    total_evals = num_runs * generations_per_run * pop_size
    logger.info(f"\n=== Approach 1: Multi-Start Genetic Algorithm Search ({total_evals:,} evaluations) ===")

    best_results_by_k = {}

    for k in range(min_k, max_k + 1):
        global_best_k = None
        global_best_k_score = -1.0
        diverse_top_patterns = []

        logger.info(f"\n--- Searching for Ensemble Size k={k} ({num_runs} independent starts) ---")

        for run_id in range(num_runs):
            rng = random.Random(1000 + k * 100 + run_id * 17)

            # 完全にランダムな初期集団
            population = []
            for _ in range(pop_size):
                ind = sorted(rng.sample(range(num_models), k))
                population.append(ind)

            pop_evals = [evaluate_subset(ind, all_val_probs, data_va) for ind in population]

            for gen in range(generations_per_run):
                # トーナメント選択
                new_pop = []
                # エリート保存 (Top 2)
                sorted_idx = np.argsort([-e["score"] for e in pop_evals])
                new_pop.append(list(population[sorted_idx[0]]))
                new_pop.append(list(population[sorted_idx[1]]))

                while len(new_pop) < pop_size:
                    # トーナメント選択 (サイズ 3)
                    t_cand1 = rng.sample(range(pop_size), 3)
                    p1_idx = max(t_cand1, key=lambda i: pop_evals[i]["score"])
                    p1 = population[p1_idx]

                    t_cand2 = rng.sample(range(pop_size), 3)
                    p2_idx = max(t_cand2, key=lambda i: pop_evals[i]["score"])
                    p2 = population[p2_idx]

                    # 交叉 (Uniform Crossover with size k constraint)
                    union_models = list(set(p1) | set(p2))
                    if len(union_models) >= k:
                        child = rng.sample(union_models, k)
                    else:
                        missing = [m for m in range(num_models) if m not in union_models]
                        child = union_models + rng.sample(missing, k - len(union_models))

                    # 突然変異 (確率 30% で1〜2モデルを入れ替え)
                    if rng.random() < 0.30:
                        avail = [m for m in range(num_models) if m not in child]
                        if avail:
                            num_mut = 1 if rng.random() < 0.7 else min(2, len(child), len(avail))
                            for _ in range(num_mut):
                                rem_i = rng.randrange(len(child))
                                add_m = rng.choice(avail)
                                child[rem_i] = add_m
                                avail.remove(add_m)

                    new_pop.append(sorted(child))

                population = new_pop
                pop_evals = [evaluate_subset(ind, all_val_probs, data_va) for ind in population]

            # 各ランのベスト
            best_idx = np.argmax([e["score"] for e in pop_evals])
            run_best = pop_evals[best_idx]

            diverse_top_patterns.append(run_best)

            if run_best["score"] > global_best_k_score:
                global_best_k_score = run_best["score"]
                global_best_k = run_best

        # スコア順にソートして重複除去
        unique_patterns = []
        seen_sets = set()
        for p in sorted(diverse_top_patterns, key=lambda x: -x["score"]):
            frozen = tuple(sorted(p["subset"]))
            if frozen not in seen_sets:
                seen_sets.add(frozen)
                unique_patterns.append(p)

        best_results_by_k[k] = {
            "best": global_best_k,
            "top_diverse": unique_patterns[:3] # 上位3つの独立パターン
        }

        b = global_best_k
        logger.info(
            f"★ [GA k={k:2d} Global Best]: Models {[m+1 for m in b['subset']]} | "
            f"Val Hit={b['hit_rate']*100:.2f}% | Val ROI={b['roi']*100:.2f}% | Score={b['score']:.4f}"
        )

    return best_results_by_k

# =========================================================================
# アプローチ 2: 相関排除・多様性重視アンサンブル (Diversity / Low Correlation Selection)
# =========================================================================
def run_diversity_correlation_search(
    all_val_probs: np.ndarray,
    data_va: dict,
    min_k: int = 8,
    max_k: int = 15
) -> dict:
    """
    100モデルの予測勝率の相関行列を計算し、
    互いに相関の低い（予測の重複が少ない）多様なモデル同士を選抜してアンサンブル。
    """
    num_models = len(all_val_probs)
    logger.info(f"\n=== Approach 2: Correlation-Filtered & Diversity-Driven Selection ===")

    # 1. 各モデルの予測ベクトル (N_val * 18 次元) を作成して相関行列を計算
    flat_probs = all_val_probs.reshape(num_models, -1) # (100, 3454 * 18)
    logger.info("Computing prediction correlation matrix across 100 models...")
    corr_matrix = np.corrcoef(flat_probs) # (100, 100)
    
    # 単体スコアの計算
    single_scores = []
    for m in range(num_models):
        res = evaluate_subset([m], all_val_probs, data_va)
        single_scores.append(res["score"])
    single_scores = np.array(single_scores)

    # 距離行列: dist = 1 - corr
    dist_matrix = np.clip(1.0 - corr_matrix, 0.0, 2.0)
    condensed_dist = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method="average")

    diversity_results = {}

    for k in range(min_k, max_k + 1):
        # 方式 2-A: 階層型クラスタリングによる代表モデル選抜
        # 100モデルを k 個のクラスタに分割し、各クラスタで単体スコア最高のモデルを代表選抜
        cluster_labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        cluster_selected = []
        for c_id in range(1, k + 1):
            members = np.where(cluster_labels == c_id)[0]
            if len(members) > 0:
                best_member = members[np.argmax(single_scores[members])]
                cluster_selected.append(int(best_member))

        # もしクラスタ数がkに満たない場合はスコア上位で補完
        while len(cluster_selected) < k:
            avail = [m for m in np.argsort(-single_scores) if m not in cluster_selected]
            cluster_selected.append(int(avail[0]))

        res_cluster = evaluate_subset(cluster_selected, all_val_probs, data_va)

        # 方式 2-B: 最小相関制約付き Greedy (Max Diversity Greedy)
        # 単体最良モデルからスタートし、既存選択モデル群との平均相関が最も低く、
        # かつスコアを改善するモデルを逐次選択
        div_selected = [int(np.argmax(single_scores))]
        while len(div_selected) < k:
            best_cand = None
            best_div_score = -1e9
            for cand in range(num_models):
                if cand in div_selected:
                    continue
                # 既存モデルとの平均相関
                avg_corr = np.mean(corr_matrix[cand, div_selected])
                # 相関ペナルティ付きスコア
                trial_res = evaluate_subset(div_selected + [cand], all_val_probs, data_va)
                div_score = trial_res["score"] - 0.20 * avg_corr # 相関が低いほど加点
                if div_score > best_div_score:
                    best_div_score = div_score
                    best_cand = cand
            div_selected.append(best_cand)

        res_min_corr = evaluate_subset(div_selected, all_val_probs, data_va)

        # 2-A と 2-B でスコアの高い方を採用
        chosen = res_cluster if res_cluster["score"] >= res_min_corr["score"] else res_min_corr
        chosen_type = "Clustering" if res_cluster["score"] >= res_min_corr["score"] else "Min-Correlation"

        diversity_results[k] = {
            "result": chosen,
            "type": chosen_type,
            "cluster_res": res_cluster,
            "min_corr_res": res_min_corr
        }

        logger.info(
            f"★ [Diversity k={k:2d} ({chosen_type})]: Models {[m+1 for m in chosen['subset']]} | "
            f"Val Hit={chosen['hit_rate']*100:.2f}% | Val ROI={chosen['roi']*100:.2f}% | Score={chosen['score']:.4f}"
        )

    return diversity_results

def evaluate_on_test(subset: list, all_test_probs: np.ndarray, data_te: dict) -> dict:
    sub_probs = all_test_probs[subset]
    ens_probs = np.mean(sub_probs, axis=0)
    metrics = calculate_race_array_payout_metric(
        probs=ens_probs,
        masks=data_te["race_masks"],
        y_true=data_te["race_y"],
        odds=data_te["odds"],
        payouts=data_te["payouts"],
        selection_mode="prob"
    )
    return metrics

def main():
    artifacts_dir = "./artifacts_models_100"
    os.makedirs(artifacts_dir, exist_ok=True)
    device = get_device()

    # 1. 予測勝率テンソルのロード（キャッシュまたは推論）
    all_val_probs, all_test_probs, data_va, data_te = load_or_cache_model_predictions(
        artifacts_dir=artifacts_dir,
        config_path="configs/config.yaml",
        device=device
    )

    # 2. アプローチ 1: 多重スタート遺伝的探索 (Multi-Start GA)
    ga_results = run_multi_start_ga_search(
        all_val_probs=all_val_probs,
        data_va=data_va,
        min_k=8,
        max_k=15,
        num_runs=15,
        generations_per_run=250,
        pop_size=40
    )

    # 3. アプローチ 2: 相関排除・多様性重視アンサンブル
    div_results = run_diversity_correlation_search(
        all_val_probs=all_val_probs,
        data_va=data_va,
        min_k=8,
        max_k=15
    )

    # 4. 未見テストデータ (2024-2025 Test 3,455レース) での総合比較
    logger.info("\n=== Final Test Evaluation across Multiple Diverse Approaches ===")

    comparison_rows = []

    # 先ほどの Greedy 系列のベスト結果 (k=9, k=14) を参照用に追加
    prev_summary_path = os.path.join(artifacts_dir, "ensemble_patterns_summary.csv")
    if os.path.exists(prev_summary_path):
        df_prev = pd.read_csv(prev_summary_path)
        for _, row in df_prev.iterrows():
            if "k=8" in str(row["pattern"]) or "k=9" in str(row["pattern"]) or "k=14" in str(row["pattern"]):
                comparison_rows.append({
                    "approach": "Greedy (Baseline)",
                    "pattern": row["pattern"],
                    "k": row["num_models"],
                    "val_hit_rate": row["val_hit_rate"],
                    "val_roi": row["val_roi"],
                    "test_hit_rate": row["test_hit_rate"],
                    "test_roi": row["test_roi"],
                    "test_hits": row["test_hits"],
                    "test_payout": row["test_payout"],
                    "models_used": row["models_used"]
                })

    # アプローチ 1 (Multi-Start GA) の結果
    for k in range(8, 16):
        b = ga_results[k]["best"]
        subset = b["subset"]
        te_m = evaluate_on_test(subset, all_test_probs, data_te)
        comparison_rows.append({
            "approach": "1. Multi-Start GA (New Patterns)",
            "pattern": f"GA Best k={k}",
            "k": k,
            "val_hit_rate": b["hit_rate"],
            "val_roi": b["roi"],
            "test_hit_rate": te_m["hit_rate"],
            "test_roi": te_m["roi"],
            "test_hits": te_m["hits"],
            "test_payout": te_m["total_payout"],
            "models_used": str([m + 1 for m in subset])
        })

    # アプローチ 2 (Diversity / Correlation-Filtered) の結果
    for k in range(8, 16):
        d = div_results[k]["result"]
        subset = d["subset"]
        te_m = evaluate_on_test(subset, all_test_probs, data_te)
        comparison_rows.append({
            "approach": "2. Diversity / Low-Correlation",
            "pattern": f"Diversity ({div_results[k]['type']}) k={k}",
            "k": k,
            "val_hit_rate": d["hit_rate"],
            "val_roi": d["roi"],
            "test_hit_rate": te_m["hit_rate"],
            "test_roi": te_m["roi"],
            "test_hits": te_m["hits"],
            "test_payout": te_m["total_payout"],
            "models_used": str([m + 1 for m in subset])
        })

    df_comp = pd.DataFrame(comparison_rows)
    comp_output_path = os.path.join(artifacts_dir, "advanced_ensemble_comparison.csv")
    df_comp.to_csv(comp_output_path, index=False)

    print("\n" + "="*110)
    print("      ★ 多様な探索アプローチによるアンサンブル組み合わせ (k=8~15) 総合比較サマリー ★")
    print("="*110)
    print(f"テスト対象: 3,455 レース (2024-2025年未見データ | selection_mode: prob (EV考慮なし))")
    print("-"*110)
    disp_cols = ["approach", "pattern", "k", "val_hit_rate", "val_roi", "test_hit_rate", "test_roi", "test_hits", "test_payout"]
    print(df_comp[disp_cols].to_string(index=False))
    print("="*110)
    print(f"\n[Saved Advanced Comparison]: {comp_output_path}\n")

if __name__ == "__main__":
    main()
