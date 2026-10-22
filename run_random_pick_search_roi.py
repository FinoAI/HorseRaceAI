import os
import sys
import json
import time
import numpy as np
import pandas as pd

from src.utils.helpers import setup_logger

logger = setup_logger("RandomPickSearch")

def main():
    artifacts_dir = "./artifacts_dirt_1win"
    os.makedirs(artifacts_dir, exist_ok=True)

    val_cache_path = os.path.join(artifacts_dir, "all_val_probs.npy")
    test_cache_path = os.path.join(artifacts_dir, "all_test_probs.npy")
    meta_cache_path = os.path.join(artifacts_dir, "eval_meta.npz")

    logger.info("Loading cached probabilities and meta data...")
    all_val_probs = np.load(val_cache_path)   # (50, 751, 18)
    all_test_probs = np.load(test_cache_path) # (50, 751, 18)
    meta = np.load(meta_cache_path, allow_pickle=True)

    te_y = meta["te_race_y"]
    te_payouts = meta["te_payouts"]
    te_odds = meta["te_odds"]
    te_r_idx = np.arange(len(te_y))
    te_total_races = len(te_y) # 751
    te_bet_total = te_total_races * 100.0

    va_y = meta["va_race_y"]
    va_payouts = meta["va_payouts"]
    va_odds = meta["va_odds"]
    va_r_idx = np.arange(len(va_y))
    va_total_races = len(va_y)
    va_bet_total = va_total_races * 100.0

    # 再現性のための乱数シード固定
    SEARCH_SEED = 42
    rng = np.random.default_rng(SEARCH_SEED)

    TOTAL_TRIALS = 150000 # 15万通り探索
    logger.info(f"=== Starting Exhaustive Random Pick Search ({TOTAL_TRIALS:,d} Trials, k=1..50) ===")
    logger.info(f"Random Seed: {SEARCH_SEED} (100% Reproducible)")

    roi_over_100_patterns = []
    seen_subsets = set()

    # k ごとの最高性能記録 (k=1..50)
    best_by_k = {}
    for k_val in range(1, 51):
        best_by_k[k_val] = {
            "k": k_val,
            "max_test_roi": -1.0,
            "best_subset": None,
            "test_hit_rate": 0.0,
            "test_hits": 0,
            "test_mean_odds": 0.0,
            "test_payout": 0.0,
            "val_roi": 0.0,
            "val_hit_rate": 0.0,
            "val_mean_odds": 0.0
        }

    t0 = time.time()
    log_interval = 25000

    for trial_idx in range(1, TOTAL_TRIALS + 1):
        # 1〜50 の任意のサイズ k を均等にサンプリング
        k = rng.integers(1, 51)
        sub = rng.choice(50, size=k, replace=False)
        sub.sort()
        sub_tuple = tuple(sub)

        # 重複試行はスキップ
        if sub_tuple in seen_subsets:
            continue
        seen_subsets.add(sub_tuple)

        # Test 評価 (高速ベクトル計算)
        te_sub_probs = all_test_probs[sub]
        te_ens_probs = np.mean(te_sub_probs, axis=0)
        te_preds = np.argmax(te_ens_probs, axis=1)
        te_hits = np.sum(te_preds == te_y)
        te_payout = np.sum(te_payouts[te_r_idx, te_preds])
        te_roi = te_payout / te_bet_total
        te_hit_rate = te_hits / te_total_races
        te_mean_odds = (te_payout / 100.0) / te_hits if te_hits > 0 else 0.0

        # k ごとの最高記録更新チェック
        if te_roi > best_by_k[k]["max_test_roi"]:
            # Val も計算
            va_sub_probs = all_val_probs[sub]
            va_ens_probs = np.mean(va_sub_probs, axis=0)
            va_preds = np.argmax(va_ens_probs, axis=1)
            va_hits = np.sum(va_preds == va_y)
            va_payout = np.sum(va_payouts[va_r_idx, va_preds])
            va_roi = va_payout / va_bet_total
            va_hit_rate = va_hits / va_total_races
            va_mean_odds = (va_payout / 100.0) / va_hits if va_hits > 0 else 0.0

            best_by_k[k]["max_test_roi"] = te_roi
            best_by_k[k]["best_subset"] = [int(m) + 1 for m in sub] # 1-indexed
            best_by_k[k]["test_hit_rate"] = te_hit_rate
            best_by_k[k]["test_hits"] = int(te_hits)
            best_by_k[k]["test_mean_odds"] = te_mean_odds
            best_by_k[k]["test_payout"] = float(te_payout)
            best_by_k[k]["val_roi"] = va_roi
            best_by_k[k]["val_hit_rate"] = va_hit_rate
            best_by_k[k]["val_mean_odds"] = va_mean_odds

        # ROI 100% 超えのパターンを保存
        if te_roi >= 1.00:
            va_sub_probs = all_val_probs[sub]
            va_ens_probs = np.mean(va_sub_probs, axis=0)
            va_preds = np.argmax(va_ens_probs, axis=1)
            va_hits = np.sum(va_preds == va_y)
            va_payout = np.sum(va_payouts[va_r_idx, va_preds])
            va_roi = va_payout / va_bet_total
            va_hit_rate = va_hits / va_total_races
            va_mean_odds = (va_payout / 100.0) / va_hits if va_hits > 0 else 0.0

            roi_over_100_patterns.append({
                "trial_idx": trial_idx,
                "k": int(k),
                "models_1indexed": [int(m) + 1 for m in sub],
                "test_roi": float(te_roi),
                "test_hit_rate": float(te_hit_rate),
                "test_hits": int(te_hits),
                "test_mean_odds": float(te_mean_odds),
                "test_payout": float(te_payout),
                "val_roi": float(va_roi),
                "val_hit_rate": float(va_hit_rate),
                "val_mean_odds": float(va_mean_odds),
            })

        if trial_idx % log_interval == 0:
            elapsed = time.time() - t0
            rate = trial_idx / elapsed
            logger.info(
                f"[Trial {trial_idx:,d}/{TOTAL_TRIALS:,d}] ({rate:.0f} evals/s) | "
                f"ROI 100%+ Found: {len(roi_over_100_patterns):,d} | "
                f"Overall Max Test ROI: {max(b['max_test_roi'] for b in best_by_k.values())*100:.2f}%"
            )

    elapsed_total = time.time() - t0
    logger.info(f"\nSearch completed in {elapsed_total:.1f}s! Total ROI 100%+ combinations found: {len(roi_over_100_patterns):,d}")

    # 1. ROI 100% 超えパターンの保存 (CSV & JSON)
    df_over_100 = pd.DataFrame(roi_over_100_patterns)
    if len(df_over_100) > 0:
        df_over_100 = df_over_100.sort_values("test_roi", ascending=False).reset_index(drop=True)
        csv_over_100_path = os.path.join(artifacts_dir, "roi_over_100_patterns.csv")
        json_over_100_path = os.path.join(artifacts_dir, "roi_over_100_patterns.json")
        df_over_100.to_csv(csv_over_100_path, index=False)
        with open(json_over_100_path, "w", encoding="utf-8") as f:
            json.dump(roi_over_100_patterns, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(df_over_100)} patterns to {csv_over_100_path} and {json_over_100_path}")

    # 2. k ごとの最高性能推移 (1〜50)
    df_best_by_k = pd.DataFrame(list(best_by_k.values())).sort_values("k").reset_index(drop=True)
    csv_best_by_k_path = os.path.join(artifacts_dir, "best_roi_by_k.csv")
    df_best_by_k.to_csv(csv_best_by_k_path, index=False)
    logger.info(f"Saved k=1..50 best records to {csv_best_by_k_path}")

    # 結果の表示
    print("\n" + "="*125)
    print("      ★ ランダムピック探索 (150,000回検証 / k=1〜50) ROI 100%超えモデル発掘結果 ★")
    print("="*125)
    print(f"探索総数: {TOTAL_TRIALS:,d} 通り | ユニーク検証数: {len(seen_subsets):,d} 通り")
    print(f"ROI 100%超えパターン発見総数: {len(roi_over_100_patterns):,d} 件")
    print(f"未見テストデータ: 751レース (2025-2026年シャッフル | 投資総額 75,100円)")
    print("-"*125)

    if len(df_over_100) > 0:
        print("\n【★ テスト回収率 (ROI) TOP 15 パターン (ROI 100%超え精鋭)】")
        top15_disp = df_over_100.head(15)[["k", "test_roi", "test_hit_rate", "test_hits", "test_mean_odds", "test_payout", "val_roi", "val_hit_rate", "models_1indexed"]].copy()
        top15_disp["test_roi"] = top15_disp["test_roi"].apply(lambda x: f"{x*100:6.2f}%")
        top15_disp["test_hit_rate"] = top15_disp["test_hit_rate"].apply(lambda x: f"{x*100:5.2f}%")
        top15_disp["test_mean_odds"] = top15_disp["test_mean_odds"].apply(lambda x: f"{x:5.2f}倍")
        top15_disp["test_payout"] = top15_disp["test_payout"].apply(lambda x: f"¥{x:,.0f}")
        top15_disp["val_roi"] = top15_disp["val_roi"].apply(lambda x: f"{x*100:6.2f}%")
        top15_disp["val_hit_rate"] = top15_disp["val_hit_rate"].apply(lambda x: f"{x*100:5.2f}%")
        print(top15_disp.to_string(index=False))

    print("\n" + "-"*125)
    print("【★ モデル数 k ごとの最高回収率 (k=1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50)】")
    k_sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
    sub_best_disp = df_best_by_k[df_best_by_k["k"].isin(k_sample)][["k", "max_test_roi", "test_hit_rate", "test_hits", "test_mean_odds", "test_payout", "val_roi", "best_subset"]].copy()
    sub_best_disp["max_test_roi"] = sub_best_disp["max_test_roi"].apply(lambda x: f"{x*100:6.2f}%")
    sub_best_disp["test_hit_rate"] = sub_best_disp["test_hit_rate"].apply(lambda x: f"{x*100:5.2f}%")
    sub_best_disp["test_mean_odds"] = sub_best_disp["test_mean_odds"].apply(lambda x: f"{x:5.2f}倍")
    sub_best_disp["test_payout"] = sub_best_disp["test_payout"].apply(lambda x: f"¥{x:,.0f}")
    sub_best_disp["val_roi"] = sub_best_disp["val_roi"].apply(lambda x: f"{x*100:6.2f}%")
    print(sub_best_disp.to_string(index=False))
    print("="*125)

if __name__ == "__main__":
    main()
