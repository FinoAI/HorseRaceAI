import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

def main():
    artifacts_dir = "./artifacts_dirt_1win"
    os.makedirs(artifacts_dir, exist_ok=True)

    cache_path = os.path.join(artifacts_dir, "cached_dirt_1win_data.pkl")
    print(f"Loading cached dirt 1-win data from {cache_path}...")
    with open(cache_path, "rb") as f:
        data_dict = pickle.load(f)

    data_tr = data_dict["data_tr"]
    data_va = data_dict["data_va"]
    data_te = data_dict["data_te"]

    # 1. シャッフル順序の再現（seed=42）
    df_bac = pd.read_csv("/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/BAC_KAB.csv", dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    dirt_ids = set(df_bac[df_bac["芝ダ障害コード"] == 2]["RACE_ID"])

    dfs = []
    for y in [2025, 2026]:
        df = pd.read_csv(f"/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/{y}.csv", dtype={"KYI_RACE_KEY": str}, low_memory=False)
        r_keys = df["KYI_RACE_KEY"].astype(str).str.zfill(10)
        df["RACE_ID"] = r_keys.str[:8]
        df_f = df[df["RACE_ID"].isin(dirt_ids) & df["KYI_条件クラス"].isin([1, 2, 3, 9])].copy()
        dfs.append(df_f)

    df_all = pd.concat(dfs, ignore_index=True)
    unique_races = np.array(df_all["RACE_ID"].drop_duplicates().tolist())

    SHUFFLE_SEED = 42
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled_races = unique_races.copy()
    rng.shuffle(shuffled_races)

    val_race_ids = shuffled_races[:751].tolist()
    test_race_ids = shuffled_races[751:].tolist()
    all_1502_race_ids = shuffled_races.tolist()

    # 2. 仕様JSONの作成
    benchmark_info = {
        "benchmark_name": "DIRT_1WIN_FIXED_BENCHMARK",
        "description": "2009-2024学習(2024年含む) および 2025-2026足してシャッフルした全量テストの固定基準",
        "train_spec": {
            "years": list(range(2009, 2025)),
            "total_years": 16,
            "total_races": len(data_tr["race_y"]),
            "note": "2009年から2024年までの全レース（2024年を学習に完全包含）"
        },
        "test_benchmark_spec": {
            "years": [2025, 2026],
            "total_races": len(all_1502_race_ids),
            "shuffle_seed": SHUFFLE_SEED,
            "shuffled_all_1502_race_ids": all_1502_race_ids,
            "half_split_val_751_race_ids": val_race_ids,
            "half_split_test_751_race_ids": test_race_ids
        },
        "evaluation_rules": {
            "primary_metric": "FULL_1502_ROI (2025-2026シャッフル全1,502レースでの回収率)",
            "primary_hit_rate": "FULL_1502_HIT_RATE",
            "reference_half_test_roi": "TEST_751_ROI",
            "reference_half_val_roi": "VAL_751_ROI"
        }
    }

    json_save_path = os.path.join(artifacts_dir, "benchmark_dataset_spec.json")
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_info, f, ensure_ascii=False, indent=2)
    print(f"Saved benchmark spec to {json_save_path}")

    # 3. 全量1,502レースのテンソル統合キャッシュ作成
    full_3d = np.concatenate([data_va["race_3d"], data_te["race_3d"]], axis=0)
    full_masks = np.concatenate([data_va["race_masks"], data_te["race_masks"]], axis=0)
    full_y = np.concatenate([data_va["race_y"], data_te["race_y"]], axis=0)
    full_odds = np.concatenate([data_va["odds"], data_te["odds"]], axis=0)
    full_payouts = np.concatenate([data_va["payouts"], data_te["payouts"]], axis=0)
    full_ranks = np.concatenate([data_va["ranks"], data_te["ranks"]], axis=0)
    full_race_ids = np.concatenate([data_va["race_ids"], data_te["race_ids"]], axis=0)

    full_cache_data = {
        "race_3d": full_3d,
        "race_masks": full_masks,
        "race_y": full_y,
        "odds": full_odds,
        "payouts": full_payouts,
        "ranks": full_ranks,
        "race_ids": full_race_ids,
        "bac_col_indices": data_dict["bac_col_indices"],
        "col_to_idx": data_dict["col_to_idx"],
        "all_feature_cols": data_dict["all_feature_cols"]
    }

    full_cache_path = os.path.join(artifacts_dir, "cached_test_full_1502.pkl")
    with open(full_cache_path, "wb") as f:
        pickle.dump(full_cache_data, f)
    print(f"Saved full 1,502 test cache to {full_cache_path}")
    print(f"Summary: Train={len(data_tr['race_y']):,d} races (2009-2024), Test Full={len(full_y):,d} races (2025-2026)")

if __name__ == "__main__":
    main()
