import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

sys.path.append("HorseRaceAI")
from src.utils.helpers import setup_logger

logger = setup_logger("SavePreprocessorStats")

def compute_and_save_stats():
    data_dir = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN"
    bac_filename = "BAC_KAB.csv"
    train_years = list(range(2009, 2025))
    output_path = "HorseRaceAI/artifacts_dirt_1win/dirt_1win_preprocessor_stats.pkl"

    # キャッシュから all_feature_cols を取得
    with open("HorseRaceAI/artifacts_dirt_1win/cached_dirt_1win_data.pkl", "rb") as f:
        data_dict = pickle.load(f)

    all_feature_cols = data_dict["all_feature_cols"]
    bac_feature_cols = data_dict["bac_feature_cols"]
    logger.info(f"Loaded all_feature_cols: {len(all_feature_cols)} cols")

    # BAC_KAB 読み込み
    bac_path = os.path.join(data_dir, bac_filename)
    logger.info(f"Loading BAC_KAB from {bac_path}...")
    df_bac = pd.read_csv(bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)
    dirt_race_ids = set(df_bac[df_bac["芝ダ障害コード"] == 2]["RACE_ID"])

    # 2009〜2024年 読み込み
    train_dfs = []
    for y in train_years:
        yp = os.path.join(data_dir, f"{y}.csv")
        t0 = time.time()
        df_y = pd.read_csv(yp, dtype={"KYI_RACE_KEY": str}, low_memory=False)
        rkeys = df_y["KYI_RACE_KEY"].astype(str).str.zfill(10)
        rids = rkeys.str[:8]
        df_y = df_y.assign(KYI_RACE_KEY=rkeys, RACE_ID=rids)
        cond_mask = df_y["RACE_ID"].isin(dirt_race_ids) & df_y["KYI_条件クラス"].isin([1, 2, 3, 9])
        df_f = df_y[cond_mask].copy().reset_index(drop=True)

        bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_f.columns]
        merged = pd.merge(df_f, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
        train_dfs.append(merged)
        logger.info(f"Year {y}: {len(df_f)} rows in {time.time()-t0:.1f}s")

    df_train = pd.concat(train_dfs, ignore_index=True)
    logger.info(f"Total Train shape: {df_train.shape}")

    # 数値行列化
    mat_train = np.zeros((len(df_train), len(all_feature_cols)), dtype=np.float32)
    for i, c in enumerate(all_feature_cols):
        mat_train[:, i] = pd.to_numeric(df_train[c], errors="coerce").fillna(np.nan).values

    logger.info("Computing medians, means, stds...")
    mat_train = np.nan_to_num(mat_train, nan=np.nan, posinf=1e6, neginf=-1e6)
    with np.errstate(all='ignore'):
        medians = np.nanmedian(mat_train, axis=0)
    medians = np.nan_to_num(medians, nan=0.0)

    for col_i in range(mat_train.shape[1]):
        m_val = medians[col_i]
        nan_tr = np.isnan(mat_train[:, col_i])
        if np.any(nan_tr):
            mat_train[nan_tr, col_i] = m_val

    with np.errstate(all='ignore'):
        means = np.mean(mat_train, axis=0)
        stds = np.std(mat_train, axis=0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds <= 1e-5] = 1.0
    means = np.nan_to_num(means, nan=0.0)

    stats = {
        "all_feature_cols": all_feature_cols,
        "col_to_idx": {c: i for i, c in enumerate(all_feature_cols)},
        "bac_feature_cols": bac_feature_cols,
        "medians": medians,
        "means": means,
        "stds": stds
    }

    with open(output_path, "wb") as f:
        pickle.dump(stats, f)

    logger.info(f"Successfully saved preprocessor stats to {output_path}!")

if __name__ == "__main__":
    compute_and_save_stats()
