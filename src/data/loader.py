import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from src.utils.helpers import setup_logger

logger = setup_logger("DataLoader")

def load_bac_kab(data_dir: str, bac_filename: str = "BAC_KAB.csv") -> pd.DataFrame:
    bac_path = os.path.join(data_dir, bac_filename)
    if not os.path.exists(bac_path):
        raise FileNotFoundError(f"BAC_KAB file not found at: {bac_path}")
    logger.info(f"Loading BAC_KAB from {bac_path}...")
    df_bac = pd.read_csv(bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)
    logger.info(f"Loaded BAC_KAB: shape={df_bac.shape}")
    return df_bac

def load_year_data(
    data_dir: str,
    year: int,
    df_bac: pd.DataFrame,
    sample_n_races: Optional[int] = None
) -> pd.DataFrame:
    year_path = os.path.join(data_dir, f"{year}.csv")
    if not os.path.exists(year_path):
        raise FileNotFoundError(f"Year file not found at: {year_path}")
    logger.info(f"Loading {year}.csv...")
    df_year = pd.read_csv(year_path, dtype={"KYI_RACE_KEY": str}, low_memory=False)
    race_keys = df_year["KYI_RACE_KEY"].astype(str).str.zfill(10)
    race_ids = race_keys.str[:8]
    
    df_year = df_year.assign(
        KYI_RACE_KEY=race_keys,
        RACE_ID=race_ids
    )
    
    if sample_n_races is not None and sample_n_races > 0:
        unique_races = df_year["RACE_ID"].drop_duplicates().iloc[:sample_n_races]
        df_year = df_year[df_year["RACE_ID"].isin(unique_races)].copy().reset_index(drop=True)
        logger.info(f"Sampled {sample_n_races} races for dry-run in {year}: shape={df_year.shape}")

    # BAC_KAB と結合
    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_year.columns]
    merged = pd.merge(df_year, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
    merged = merged.assign(YEAR=year)
    logger.info(f"Merged {year}.csv with BAC_KAB: shape={merged.shape}")
    return merged

def get_train_val_test_datasets(
    data_dir: str,
    bac_filename: str = "BAC_KAB.csv",
    train_years: List[int] = [2018, 2019, 2020, 2021, 2022, 2023],
    val_test_years: List[int] = [2024, 2025],
    val_ratio: float = 0.5,
    random_seed: int = 42,
    sample_n_races_per_year: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    全データをロードし、結合、およびTrain/Val/Testへの分割を実行。
    2024と2025のデータを統合し、レースID単位でシャッフル分割してValとTestを作成。
    """
    df_bac = load_bac_kab(data_dir, bac_filename)
    bac_feature_cols = [c for c in df_bac.columns if c != "RACE_ID"]

    # 1. 学習データのロード
    train_dfs = []
    for y in train_years:
        df_y = load_year_data(data_dir, y, df_bac, sample_n_races=sample_n_races_per_year)
        train_dfs.append(df_y)
    df_train = pd.concat(train_dfs, ignore_index=True)
    logger.info(f"Total Train DataFrame: shape={df_train.shape}")

    # 2. 2024 & 2025 (Val & Test) データのロード
    val_test_dfs = []
    for y in val_test_years:
        df_y = load_year_data(data_dir, y, df_bac, sample_n_races=sample_n_races_per_year)
        val_test_dfs.append(df_y)
    df_val_test = pd.concat(val_test_dfs, ignore_index=True)
    logger.info(f"Combined 2024-2025 DataFrame: shape={df_val_test.shape}")

    # レース単位でシャッフル分割
    unique_races = np.array(df_val_test["RACE_ID"].drop_duplicates().tolist())
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_races)

    n_val_races = int(len(unique_races) * val_ratio)
    val_race_ids = set(unique_races[:n_val_races])
    test_race_ids = set(unique_races[n_val_races:])

    df_val = df_val_test[df_val_test["RACE_ID"].isin(val_race_ids)].reset_index(drop=True)
    df_test = df_val_test[df_val_test["RACE_ID"].isin(test_race_ids)].reset_index(drop=True)

    logger.info(f"Val DataFrame (shuffled 2024-2025): shape={df_val.shape}, races={len(val_race_ids)}")
    logger.info(f"Test DataFrame (shuffled 2024-2025): shape={df_test.shape}, races={len(test_race_ids)}")

    return df_train, df_val, df_test, bac_feature_cols
