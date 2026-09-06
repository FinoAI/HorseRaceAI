import json
import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from src.utils.helpers import setup_logger, save_json

logger = setup_logger("Preprocessor")

class FeaturePreprocessor:
    def __init__(
        self,
        prohibited_columns: List[str],
        id_and_str_columns: List[str],
        bac_feature_cols: List[str],
        artifacts_dir: str
    ):
        self.prohibited_columns = set(prohibited_columns)
        self.id_and_str_columns = set(id_and_str_columns)
        self.bac_feature_cols = [c for c in bac_feature_cols if c not in self.prohibited_columns and not c.startswith("Target_")]
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.scalers = {} # 各モデルごとの平均・標準偏差
        self.fill_values = {}

    def get_candidate_yearly_columns(self, all_columns: List[str]) -> List[str]:
        """
        開催年データから、除外対象（禁止カラム、Target_*、ID・文字列、BACカラム）を除いた候補カラムを抽出
        """
        candidate_cols = []
        for c in all_columns:
            if c.startswith("Target_"):
                continue
            if c in self.prohibited_columns:
                continue
            if c in self.id_and_str_columns:
                continue
            if c in ["RACE_ID", "KYI_RACE_KEY", "YEAR"]:
                continue
            if c in self.bac_feature_cols:
                continue
            candidate_cols.append(c)
        return candidate_cols

    def sample_features_for_model(
        self,
        model_idx: int,
        yearly_candidate_cols: List[str],
        sample_ratio: float = 0.35,
        random_seed: int = 42
    ) -> List[str]:
        """
        前段モデル用の特徴量を決定（BAC_KABカラムは全使用 + 開催年カラムからランダム抽出）
        選定されたカラムを JSON に保存
        """
        rng = random.Random(random_seed + model_idx)
        k = int(len(yearly_candidate_cols) * sample_ratio)
        k = max(10, min(k, len(yearly_candidate_cols)))
        selected_yearly = rng.sample(yearly_candidate_cols, k)
        
        # BAC_KAB カラム（全使用） + ランダム抽出した開催年カラム
        selected_features = list(self.bac_feature_cols) + selected_yearly
        
        # JSON保存
        json_path = os.path.join(self.artifacts_dir, f"features_model_{model_idx}.json")
        save_data = {
            "model_idx": model_idx,
            "num_features": len(selected_features),
            "bac_features_count": len(self.bac_feature_cols),
            "yearly_features_count": len(selected_yearly),
            "features": selected_features
        }
        save_json(save_data, json_path)
        logger.info(f"[Model {model_idx}] Saved {len(selected_features)} features to {json_path}")
        return selected_features

    def fit_transform_features(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        model_idx: int
    ) -> np.ndarray:
        """
        指定された特徴量セットについて、Trainデータから統計量（平均・標準偏差・中央値）を計算して正規化
        """
        sub_df = df_train[features].copy()
        
        # すべて数値に変換
        for c in features:
            sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")

        means = sub_df.mean(axis=0)
        stds = sub_df.std(axis=0)
        stds = stds.replace(0, 1.0).fillna(1.0)
        medians = sub_df.median(axis=0).fillna(0.0)

        self.fill_values[model_idx] = medians.to_dict()
        self.scalers[model_idx] = {
            "mean": means.to_dict(),
            "std": stds.to_dict()
        }

        # 欠損補完 & 標準化
        sub_df = sub_df.fillna(medians)
        norm_arr = ((sub_df - means) / stds).values
        # NaN / Inf 安全対策
        norm_arr = np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return norm_arr.astype(np.float32)

    def transform_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        model_idx: int
    ) -> np.ndarray:
        """
        学習済み統計量を用いてVal/Testデータを正規化
        """
        sub_df = df[features].copy()
        for c in features:
            sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce")

        fill_v = self.fill_values[model_idx]
        mean_v = pd.Series(self.scalers[model_idx]["mean"])
        std_v = pd.Series(self.scalers[model_idx]["std"])

        sub_df = sub_df.fillna(fill_v)
        norm_arr = ((sub_df - mean_v) / std_v).values
        norm_arr = np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return norm_arr.astype(np.float32)

def extract_target_and_meta(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    目的変数 (1着なら 1, それ以外 0) と評価用メタ情報を抽出
    """
    # Target_着順 == 1 が正解
    order_col = pd.to_numeric(df["Target_着順"], errors="coerce").fillna(99)
    y = (order_col == 1).astype(np.float32).values

    # 評価・シミュレーション用メタデータ
    meta_df = pd.DataFrame({
        "RACE_ID": df["RACE_ID"].astype(str),
        "KYI_RACE_KEY": df["KYI_RACE_KEY"].astype(str),
        "KYI_馬番": pd.to_numeric(df["KYI_馬番"], errors="coerce").fillna(0).astype(int),
        "Target_着順": order_col.values,
        "Target_確定単勝オッズ": pd.to_numeric(df.get("Target_確定単勝オッズ", 0), errors="coerce").fillna(0.0).values,
        "Target_単勝": pd.to_numeric(df.get("Target_単勝", np.nan), errors="coerce").values
    })
    return y, meta_df
