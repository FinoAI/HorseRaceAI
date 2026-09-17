import json
import os
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
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
        self.scalers = {}
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
        sample_count: int = 14,
        sample_ratio: Optional[float] = None,
        random_seed: int = 42
    ) -> List[str]:
        """
        前段モデル用の特徴量を決定（BAC_KABカラムは全使用 + 開催年カラムから指定数ランダム抽出）
        ※選定された特徴量はそのモデル内の全ての馬で完全に共通
        選定されたカラムリストを JSON に保存
        """
        rng = random.Random(random_seed + model_idx)
        if sample_ratio is not None:
            k = int(len(yearly_candidate_cols) * sample_ratio)
        else:
            k = sample_count
        k = max(2, min(k, len(yearly_candidate_cols)))
        selected_yearly = rng.sample(yearly_candidate_cols, k)
        
        # BAC_KAB カラム（全使用） + ランダム抽出した開催年カラム
        selected_features = list(self.bac_feature_cols) + selected_yearly
        
        json_path = os.path.join(self.artifacts_dir, f"features_model_{model_idx}.json")
        save_data = {
            "model_idx": model_idx,
            "num_features": len(selected_features),
            "bac_features_count": len(self.bac_feature_cols),
            "yearly_features_count": len(selected_yearly),
            "features": selected_features
        }
        save_json(save_data, json_path)
        logger.info(f"[Model {model_idx}] Saved {len(selected_features)} features (BAC:{len(self.bac_feature_cols)} + Yearly:{len(selected_yearly)}) to {json_path}")
        return selected_features

    def fit_transform_features(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        model_idx: int
    ) -> np.ndarray:
        """
        指定された特徴量セットについて、Trainデータから統計量を計算して正規化
        """
        sub_df = df_train[features].copy()
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

        sub_df = sub_df.fillna(medians)
        norm_arr = ((sub_df - means) / stds).values
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
    order_col = pd.to_numeric(df["Target_着順"], errors="coerce").fillna(99)
    y = (order_col == 1).astype(np.float32).values

    meta_df = pd.DataFrame({
        "RACE_ID": df["RACE_ID"].astype(str),
        "KYI_RACE_KEY": df["KYI_RACE_KEY"].astype(str),
        "KYI_馬番": pd.to_numeric(df["KYI_馬番"], errors="coerce").fillna(0).astype(int),
        "Target_着順": order_col.values,
        "Target_確定単勝オッズ": pd.to_numeric(df.get("Target_確定単勝オッズ", 0), errors="coerce").fillna(0.0).values,
        "Target_単勝": pd.to_numeric(df.get("Target_単勝", np.nan), errors="coerce").values
    })
    return y, meta_df

def build_race_level_dataset(
    X_norm: np.ndarray,
    meta_df: pd.DataFrame,
    max_horses: int = 18
) -> Dict[str, Any]:
    """
    馬単位の正規化特徴量を行列からレース単位のテンソル表現に変換。
    各レースで馬番 (1〜18) のスロット (0〜17) に配置し、非出走スロットは 0 パディング。
    
    戻り値:
      - 'race_X': shape (N_races, max_horses * D) のフラット化レース特徴量
      - 'race_masks': shape (N_races, max_horses) の出走馬マスク (出走: 1.0, 未出走: 0.0)
      - 'race_y': shape (N_races,) の1着馬スロットインデックス (0〜17)
      - 'race_meta': レース単位のメタ情報辞書 (RACE_ID, odds (N, 18), payouts (N, 18), ranks (N, 18))
    """
    num_features = X_norm.shape[1]
    
    # レース順序を維持
    race_ids_ordered = []
    seen = set()
    for r in meta_df["RACE_ID"].values:
        if r not in seen:
            seen.add(r)
            race_ids_ordered.append(r)
    
    num_races = len(race_ids_ordered)
    race_id_to_idx = {r: i for i, r in enumerate(race_ids_ordered)}
    
    race_X_3d = np.zeros((num_races, max_horses, num_features), dtype=np.float32)
    race_masks = np.zeros((num_races, max_horses), dtype=np.float32)
    race_y = np.zeros(num_races, dtype=np.int64)
    race_odds = np.zeros((num_races, max_horses), dtype=np.float32)
    race_payouts = np.zeros((num_races, max_horses), dtype=np.float32)
    race_ranks = np.full((num_races, max_horses), 99.0, dtype=np.float32)

    # 各馬のデータをレースと馬番スロットに配置
    for i in range(len(meta_df)):
        r_id = meta_df["RACE_ID"].iloc[i]
        r_idx = race_id_to_idx[r_id]
        horse_num = meta_df["KYI_馬番"].iloc[i]
        
        # 1〜18の範囲内
        slot = horse_num - 1
        if 0 <= slot < max_horses:
            race_X_3d[r_idx, slot, :] = X_norm[i]
            race_masks[r_idx, slot] = 1.0
            
            rank = meta_df["Target_着順"].iloc[i]
            race_ranks[r_idx, slot] = rank
            if rank == 1:
                race_y[r_idx] = slot
                
            race_odds[r_idx, slot] = meta_df["Target_確定単勝オッズ"].iloc[i]
            payout = meta_df["Target_単勝"].iloc[i]
            race_payouts[r_idx, slot] = payout if not np.isnan(payout) else 0.0

    # (N_races, max_horses * num_features) にフラット化
    race_X = race_X_3d.reshape(num_races, max_horses * num_features)

    return {
        "race_X": race_X,
        "race_masks": race_masks,
        "race_y": race_y,
        "race_ids": np.array(race_ids_ordered),
        "odds": race_odds,
        "payouts": race_payouts,
        "ranks": race_ranks
    }
