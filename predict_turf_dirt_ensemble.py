#!/usr/local/bin/python3.11
"""
predict_turf_dirt_ensemble.py
========================================================================
JRA 芝・ダート 1勝クラス以上 完全統合アンサンブル予想プログラム
- 芝レース: 神アンサンブル [Model 11 + Model 25 + Model 35]
  (未見テスト回収率 112.90% / 検証回収率 124.66% 完全黒字実証済み)
- ダートレース: BESTアンサンブル [Model 24 + Model 49]
  (未見テスト回収率 117.18% / 検証回収率 116.79% 完全黒字実証済み)
------------------------------------------------------------------------
【条件・仕様】
1. 対象レース: JRA 芝・ダート 1勝クラス以上 (1勝/2勝/3勝/オープン/重賞)
2. 正規化パラメータ:
   - 芝: 16年分 15,547レース基準 (turf_1win_preprocessor_stats.pkl)
   - ダート: 16年分 14,742レース基準 (dirt_1win_preprocessor_stats.pkl)
3. 算出項目: 各レース全頭の予測勝率 (ENS / 構成モデル内訳)、印 (◎◯▲△△★)、
   基準オッズ、期待値 (EV = 勝率 × 基準オッズ)、推奨度判定
4. 出力:
   - コンソール: 全対象レース一覧サマリー (場・R順、芝/ダート明記)、激熱推奨ピックアップ
   - Markdownレポート: 総合サマリー、推奨馬ピックアップ、全21レース全頭詳細表
   - CSV出力: 全出走馬の詳細予想データ統合ファイル
========================================================================
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# パス解決
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import get_device
from src.models.base_nn import RaceLevelStage1NN

# 競馬場・クラス表示マッピング
PLACE_DICT = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"
}

CLASS_DICT = {
    1: "1勝クラス (500万下)",
    2: "2勝クラス (1000万下)",
    3: "3勝クラス (1600万下)",
    9: "オープン/重賞"
}

def load_turf_models(base_dir: str, device: torch.device):
    """芝用 M11, M25, M35 モデルおよび特徴量定義のロード"""
    art_dir = os.path.join(base_dir, "artifacts_turf_1win")

    with open(os.path.join(art_dir, "features_11.json"), "r", encoding="utf-8") as f:
        f11_data = json.load(f)
    with open(os.path.join(art_dir, "features_25.json"), "r", encoding="utf-8") as f:
        f25_data = json.load(f)
    with open(os.path.join(art_dir, "features_35.json"), "r", encoding="utf-8") as f:
        f35_data = json.load(f)

    max_horses = 18
    num_features = 200
    input_dim = max_horses * num_features

    net11 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w11_path = os.path.join(art_dir, "model_11.pth")
    net11.load_state_dict(torch.load(w11_path, map_location=device))
    net11.eval()

    net25 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w25_path = os.path.join(art_dir, "model_25.pth")
    net25.load_state_dict(torch.load(w25_path, map_location=device))
    net25.eval()

    net35 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w35_path = os.path.join(art_dir, "model_35.pth")
    net35.load_state_dict(torch.load(w35_path, map_location=device))
    net35.eval()

    return {
        "net11": net11,
        "net25": net25,
        "net35": net35,
        "m11_yearly": f11_data["yearly_features"],
        "m25_yearly": f25_data["yearly_features"],
        "m35_yearly": f35_data["yearly_features"]
    }

def load_dirt_models(base_dir: str, device: torch.device):
    """ダート用 M24, M49 モデルおよび特徴量定義のロード"""
    art_dir = os.path.join(base_dir, "artifacts_dirt_1win")
    best_dir = os.path.join(art_dir, "BEST_MODEL_DIRT_1WIN")
    if not os.path.exists(best_dir):
        best_dir = art_dir

    with open(os.path.join(art_dir, "features_24.json"), "r", encoding="utf-8") as f:
        f24_data = json.load(f)
    with open(os.path.join(art_dir, "features_49.json"), "r", encoding="utf-8") as f:
        f49_data = json.load(f)

    max_horses = 18
    num_features = 200
    input_dim = max_horses * num_features

    net24 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w24_path = os.path.join(best_dir, "model_24.pth")
    if not os.path.exists(w24_path):
        w24_path = os.path.join(art_dir, "model_24.pth")
    net24.load_state_dict(torch.load(w24_path, map_location=device))
    net24.eval()

    net49 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w49_path = os.path.join(best_dir, "model_49.pth")
    if not os.path.exists(w49_path):
        w49_path = os.path.join(art_dir, "model_49.pth")
    net49.load_state_dict(torch.load(w49_path, map_location=device))
    net49.eval()

    return {
        "net24": net24,
        "net49": net49,
        "m24_yearly": f24_data["yearly_features"],
        "m49_yearly": f49_data["yearly_features"]
    }

def normalize_features(df_subset: pd.DataFrame, stats: dict) -> np.ndarray:
    """特徴量抽出および正規化パラメータ適用"""
    all_feature_cols = stats["all_feature_cols"]
    medians = stats["medians"]
    means = stats["means"]
    stds = stats["stds"]

    mat = np.zeros((len(df_subset), len(all_feature_cols)), dtype=np.float32)
    for i, c in enumerate(all_feature_cols):
        if c in df_subset.columns:
            mat[:, i] = pd.to_numeric(df_subset[c], errors="coerce").fillna(np.nan).values
        else:
            mat[:, i] = np.nan

    # median 補完
    for col_i in range(len(all_feature_cols)):
        nan_mask = np.isnan(mat[:, col_i])
        if np.any(nan_mask):
            mat[nan_mask, col_i] = medians[col_i]

    # 標準化 (mean, std) & クリッピング
    mat = np.clip((mat - means) / stds, -10.0, 10.0)
    mat = np.nan_to_num(mat, nan=0.0)
    return mat

def get_recommendation_label(prob: float, ev: float) -> tuple:
    """勝率と期待値(EV)から推奨ラベルとグレードを取得"""
    if prob >= 0.25 and ev >= 1.2:
        return "★★★ 激熱鉄板 (勝率25%+ & EV1.2+)", 3, "激熱鉄板"
    elif ev >= 2.0:
        return "★★★ 爆穴妙味 (EV2.0+ 超高期待値)", 3, "爆穴妙味"
    elif prob >= 0.25:
        return "★★☆ 有力本命 (勝率25%+)", 2, "有力本命"
    elif ev >= 1.2:
        return "★★☆ 妙味中穴 (EV1.2+)", 2, "妙味中穴"
    else:
        return "★☆☆ 通常", 1, "通常"

def main():
    parser = argparse.ArgumentParser(description="JRA Turf & Dirt 1-Win+ Unified Ensemble Prediction")
    parser.add_argument("--target_path", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/target_v2.csv", help="Path to target_v2.csv")
    parser.add_argument("--bac_path", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/BAC_KAB.csv", help="Path to BAC_KAB.csv")
    parser.add_argument("--output_dir", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/", help="Output directory")
    args = parser.parse_args()

    print("=" * 115)
    print("  ★ JRA 芝・ダート 1勝クラス以上 完全統合アンサンブル予想プログラム ★")
    print("  - 芝  : 神アンサンブル [Model 11 + Model 25 + Model 35] (未見回収率 112.90% / 検証 124.66%)")
    print("  - ダート: BESTアンサンブル [Model 24 + Model 49]       (未見回収率 117.18% / 検証 116.79%)")
    print("=" * 115)

    device = get_device()
    print(f"使用デバイス: {device}")

    # 1. 芝・ダート正規化パラメータのロード
    turf_stats_path = os.path.join(CURRENT_DIR, "artifacts_turf_1win/turf_1win_preprocessor_stats.pkl")
    dirt_stats_path = os.path.join(CURRENT_DIR, "artifacts_dirt_1win/dirt_1win_preprocessor_stats.pkl")

    if not os.path.exists(turf_stats_path):
        print(f"Error: 芝正規化パラメータファイル {turf_stats_path} が見つかりません。")
        sys.exit(1)
    if not os.path.exists(dirt_stats_path):
        print(f"Error: ダート正規化パラメータファイル {dirt_stats_path} が見つかりません。")
        sys.exit(1)

    with open(turf_stats_path, "rb") as f:
        turf_stats = pickle.load(f)
    with open(dirt_stats_path, "rb") as f:
        dirt_stats = pickle.load(f)

    print(f"芝正規化パラメータ  : {turf_stats_path} (16年分 15,547レース基準)")
    print(f"ダート正規化パラメータ: {dirt_stats_path} (16年分 14,742レース基準)")

    # 2. モデルロード
    print("\n学習済みモデルをロード中...")
    turf_models = load_turf_models(CURRENT_DIR, device)
    dirt_models = load_dirt_models(CURRENT_DIR, device)
    print("芝 [M11, M25, M35] および ダート [M24, M49] のロードが完了しました。")

    # 特徴量インデックス
    turf_bac_cols = turf_stats["bac_feature_cols"]
    turf_col_to_idx = turf_stats["col_to_idx"]
    f11_cols = list(turf_bac_cols) + turf_models["m11_yearly"]
    f25_cols = list(turf_bac_cols) + turf_models["m25_yearly"]
    f35_cols = list(turf_bac_cols) + turf_models["m35_yearly"]
    f11_indices = [turf_col_to_idx[c] for c in f11_cols]
    f25_indices = [turf_col_to_idx[c] for c in f25_cols]
    f35_indices = [turf_col_to_idx[c] for c in f35_cols]

    dirt_bac_cols = dirt_stats["bac_feature_cols"]
    dirt_col_to_idx = dirt_stats["col_to_idx"]
    f24_cols = list(dirt_bac_cols) + dirt_models["m24_yearly"]
    f49_cols = list(dirt_bac_cols) + dirt_models["m49_yearly"]
    f24_indices = [dirt_col_to_idx[c] for c in f24_cols]
    f49_indices = [dirt_col_to_idx[c] for c in f49_cols]

    # 3. 入力データロード
    print(f"\n出走馬データをロード中: {args.target_path}...")
    df_target = pd.read_csv(args.target_path, low_memory=False)

    print(f"レース情報をロード中: {args.bac_path}...")
    df_bac = pd.read_csv(args.bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)

    # RACE_ID 付与
    target_rkeys = df_target["KYI_RACE_KEY"].astype(str).str.zfill(10)
    df_target["RACE_ID"] = target_rkeys.str[:8]
    df_target["KYI_RACE_KEY"] = target_rkeys

    # BAC とマージ
    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_target.columns]
    merged = pd.merge(df_target, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")

    # 4. フィルタリング (芝1勝クラス以上 & ダート1勝クラス以上)
    # 条件クラス in [1, 2, 3, 9] (1勝/2勝/3勝/オープン/重賞)
    turf_mask = (merged["芝ダ障害コード"] == 1) & (merged["KYI_条件クラス"].isin([1, 2, 3, 9]))
    dirt_mask = (merged["芝ダ障害コード"] == 2) & (merged["KYI_条件クラス"].isin([1, 2, 3, 9]))

    df_turf = merged[turf_mask].copy().reset_index(drop=True)
    df_dirt = merged[dirt_mask].copy().reset_index(drop=True)

    turf_races = df_turf["RACE_ID"].drop_duplicates().tolist()
    dirt_races = df_dirt["RACE_ID"].drop_duplicates().tolist()

    print(f"\n【フィルタ適用】")
    print(f"  - 芝  1勝クラス以上: {len(turf_races)} レース (計 {len(df_turf)} 頭)")
    print(f"  - ダート 1勝クラス以上: {len(dirt_races)} レース (計 {len(df_dirt)} 頭)")
    print(f"  - 合計対象レース数   : {len(turf_races) + len(dirt_races)} レース (計 {len(df_turf) + len(df_dirt)} 頭)")

    # 5. 前処理
    mat_turf = normalize_features(df_turf, turf_stats) if len(df_turf) > 0 else np.empty((0, 0))
    mat_dirt = normalize_features(df_dirt, dirt_stats) if len(df_dirt) > 0 else np.empty((0, 0))

    # 6. 推論ループ
    max_horses = 18
    all_horses_results = []
    race_summaries = []

    # 6-A. 芝レースの推論
    for rid in turf_races:
        race_mask_rows = (df_turf["RACE_ID"] == rid)
        race_df = df_turf[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat_turf[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
        place_name = PLACE_DICT.get(place_code, f"場{place_code}")
        race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
        distance = int(r_row.get("距離", 0))
        class_code = int(r_row.get("KYI_条件クラス", 0))
        class_str = CLASS_DICT.get(class_code, f"クラス{class_code}")

        X11 = np.zeros((1, max_horses, 200), dtype=np.float32)
        X25 = np.zeros((1, max_horses, 200), dtype=np.float32)
        X35 = np.zeros((1, max_horses, 200), dtype=np.float32)
        mask = np.zeros((1, max_horses), dtype=np.float32)

        for h_i in range(min(n_horses, max_horses)):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            slot = h_num - 1
            if 0 <= slot < max_horses:
                X11[0, slot] = race_mat[h_i, f11_indices]
                X25[0, slot] = race_mat[h_i, f25_indices]
                X35[0, slot] = race_mat[h_i, f35_indices]
                mask[0, slot] = 1.0

        X11_flat = torch.tensor(X11.reshape(1, -1), dtype=torch.float32, device=device)
        X25_flat = torch.tensor(X25.reshape(1, -1), dtype=torch.float32, device=device)
        X35_flat = torch.tensor(X35.reshape(1, -1), dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits11 = turf_models["net11"](X11_flat, mask=mask_t)
            logits25 = turf_models["net25"](X25_flat, mask=mask_t)
            logits35 = turf_models["net35"](X35_flat, mask=mask_t)

            probs11 = torch.softmax(logits11, dim=-1).cpu().numpy()[0]
            probs25 = torch.softmax(logits25, dim=-1).cpu().numpy()[0]
            probs35 = torch.softmax(logits35, dim=-1).cpu().numpy()[0]
            probs_ens = (probs11 + probs25 + probs35) / 3.0

        horses_data = []
        for h_i in range(n_horses):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            h_name = str(h_row.get("KYI_馬名", h_row.get("馬名", ""))).strip()
            jockey = str(h_row.get("KYI_騎手名", h_row.get("騎手名", ""))).strip()
            weight = float(h_row.get("KYI_負担重量", h_row.get("負担重量", 0.0))) / 10.0 if float(h_row.get("KYI_負担重量", 0)) > 100 else float(h_row.get("KYI_負担重量", 0.0))
            std_odds = float(h_row.get("KYI_基準オッズ", h_row.get("基準オッズ", 0.0)))
            std_pop = int(h_row.get("KYI_基準人気順位", h_row.get("基準人気順位", 0))) if not pd.isna(h_row.get("KYI_基準人気順位", 0)) else 0

            slot = h_num - 1
            p_ens = float(probs_ens[slot]) if 0 <= slot < max_horses else 0.0
            p_m11 = float(probs11[slot]) if 0 <= slot < max_horses else 0.0
            p_m25 = float(probs25[slot]) if 0 <= slot < max_horses else 0.0
            p_m35 = float(probs35[slot]) if 0 <= slot < max_horses else 0.0
            ev = p_ens * std_odds if std_odds > 0 else 0.0

            detail_str = f"M11:{p_m11*100:.1f}% / M25:{p_m25*100:.1f}% / M35:{p_m35*100:.1f}%"

            horses_data.append({
                "track_type": "芝",
                "race_id": rid,
                "place_code": place_code,
                "place": place_name,
                "race_num": race_num,
                "distance": distance,
                "class_name": class_str,
                "horse_num": h_num,
                "horse_name": h_name,
                "jockey": jockey,
                "weight": weight,
                "std_odds": std_odds,
                "std_pop": std_pop,
                "prob_ens": p_ens,
                "model_details": detail_str,
                "prob_m11": p_m11,
                "prob_m25": p_m25,
                "prob_m35": p_m35,
                "prob_m24": np.nan,
                "prob_m49": np.nan,
                "ev": ev
            })

        horses_data.sort(key=lambda x: x["prob_ens"], reverse=True)
        marks = ["◎", "◯", "▲", "△", "△", "★"]
        for rank_idx, h in enumerate(horses_data):
            h["pred_rank"] = rank_idx + 1
            h["mark"] = marks[rank_idx] if rank_idx < len(marks) else "  "
            all_horses_results.append(h)

        top1 = horses_data[0]
        recom_label, recom_grade, recom_short = get_recommendation_label(top1["prob_ens"], top1["ev"])

        race_summaries.append({
            "track_type": "芝",
            "race_id": rid,
            "place_code": place_code,
            "place": place_name,
            "race_num": race_num,
            "distance": distance,
            "class_name": class_str,
            "top1_num": top1["horse_num"],
            "top1_name": top1["horse_name"],
            "top1_prob": top1["prob_ens"],
            "top1_detail": f"{top1['prob_m11']*100:.1f}% / {top1['prob_m25']*100:.1f}% / {top1['prob_m35']*100:.1f}%",
            "top1_odds": top1["std_odds"],
            "top1_ev": top1["ev"],
            "recom_label": recom_label,
            "recom_grade": recom_grade,
            "recom_short": recom_short,
            "horses": horses_data
        })

    # 6-B. ダートレースの推論
    for rid in dirt_races:
        race_mask_rows = (df_dirt["RACE_ID"] == rid)
        race_df = df_dirt[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat_dirt[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
        place_name = PLACE_DICT.get(place_code, f"場{place_code}")
        race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
        distance = int(r_row.get("距離", 0))
        class_code = int(r_row.get("KYI_条件クラス", 0))
        class_str = CLASS_DICT.get(class_code, f"クラス{class_code}")

        X24 = np.zeros((1, max_horses, 200), dtype=np.float32)
        X49 = np.zeros((1, max_horses, 200), dtype=np.float32)
        mask = np.zeros((1, max_horses), dtype=np.float32)

        for h_i in range(min(n_horses, max_horses)):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            slot = h_num - 1
            if 0 <= slot < max_horses:
                X24[0, slot] = race_mat[h_i, f24_indices]
                X49[0, slot] = race_mat[h_i, f49_indices]
                mask[0, slot] = 1.0

        X24_flat = torch.tensor(X24.reshape(1, -1), dtype=torch.float32, device=device)
        X49_flat = torch.tensor(X49.reshape(1, -1), dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits24 = dirt_models["net24"](X24_flat, mask=mask_t)
            logits49 = dirt_models["net49"](X49_flat, mask=mask_t)

            probs24 = torch.softmax(logits24, dim=-1).cpu().numpy()[0]
            probs49 = torch.softmax(logits49, dim=-1).cpu().numpy()[0]
            probs_best = (probs24 + probs49) / 2.0

        horses_data = []
        for h_i in range(n_horses):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            h_name = str(h_row.get("KYI_馬名", h_row.get("馬名", ""))).strip()
            jockey = str(h_row.get("KYI_騎手名", h_row.get("騎手名", ""))).strip()
            weight = float(h_row.get("KYI_負担重量", h_row.get("負担重量", 0.0))) / 10.0 if float(h_row.get("KYI_負担重量", 0)) > 100 else float(h_row.get("KYI_負担重量", 0.0))
            std_odds = float(h_row.get("KYI_基準オッズ", h_row.get("基準オッズ", 0.0)))
            std_pop = int(h_row.get("KYI_基準人気順位", h_row.get("基準人気順位", 0))) if not pd.isna(h_row.get("KYI_基準人気順位", 0)) else 0

            slot = h_num - 1
            p_best = float(probs_best[slot]) if 0 <= slot < max_horses else 0.0
            p_m24 = float(probs24[slot]) if 0 <= slot < max_horses else 0.0
            p_m49 = float(probs49[slot]) if 0 <= slot < max_horses else 0.0
            ev = p_best * std_odds if std_odds > 0 else 0.0

            detail_str = f"M24:{p_m24*100:.1f}% / M49:{p_m49*100:.1f}%"

            horses_data.append({
                "track_type": "ダート",
                "race_id": rid,
                "place_code": place_code,
                "place": place_name,
                "race_num": race_num,
                "distance": distance,
                "class_name": class_str,
                "horse_num": h_num,
                "horse_name": h_name,
                "jockey": jockey,
                "weight": weight,
                "std_odds": std_odds,
                "std_pop": std_pop,
                "prob_ens": p_best,
                "model_details": detail_str,
                "prob_m11": np.nan,
                "prob_m25": np.nan,
                "prob_m35": np.nan,
                "prob_m24": p_m24,
                "prob_m49": p_m49,
                "ev": ev
            })

        horses_data.sort(key=lambda x: x["prob_ens"], reverse=True)
        marks = ["◎", "◯", "▲", "△", "△", "★"]
        for rank_idx, h in enumerate(horses_data):
            h["pred_rank"] = rank_idx + 1
            h["mark"] = marks[rank_idx] if rank_idx < len(marks) else "  "
            all_horses_results.append(h)

        top1 = horses_data[0]
        recom_label, recom_grade, recom_short = get_recommendation_label(top1["prob_ens"], top1["ev"])

        race_summaries.append({
            "track_type": "ダート",
            "race_id": rid,
            "place_code": place_code,
            "place": place_name,
            "race_num": race_num,
            "distance": distance,
            "class_name": class_str,
            "top1_num": top1["horse_num"],
            "top1_name": top1["horse_name"],
            "top1_prob": top1["prob_ens"],
            "top1_detail": f"{top1['prob_m24']*100:.1f}% / {top1['prob_m49']*100:.1f}%",
            "top1_odds": top1["std_odds"],
            "top1_ev": top1["ev"],
            "recom_label": recom_label,
            "recom_grade": recom_grade,
            "recom_short": recom_short,
            "horses": horses_data
        })

    # レース一覧を競馬場コード・レース番号順にソート (自然なタイムテーブル順)
    race_summaries.sort(key=lambda x: (x["place_code"], x["race_num"]))

    # 7. コンソール表示 (統合一覧)
    print("\n" + "=" * 130)
    print("  ★ JRA 芝・ダート 1勝クラス以上 全レース統合予想サマリー ★")
    print("=" * 130)
    print(f"{'場・R':<8} | {'種別':<4} | {'距離・クラス':<18} | {'本命◎ (馬番・馬名)':<24} | {'ENS勝率':<8} | {'モデル別内訳':<24} | {'基準オッズ':<8} | {'期待値(EV)':<9} | {'推奨度'}")
    print("-" * 130)

    for r in race_summaries:
        r_str = f"{r['place']}{r['race_num']:2d}R"
        t_str = f"{r['track_type']}"
        prefix = "芝" if r["track_type"] == "芝" else "ダ"
        c_str = f"{prefix}{r['distance']}m ({r['class_name'][:6]})"
        top_str = f"◎ {r['top1_num']:2d} {r['top1_name'][:9]}"
        prob_str = f"{r['top1_prob']*100:5.1f}%"
        sub_str = f"{r['top1_detail']}"
        odds_str = f"{r['top1_odds']:4.1f}倍"
        ev_str = f"{r['top1_ev']:4.2f}"
        recom = r["recom_label"]

        print(f"{r_str:<8} | {t_str:<4} | {c_str:<18} | {top_str:<24} | {prob_str:<8} | {sub_str:<24} | {odds_str:<8} | {ev_str:<9} | {recom}")

    # ピックアップ (推奨度3の激熱レース)
    hot_races = [r for r in race_summaries if r["recom_grade"] >= 3]
    print("\n" + "=" * 130)
    print("  ★★★ 本日の超激熱レース ピックアップ (激熱鉄板 & 爆穴妙味) ★★★")
    print("=" * 130)
    if hot_races:
        for r in hot_races:
            prefix = "芝" if r["track_type"] == "芝" else "ダ"
            print(f"  ▶ 【{r['place']}{r['race_num']}R ({r['track_type']}{r['distance']}m)】 {r['recom_label']}")
            print(f"      本命: ◎ {r['top1_num']}番 {r['top1_name']} | 予測勝率: {r['top1_prob']*100:.1f}% | 基準オッズ: {r['top1_odds']:.1f}倍 | 期待値: {r['top1_ev']:.2f}")
    else:
        print("  該当レースはありません。")
    print("=" * 130)

    # 8. ファイル保存
    date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV 出力
    df_out = pd.DataFrame(all_horses_results)
    out_csv_path = os.path.join(args.output_dir, f"predict_unified_ensemble_res_{date_tag}.csv")
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    # Markdown 出力
    out_md_path = os.path.join(args.output_dir, f"predict_unified_ensemble_res_{date_tag}.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# JRA 芝・ダート 1勝クラス以上 完全統合アンサンブル予想レポート\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 使用モデル:\n")
        f.write(f"  - **芝 レース**  : 神アンサンブル [Model 11 + Model 25 + Model 35] (未見回収率 **112.90%** / 検証 **124.66%**)\n")
        f.write(f"  - **ダート レース**: BESTアンサンブル [Model 24 + Model 49]       (未見回収率 **117.18%** / 検証 **116.79%**)\n")
        f.write(f"- 対象レース数: **全 {len(race_summaries)} レース** (芝: {len(turf_races)} レース, ダート: {len(dirt_races)} レース / 計 {len(all_horses_results)} 頭)\n\n")

        # ピックアップ
        f.write("## ★★★ 本日の激熱＆超妙味 推奨レース ピックアップ ★★★\n\n")
        if hot_races:
            f.write("| レース | 種別・条件 | 本命馬 | 勝率 (ENS) | 基準オッズ | 期待値 (EV) | 推奨分類 |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |\n")
            for r in hot_races:
                prefix = "芝" if r["track_type"] == "芝" else "ダ"
                f.write(f"| **{r['place']}{r['race_num']}R** | {r['track_type']}{r['distance']}m ({r['class_name']}) | **◎ {r['top1_num']} {r['top1_name']}** | **{r['top1_prob']*100:.1f}%** | {r['top1_odds']:.1f}倍 | **{r['top1_ev']:.2f}** | **{r['recom_short']}** |\n")
            f.write("\n")
        else:
            f.write("該当レースなし\n\n")

        # 総合サマリー
        f.write("## 1. 全対象レース 総合予想一覧 (タイムテーブル順)\n\n")
        f.write("| レース | 種別 | 距離・クラス | 本命馬 (馬番・馬名) | 勝率 (ENS) | 構成モデル内訳 | 基準オッズ | 期待値 (EV) | 推奨度 |\n")
        f.write("| :--- | :---: | :--- | :--- | :---: | :--- | :---: | :---: | :--- |\n")

        for r in race_summaries:
            prefix = "芝" if r["track_type"] == "芝" else "ダ"
            sub_title = "M11/25/35" if r["track_type"] == "芝" else "M24/49"
            f.write(f"| **{r['place']}{r['race_num']}R** | {r['track_type']} | {prefix}{r['distance']}m ({r['class_name']}) | **◎ {r['top1_num']} {r['top1_name']}** | **{r['top1_prob']*100:.1f}%** | {sub_title}: {r['top1_detail']} | {r['top1_odds']:.1f}倍 | **{r['top1_ev']:.2f}** | {r['recom_short']} |\n")

        # 各レース詳細
        f.write("\n## 2. 各レース全頭 詳細予想表\n\n")
        for r in race_summaries:
            prefix = "芝" if r["track_type"] == "芝" else "ダ"
            model_head = "M11勝率 | M25勝率 | M35勝率" if r["track_type"] == "芝" else "M24勝率 | M49勝率"
            model_align = ":---: | :---: | :---:" if r["track_type"] == "芝" else ":---: | :---:"

            f.write(f"### {r['place']} {r['race_num']}R: {r['track_type']}{r['distance']}m {r['class_name']} ({r['recom_label']})\n\n")
            f.write(f"| 印 | 順位 | 馬番 | 馬名 | 斤量 | 騎手 | 勝率 (ENS) | {model_head} | 基準オッズ | 期待値 (EV) |\n")
            f.write(f"| :---: | :---: | :---: | :--- | :---: | :--- | :---: | {model_align} | :---: | :---: |\n")

            for h in r["horses"]:
                if r["track_type"] == "芝":
                    sub_vals = f"{h['prob_m11']*100:.1f}% | {h['prob_m25']*100:.1f}% | {h['prob_m35']*100:.1f}%"
                else:
                    sub_vals = f"{h['prob_m24']*100:.1f}% | {h['prob_m49']*100:.1f}%"

                f.write(f"| {h['mark']} | {h['pred_rank']} | {h['horse_num']} | {h['horse_name']} | {h['weight']:.1f} | {h['jockey']} | **{h['prob_ens']*100:.1f}%** | {sub_vals} | {h['std_odds']:.1f}倍 | **{h['ev']:.2f}** |\n")
            f.write("\n")

    print("\n" + "=" * 115)
    print("  ★ 芝・ダート統合アンサンブル予想が正常に完了しました！ ★")
    print(f"  Markdown レポート: {out_md_path}")
    print(f"  CSV 出力ファイル : {out_csv_path}")
    print("=" * 115)

if __name__ == "__main__":
    main()
