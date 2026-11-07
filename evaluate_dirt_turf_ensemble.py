"""
evaluate_dirt_turf_ensemble.py
========================================================================
JRA ダート＆芝 1勝クラス以上 アンサンブルモデル 回収率・的中率 汎用検証プログラム
・ダート1勝クラス以上: JRA BESTモデル [Model 24 + Model 49]
・芝1勝クラス以上: 神アンサンブル [Model 11 + Model 25 + Model 35]

【使用例】
# 1. 2026年10月3日以降〜最新までの全レースを検証 (デフォルト)
python3.11 evaluate_dirt_turf_ensemble.py

# 2. 期間を指定して検証 (例: 10月24日〜11月1日)
python3.11 evaluate_dirt_turf_ensemble.py --start_date 20261024 --end_date 20261101

# 3. 特定の日付のみをピンポイントで検証 (例: 10/31と11/1)
python3.11 evaluate_dirt_turf_ensemble.py --target_dates 20261031,20261101

# 4. ダートのみ、または芝のみを検証
python3.11 evaluate_dirt_turf_ensemble.py --category dirt
python3.11 evaluate_dirt_turf_ensemble.py --category turf
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import get_device
from src.models.base_nn import RaceLevelStage1NN
from predict_best_ensemble import load_best_models, PLACE_DICT, CLASS_DICT
from predict_turf_ensemble_m11_m25_m35 import load_turf_models


def parse_args():
    parser = argparse.ArgumentParser(
        description="JRA ダート&芝 1勝クラス以上 アンサンブル回収率・的中率 汎用検証スクリプト"
    )
    parser.add_argument(
        "--start_date",
        type=int,
        default=20261003,
        help="検証開始日 (YYYYMMDD形式、デフォルト: 20261003)",
    )
    parser.add_argument(
        "--end_date",
        type=int,
        default=None,
        help="検証終了日 (YYYYMMDD形式、指定なしの場合は最新日まで)",
    )
    parser.add_argument(
        "--target_dates",
        type=str,
        default=None,
        help="特定の日付のみを検証する場合、カンマ区切りで指定 (例: 20261031,20261101)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "dirt", "turf"],
        help="検証対象カテゴリ ('all': 両方, 'dirt': ダートのみ, 'turf': 芝のみ)",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/2026.csv",
        help="出走馬・結果CSVのパス (デフォルト: Data/RESULT/MAIN/2026.csv)",
    )
    parser.add_argument(
        "--bac_path",
        type=str,
        default="/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/BAC_KAB.csv",
        help="レース情報CSVのパス (デフォルト: Data/RESULT/MAIN/BAC_KAB.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CURRENT_DIR,
        help="結果CSVの保存先ディレクトリ",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="出力CSVファイル名 (省略時は自動生成)",
    )
    parser.add_argument(
        "--hide_details",
        action="store_true",
        help="レースごとの詳細一覧テーブルの表示を省略する",
    )
    return parser.parse_args()


def evaluate_metrics(name: str, df: pd.DataFrame):
    """単勝・複勝の回収率・的中率を算出"""
    n = len(df)
    if n == 0:
        print(f"\n【{name} 成績】 対象レースなし")
        return {"n": 0}

    w_hits = int((df["top1_actual_rank"] == 1).sum())
    w_rate = w_hits / n
    w_pay = float(df["top1_win_payout"].fillna(0.0).sum())
    w_invest = n * 100
    w_roi = (w_pay / w_invest) * 100.0 if w_invest > 0 else 0.0

    p_hits = int((df["top1_actual_rank"] <= 3).sum())
    p_rate = p_hits / n
    p_pay = float(df["top1_place_payout"].fillna(0.0).sum())
    p_roi = (p_pay / w_invest) * 100.0 if w_invest > 0 else 0.0

    print(f"\n【{name} 成績】(確定 {n} レース)")
    print(
        f"  ■ 単勝: 的中 {w_hits:2d}/{n:2d} ({w_rate*100:5.2f}%) | "
        f"投資 {w_invest:6d}円 | 払戻 {int(w_pay):6d}円 | 回収率: {w_roi:6.2f}% (収支: {int(w_pay - w_invest):+d}円)"
    )
    print(
        f"  ■ 複勝: 的中 {p_hits:2d}/{n:2d} ({p_rate*100:5.2f}%) | "
        f"投資 {w_invest:6d}円 | 払戻 {int(p_pay):6d}円 | 回収率: {p_roi:6.2f}% (収支: {int(p_pay - w_invest):+d}円)"
    )
    return {
        "n": n,
        "w_hits": w_hits,
        "w_rate": w_rate,
        "w_pay": w_pay,
        "w_roi": w_roi,
        "p_hits": p_hits,
        "p_rate": p_rate,
        "p_pay": p_pay,
        "p_roi": p_roi,
    }


def main():
    args = parse_args()
    device = get_device()

    print("=" * 105)
    print("  ★ JRA ダート＆芝 1勝クラス以上 BESTアンサンブル 汎用回収率・的中率検証 ★")
    print(f"  実行デバイス: {device}")
    print("=" * 105)

    # 1. データロード
    if not os.path.exists(args.target_path):
        print(f"Error: {args.target_path} が見つかりません。")
        sys.exit(1)
    if not os.path.exists(args.bac_path):
        print(f"Error: {args.bac_path} が見つかりません。")
        sys.exit(1)

    print(f"出走馬・結果データ ロード中: {args.target_path}...")
    df_target = pd.read_csv(args.target_path, low_memory=False)

    print(f"レース情報データ ロード中: {args.bac_path}...")
    df_bac = pd.read_csv(args.bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)

    # RACE_ID 付与
    target_rkeys = df_target["KYI_RACE_KEY"].astype(str).str.zfill(10)
    df_target["RACE_ID"] = target_rkeys.str[:8]
    df_target["KYI_RACE_KEY"] = target_rkeys

    # BAC とマージ
    bac_cols_to_merge = [
        c for c in df_bac.columns if c == "RACE_ID" or c not in df_target.columns
    ]
    merged = pd.merge(df_target, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")
    merged["年月日_int"] = (
        pd.to_numeric(merged["年月日"], errors="coerce").fillna(0).astype(int)
    )

    # 日付フィルタ
    if args.target_dates:
        t_dates = [int(d.strip()) for d in args.target_dates.split(",") if d.strip()]
        date_mask = merged["年月日_int"].isin(t_dates)
        date_desc = f"指定日付: {t_dates}"
        file_date_tag = "_".join(str(d) for d in t_dates)
    else:
        date_mask = merged["年月日_int"] >= args.start_date
        if args.end_date:
            date_mask = date_mask & (merged["年月日_int"] <= args.end_date)
            date_desc = f"期間: {args.start_date} 〜 {args.end_date}"
            file_date_tag = f"{args.start_date}_{args.end_date}"
        else:
            date_desc = f"期間: {args.start_date} 以降最新まで"
            file_date_tag = f"{args.start_date}_latest"

    merged_filtered = merged[date_mask].copy().reset_index(drop=True)
    all_dates = sorted(merged_filtered["年月日_int"].unique().tolist())
    print(f"\n【対象日程】{date_desc} (計 {len(all_dates)} 開催日: {all_dates})")

    dirt_results = []
    turf_results = []

    # =========================================================================
    # PART 1: ダート 1勝クラス以上 (BESTモデル: Model 24 + Model 49)
    # =========================================================================
    if args.category in ["all", "dirt"]:
        print("\n" + "-" * 80)
        print("【ダート 1勝クラス以上】BESTモデル (Model 24 + Model 49) 推論中...")
        print("-" * 80)

        stats_dirt_path = os.path.join(
            CURRENT_DIR, "artifacts_dirt_1win/dirt_1win_preprocessor_stats.pkl"
        )
        with open(stats_dirt_path, "rb") as f:
            stats_dirt = pickle.load(f)

        d_all_cols = stats_dirt["all_feature_cols"]
        d_col_to_idx = stats_dirt["col_to_idx"]
        d_bac_cols = stats_dirt["bac_feature_cols"]
        d_medians = stats_dirt["medians"]
        d_means = stats_dirt["means"]
        d_stds = stats_dirt["stds"]

        dirt_models = load_best_models(CURRENT_DIR, device)
        f24_cols = list(d_bac_cols) + dirt_models["m24_yearly"]
        f49_cols = list(d_bac_cols) + dirt_models["m49_yearly"]
        f24_idx = [d_col_to_idx[c] for c in f24_cols]
        f49_idx = [d_col_to_idx[c] for c in f49_cols]

        dirt_race_max = (
            merged_filtered[merged_filtered["芝ダ障害コード"] == 2]
            .groupby("RACE_ID")["KYI_条件クラス"]
            .max()
        )
        dirt_target_races = dirt_race_max[dirt_race_max.isin([1, 2, 3, 9])].index.tolist()
        merged_dirt = (
            merged_filtered[merged_filtered["RACE_ID"].isin(dirt_target_races)]
            .copy()
            .reset_index(drop=True)
        )

        print(
            f"ダート対象レース数: {len(dirt_target_races)} レース (計 {len(merged_dirt)} 頭)"
        )

        if len(dirt_target_races) > 0:
            mat_dirt = np.zeros((len(merged_dirt), len(d_all_cols)), dtype=np.float32)
            for i, c in enumerate(d_all_cols):
                if c in merged_dirt.columns:
                    mat_dirt[:, i] = (
                        pd.to_numeric(merged_dirt[c], errors="coerce")
                        .fillna(np.nan)
                        .values
                    )
                else:
                    mat_dirt[:, i] = np.nan

            for col_i in range(len(d_all_cols)):
                nan_m = np.isnan(mat_dirt[:, col_i])
                if np.any(nan_m):
                    mat_dirt[nan_m, col_i] = d_medians[col_i]

            mat_dirt = np.clip((mat_dirt - d_means) / d_stds, -10.0, 10.0)
            mat_dirt = np.nan_to_num(mat_dirt, nan=0.0)

            max_horses = 18
            for rid in dirt_target_races:
                race_mask_rows = merged_dirt["RACE_ID"] == rid
                race_df = merged_dirt[race_mask_rows].copy().reset_index(drop=True)
                race_mat = mat_dirt[race_mask_rows]
                n_horses = len(race_df)

                r_row = race_df.iloc[0]
                ymd = int(r_row["年月日_int"])
                place_code = str(
                    r_row.get("KYI_場コード", str(r_row.get("場コード", "")))
                ).zfill(2)
                place_name = PLACE_DICT.get(place_code, f"場{place_code}")
                race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
                distance = int(r_row.get("距離", 0))
                class_code = int(race_df["KYI_条件クラス"].max())
                class_str = CLASS_DICT.get(class_code, f"クラス{class_code}")

                X24 = np.zeros((1, max_horses, 200), dtype=np.float32)
                X49 = np.zeros((1, max_horses, 200), dtype=np.float32)
                mask = np.zeros((1, max_horses), dtype=np.float32)

                for h_i in range(min(n_horses, max_horses)):
                    h_row = race_df.iloc[h_i]
                    h_num = int(h_row.get("KYI_馬番", h_i + 1))
                    slot = h_num - 1
                    if 0 <= slot < max_horses:
                        X24[0, slot] = race_mat[h_i, f24_idx]
                        X49[0, slot] = race_mat[h_i, f49_idx]
                        mask[0, slot] = 1.0

                X24_flat = torch.tensor(
                    X24.reshape(1, -1), dtype=torch.float32, device=device
                )
                X49_flat = torch.tensor(
                    X49.reshape(1, -1), dtype=torch.float32, device=device
                )
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
                    actual_rank = h_row.get("Target_着順", np.nan)
                    final_odds = h_row.get("Target_確定単勝オッズ", np.nan)
                    final_pop = h_row.get("Target_確定単勝人気順位", np.nan)
                    win_payout = h_row.get("Target_単勝", np.nan)
                    place_payout = h_row.get("Target_複勝", np.nan)
                    std_odds = float(h_row.get("KYI_基準オッズ", 0.0))

                    slot = h_num - 1
                    p_best = (
                        float(probs_best[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    p_m24 = (
                        float(probs24[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    p_m49 = (
                        float(probs49[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    ev_std = p_best * std_odds if std_odds > 0 else 0.0

                    horses_data.append({
                        "horse_num": h_num,
                        "horse_name": h_name,
                        "actual_rank": actual_rank,
                        "final_odds": final_odds,
                        "final_pop": final_pop,
                        "win_payout": win_payout,
                        "place_payout": place_payout,
                        "std_odds": std_odds,
                        "prob": p_best,
                        "prob_m24": p_m24,
                        "prob_m49": p_m49,
                        "ev_std": ev_std,
                    })

                horses_data.sort(key=lambda x: x["prob"], reverse=True)
                top1 = horses_data[0]
                top_m24 = max(horses_data, key=lambda x: x["prob_m24"])
                top_m49 = max(horses_data, key=lambda x: x["prob_m49"])

                dirt_results.append({
                    "race_id": rid,
                    "category": "ダート",
                    "ymd": ymd,
                    "place": place_name,
                    "race_num": race_num,
                    "distance": distance,
                    "class_name": class_str,
                    "top1_num": top1["horse_num"],
                    "top1_name": top1["horse_name"],
                    "top1_prob": top1["prob"],
                    "top1_std_odds": top1["std_odds"],
                    "top1_ev_std": top1["ev_std"],
                    "top1_actual_rank": top1["actual_rank"],
                    "top1_final_odds": top1["final_odds"],
                    "top1_final_pop": top1["final_pop"],
                    "top1_win_payout": top1["win_payout"],
                    "top1_place_payout": top1["place_payout"],
                    "top_m1_num": top_m24["horse_num"],
                    "top_m1_rank": top_m24["actual_rank"],
                    "top_m1_win_payout": top_m24["win_payout"],
                    "top_m2_num": top_m49["horse_num"],
                    "top_m2_rank": top_m49["actual_rank"],
                    "top_m2_win_payout": top_m49["win_payout"],
                    "top_m3_num": np.nan,
                    "top_m3_rank": np.nan,
                    "top_m3_win_payout": np.nan,
                })

    df_res_dirt = pd.DataFrame(dirt_results)

    # =========================================================================
    # PART 2: 芝 1勝クラス以上 (神アンサンブル: Model 11 + Model 25 + Model 35)
    # =========================================================================
    if args.category in ["all", "turf"]:
        print("\n" + "-" * 80)
        print("【芝 1勝クラス以上】神アンサンブル (Model 11 + Model 25 + Model 35) 推論中...")
        print("-" * 80)

        stats_turf_path = os.path.join(
            CURRENT_DIR, "artifacts_turf_1win/turf_1win_preprocessor_stats.pkl"
        )
        with open(stats_turf_path, "rb") as f:
            stats_turf = pickle.load(f)

        t_all_cols = stats_turf["all_feature_cols"]
        t_col_to_idx = stats_turf["col_to_idx"]
        t_bac_cols = stats_turf["bac_feature_cols"]
        t_medians = stats_turf["medians"]
        t_means = stats_turf["means"]
        t_stds = stats_turf["stds"]

        turf_models = load_turf_models(CURRENT_DIR, device)
        f11_cols = list(t_bac_cols) + turf_models["m11_yearly"]
        f25_cols = list(t_bac_cols) + turf_models["m25_yearly"]
        f35_cols = list(t_bac_cols) + turf_models["m35_yearly"]
        f11_idx = [t_col_to_idx[c] for c in f11_cols]
        f25_idx = [t_col_to_idx[c] for c in f25_cols]
        f35_idx = [t_col_to_idx[c] for c in f35_cols]

        turf_race_max = (
            merged_filtered[merged_filtered["芝ダ障害コード"] == 1]
            .groupby("RACE_ID")["KYI_条件クラス"]
            .max()
        )
        turf_target_races = turf_race_max[turf_race_max.isin([1, 2, 3, 9])].index.tolist()
        merged_turf = (
            merged_filtered[merged_filtered["RACE_ID"].isin(turf_target_races)]
            .copy()
            .reset_index(drop=True)
        )

        print(
            f"芝対象レース数: {len(turf_target_races)} レース (計 {len(merged_turf)} 頭)"
        )

        if len(turf_target_races) > 0:
            mat_turf = np.zeros((len(merged_turf), len(t_all_cols)), dtype=np.float32)
            for i, c in enumerate(t_all_cols):
                if c in merged_turf.columns:
                    mat_turf[:, i] = (
                        pd.to_numeric(merged_turf[c], errors="coerce")
                        .fillna(np.nan)
                        .values
                    )
                else:
                    mat_turf[:, i] = np.nan

            for col_i in range(len(t_all_cols)):
                nan_m = np.isnan(mat_turf[:, col_i])
                if np.any(nan_m):
                    mat_turf[nan_m, col_i] = t_medians[col_i]

            mat_turf = np.clip((mat_turf - t_means) / t_stds, -10.0, 10.0)
            mat_turf = np.nan_to_num(mat_turf, nan=0.0)

            max_horses = 18
            for rid in turf_target_races:
                race_mask_rows = merged_turf["RACE_ID"] == rid
                race_df = merged_turf[race_mask_rows].copy().reset_index(drop=True)
                race_mat = mat_turf[race_mask_rows]
                n_horses = len(race_df)

                r_row = race_df.iloc[0]
                ymd = int(r_row["年月日_int"])
                place_code = str(
                    r_row.get("KYI_場コード", str(r_row.get("場コード", "")))
                ).zfill(2)
                place_name = PLACE_DICT.get(place_code, f"場{place_code}")
                race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
                distance = int(r_row.get("距離", 0))
                class_code = int(race_df["KYI_条件クラス"].max())
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
                        X11[0, slot] = race_mat[h_i, f11_idx]
                        X25[0, slot] = race_mat[h_i, f25_idx]
                        X35[0, slot] = race_mat[h_i, f35_idx]
                        mask[0, slot] = 1.0

                X11_flat = torch.tensor(
                    X11.reshape(1, -1), dtype=torch.float32, device=device
                )
                X25_flat = torch.tensor(
                    X25.reshape(1, -1), dtype=torch.float32, device=device
                )
                X35_flat = torch.tensor(
                    X35.reshape(1, -1), dtype=torch.float32, device=device
                )
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
                    actual_rank = h_row.get("Target_着順", np.nan)
                    final_odds = h_row.get("Target_確定単勝オッズ", np.nan)
                    final_pop = h_row.get("Target_確定単勝人気順位", np.nan)
                    win_payout = h_row.get("Target_単勝", np.nan)
                    place_payout = h_row.get("Target_複勝", np.nan)
                    std_odds = float(h_row.get("KYI_基準オッズ", 0.0))

                    slot = h_num - 1
                    p_ens = (
                        float(probs_ens[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    p_m11 = (
                        float(probs11[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    p_m25 = (
                        float(probs25[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    p_m35 = (
                        float(probs35[slot]) if 0 <= slot < max_horses else 0.0
                    )
                    ev_std = p_ens * std_odds if std_odds > 0 else 0.0

                    horses_data.append({
                        "horse_num": h_num,
                        "horse_name": h_name,
                        "actual_rank": actual_rank,
                        "final_odds": final_odds,
                        "final_pop": final_pop,
                        "win_payout": win_payout,
                        "place_payout": place_payout,
                        "std_odds": std_odds,
                        "prob": p_ens,
                        "prob_m11": p_m11,
                        "prob_m25": p_m25,
                        "prob_m35": p_m35,
                        "ev_std": ev_std,
                    })

                horses_data.sort(key=lambda x: x["prob"], reverse=True)
                top1 = horses_data[0]
                top_m11 = max(horses_data, key=lambda x: x["prob_m11"])
                top_m25 = max(horses_data, key=lambda x: x["prob_m25"])
                top_m35 = max(horses_data, key=lambda x: x["prob_m35"])

                turf_results.append({
                    "race_id": rid,
                    "category": "芝",
                    "ymd": ymd,
                    "place": place_name,
                    "race_num": race_num,
                    "distance": distance,
                    "class_name": class_str,
                    "top1_num": top1["horse_num"],
                    "top1_name": top1["horse_name"],
                    "top1_prob": top1["prob"],
                    "top1_std_odds": top1["std_odds"],
                    "top1_ev_std": top1["ev_std"],
                    "top1_actual_rank": top1["actual_rank"],
                    "top1_final_odds": top1["final_odds"],
                    "top1_final_pop": top1["final_pop"],
                    "top1_win_payout": top1["win_payout"],
                    "top1_place_payout": top1["place_payout"],
                    "top_m1_num": top_m11["horse_num"],
                    "top_m1_rank": top_m11["actual_rank"],
                    "top_m1_win_payout": top_m11["win_payout"],
                    "top_m2_num": top_m25["horse_num"],
                    "top_m2_rank": top_m25["actual_rank"],
                    "top_m2_win_payout": top_m25["win_payout"],
                    "top_m3_num": top_m35["horse_num"],
                    "top_m3_rank": top_m35["actual_rank"],
                    "top_m3_win_payout": top_m35["win_payout"],
                })

    df_res_turf = pd.DataFrame(turf_results)

    # =========================================================================
    # PART 3: 成績集計 & レポート表示
    # =========================================================================
    dfs_to_concat = []
    if len(df_res_dirt) > 0:
        dfs_to_concat.append(df_res_dirt)
    if len(df_res_turf) > 0:
        dfs_to_concat.append(df_res_turf)

    if len(dfs_to_concat) == 0:
        print("対象条件に合致するレースが存在しませんでした。")
        return

    df_combined = pd.concat(dfs_to_concat, ignore_index=True)

    # 着順確定済みレース（Target_着順がnotna）で回収率・的中率を算出
    df_eval = df_combined[df_combined["top1_actual_rank"].notna()].copy()
    unconfirmed = len(df_combined) - len(df_eval)

    print("\n" + "=" * 95)
    print(f"  ★ レース成績結果サマリー ({date_desc}) ★")
    if unconfirmed > 0:
        print(f"  ※ 全 {len(df_combined)} レース中、結果確定: {len(df_eval)} レース (未確定: {unconfirmed} レース)")
    print("=" * 95)

    if args.category in ["all", "dirt"] and len(df_res_dirt) > 0:
        evaluate_metrics(
            "ダート 1勝クラス以上 (BESTモデル: M24+M49)",
            df_res_dirt[df_res_dirt["top1_actual_rank"].notna()],
        )
    if args.category in ["all", "turf"] and len(df_res_turf) > 0:
        evaluate_metrics(
            "芝 1勝クラス以上 (神アンサンブル: M11+M25+M35)",
            df_res_turf[df_res_turf["top1_actual_rank"].notna()],
        )
    if args.category == "all" and len(dfs_to_concat) == 2:
        evaluate_metrics("【総合計】ダート ＋ 芝 全対象レース", df_eval)

    # 日別成績一覧
    if len(all_dates) > 1:
        if args.category == "all" and len(dfs_to_concat) == 2:
            print("\n" + "=" * 105)
            print("【開催日別 成績推移 (ダート / 芝 / 総合計)】")
            print(f"{'日付':<10} | {'ダート(単回/複回)':<20} | {'芝(単回/複回)':<20} | {'総合計 レース数・単的中・単回収・複回収'}")
            print("-" * 105)
            for ymd, g in df_eval.groupby("ymd"):
                gd = g[g["category"] == "ダート"]
                gt = g[g["category"] == "芝"]
                
                # ダート
                nd = len(gd)
                wd_roi = (gd["top1_win_payout"].fillna(0).sum() / (nd * 100)) * 100 if nd > 0 else 0.0
                pd_roi = (gd["top1_place_payout"].fillna(0).sum() / (nd * 100)) * 100 if nd > 0 else 0.0
                d_str = f"{nd:2d}R ({wd_roi:5.1f}% / {pd_roi:5.1f}%)" if nd > 0 else "---"
                
                # 芝
                nt = len(gt)
                wt_roi = (gt["top1_win_payout"].fillna(0).sum() / (nt * 100)) * 100 if nt > 0 else 0.0
                pt_roi = (gt["top1_place_payout"].fillna(0).sum() / (nt * 100)) * 100 if nt > 0 else 0.0
                t_str = f"{nt:2d}R ({wt_roi:5.1f}% / {pt_roi:5.1f}%)" if nt > 0 else "---"
                
                # 総合
                n = len(g)
                w_hits = (g["top1_actual_rank"] == 1).sum()
                w_pay = g["top1_win_payout"].fillna(0).sum()
                w_roi = (w_pay / (n * 100)) * 100 if n > 0 else 0.0
                p_hits = (g["top1_actual_rank"] <= 3).sum()
                p_pay = g["top1_place_payout"].fillna(0).sum()
                p_roi = (p_pay / (n * 100)) * 100 if n > 0 else 0.0
                
                print(f"{ymd:<10} | {d_str:<20} | {t_str:<20} | {n:2d}R  単:{w_hits:2d}/{n:2d}({w_roi:5.1f}%) 複:{p_hits:2d}/{n:2d}({p_roi:5.1f}%)")
        else:
            print("\n" + "=" * 95)
            print("【開催日別 成績一覧】")
            print(f"{'日付':<10} | {'レース数':<6} | {'単勝的中':<10} | {'単勝払戻':<8} | {'単勝回収率':<9} | {'複勝的中':<10} | {'複勝払戻':<8} | {'複勝回収率'}")
            print("-" * 95)
            for ymd, g in df_eval.groupby("ymd"):
                g_n = len(g)
                g_w_hits = (g["top1_actual_rank"] == 1).sum()
                g_w_pay = g["top1_win_payout"].fillna(0.0).sum()
                g_w_roi = (g_w_pay / (g_n * 100)) * 100.0 if g_n > 0 else 0.0
                g_p_hits = (g["top1_actual_rank"] <= 3).sum()
                g_p_pay = g["top1_place_payout"].fillna(0.0).sum()
                g_p_roi = (g_p_pay / (g_n * 100)) * 100.0 if g_n > 0 else 0.0
                print(
                    f"{ymd:<10} | {g_n:2d}R    | {g_w_hits:2d}/{g_n:2d} ({g_w_hits/g_n*100:4.1f}%) | "
                    f"{int(g_w_pay):>6}円 | {g_w_roi:6.1f}%   | {g_p_hits:2d}/{g_n:2d} ({g_p_hits/g_n*100:4.1f}%) | {int(g_p_pay):>6}円 | {g_p_roi:6.1f}%"
                )

    # 期待値 (EV) フィルタ別
    print("\n" + "=" * 95)
    print("【期待値(EV_std) フィルタ別 単勝成績】")
    if args.category == "all" and len(dfs_to_concat) == 2:
        print(f"{'EV条件':<10} | {'総合 回収率 (的中/R数)':<24} | {'ダート 回収率 (的中/R数)':<24} | {'芝 回収率 (的中/R数)'}")
        print("-" * 95)
        for ev_th in [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]:
            # 総合
            g_all = df_eval[df_eval["top1_ev_std"] >= ev_th]
            na = len(g_all)
            ha = (g_all["top1_actual_rank"] == 1).sum() if na > 0 else 0
            roia = (g_all["top1_win_payout"].fillna(0).sum() / (na * 100)) * 100 if na > 0 else 0.0
            a_str = f"{roia:5.1f}% ({ha:2d}/{na:2d}R)" if na > 0 else "---"

            # ダート
            g_d = df_eval[(df_eval["top1_ev_std"] >= ev_th) & (df_eval["category"] == "ダート")]
            nd = len(g_d)
            hd = (g_d["top1_actual_rank"] == 1).sum() if nd > 0 else 0
            roid = (g_d["top1_win_payout"].fillna(0).sum() / (nd * 100)) * 100 if nd > 0 else 0.0
            d_str = f"{roid:5.1f}% ({hd:2d}/{nd:2d}R)" if nd > 0 else "---"

            # 芝
            g_t = df_eval[(df_eval["top1_ev_std"] >= ev_th) & (df_eval["category"] == "芝")]
            nt = len(g_t)
            ht = (g_t["top1_actual_rank"] == 1).sum() if nt > 0 else 0
            roit = (g_t["top1_win_payout"].fillna(0).sum() / (nt * 100)) * 100 if nt > 0 else 0.0
            t_str = f"{roit:5.1f}% ({ht:2d}/{nt:2d}R)" if nt > 0 else "---"

            print(f"EV >= {ev_th:<4.1f} | {a_str:<24} | {d_str:<24} | {t_str}")
    else:
        print(f"{'EV条件':<12} | {'レース数':<6} | {'的中数':<8} | {'的中率':<8} | {'総投資':<8} | {'総払戻':<8} | {'単勝回収率'}")
        print("-" * 95)
        for ev_th in [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]:
            g = df_eval[df_eval["top1_ev_std"] >= ev_th]
            g_n = len(g)
            if g_n > 0:
                hits = (g["top1_actual_rank"] == 1).sum()
                rate = hits / g_n
                pay = g["top1_win_payout"].fillna(0.0).sum()
                invest = g_n * 100
                roi = (pay / invest) * 100.0
                print(
                    f"EV >= {ev_th:<6.1f} | {g_n:2d}R    | {hits:2d}/{g_n:2d}    | {rate*100:5.1f}%  | {invest:>6}円 | {int(pay):>6}円 | {roi:6.1f}%"
                )

    # レース詳細一覧
    if not args.hide_details:
        print("\n" + "=" * 95)
        print("【全対象レース 詳細結果一覧】")
        print("=" * 95)
        for _, r in df_combined.sort_values(
            ["ymd", "category", "place", "race_num"]
        ).iterrows():
            if pd.isna(r["top1_actual_rank"]):
                hit_mark = "・未確定"
                pay_str = ""
                rank_str = "--着"
            else:
                rank_int = int(r["top1_actual_rank"])
                rank_str = f"{rank_int:2d}着"
                if rank_int == 1:
                    hit_mark = "★単的中"
                    pay_str = f"単{int(r['top1_win_payout'])}円 複{int(r['top1_place_payout'])}円"
                elif rank_int <= 3:
                    hit_mark = "◯複的中"
                    pay_str = f"複{int(r['top1_place_payout'])}円"
                else:
                    hit_mark = "×ハズレ"
                    pay_str = ""

            pop_str = (
                f"確定{int(r['top1_final_pop'])}人気"
                if pd.notna(r["top1_final_pop"])
                else "人気未定"
            )
            odds_str = (
                f"{r['top1_final_odds']:4.1f}倍"
                if pd.notna(r["top1_final_odds"])
                else "オッズ未定"
            )

            print(
                f"{r['ymd']} {r['category']} {r['place']}{r['race_num']:2d}R {r['distance']}m {r['class_name'][:6]} | "
                f"本命: {r['top1_num']:2d}番 {r['top1_name'][:8]} (勝率{r['top1_prob']*100:4.1f}%, 基準オッズ{r['top1_std_odds']:4.1f}, EV{r['top1_ev_std']:4.2f}) | "
                f"{pop_str} {odds_str} | 結果: {rank_str} {hit_mark} {pay_str}"
            )

    # CSV出力
    if args.output_csv:
        out_csv = (
            args.output_csv
            if os.path.isabs(args.output_csv)
            else os.path.join(args.output_dir, args.output_csv)
        )
    else:
        out_csv = os.path.join(
            args.output_dir, f"eval_ensemble_{args.category}_{file_date_tag}.csv"
        )

    df_combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n全 {len(df_combined)} レース詳細結果CSV保存完了: {out_csv}")


if __name__ == "__main__":
    main()
