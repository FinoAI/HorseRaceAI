#!/usr/local/bin/python3.11
"""
calc_daily_stats_2025_2026.py
========================================================================
2025年〜2026年最新までのJRA 芝・ダート 1勝クラス以上 アンサンブルモデル評価
各日毎の的中率・回収率の算出、および平均・分散（標準偏差）の統計分析
========================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.utils.helpers import get_device
from src.models.base_nn import RaceLevelStage1NN

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
    net11.load_state_dict(torch.load(os.path.join(art_dir, "model_11.pth"), map_location=device))
    net11.eval()

    net25 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net25.load_state_dict(torch.load(os.path.join(art_dir, "model_25.pth"), map_location=device))
    net25.eval()

    net35 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net35.load_state_dict(torch.load(os.path.join(art_dir, "model_35.pth"), map_location=device))
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
    art_dir = os.path.join(base_dir, "artifacts_dirt_1win")
    best_dir = os.path.join(art_dir, "BEST_MODEL_DIRT_1WIN")
    if not os.path.exists(best_dir):
        best_dir = art_dir

    with open(os.path.join(best_dir, "features_24.json"), "r", encoding="utf-8") as f:
        f24_data = json.load(f)
    with open(os.path.join(best_dir, "features_49.json"), "r", encoding="utf-8") as f:
        f49_data = json.load(f)

    max_horses = 18
    num_features = 200
    input_dim = max_horses * num_features

    net24 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net24.load_state_dict(torch.load(os.path.join(best_dir, "model_24.pth"), map_location=device))
    net24.eval()

    net49 = RaceLevelStage1NN(input_dim=input_dim, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    net49.load_state_dict(torch.load(os.path.join(best_dir, "model_49.pth"), map_location=device))
    net49.eval()

    return {
        "net24": net24,
        "net49": net49,
        "m24_yearly": f24_data["yearly_features"],
        "m49_yearly": f49_data["yearly_features"]
    }

def main():
    device = get_device()
    print("=" * 105)
    print("  ★ 2025年〜2026年最新 JRA 芝・ダート 1勝クラス以上 日別成績・平均・分散 統計分析 ★")
    print(f"  実行デバイス: {device}")
    print("=" * 105)

    # 1. データロード
    p2025 = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/2025.csv"
    p2026 = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/2026.csv"
    bac_path = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/BAC_KAB.csv"

    print(f"出走馬・結果データ ロード中: {p2025} & {p2026}...")
    df25 = pd.read_csv(p2025, low_memory=False)
    df26 = pd.read_csv(p2026, low_memory=False)
    df_target = pd.concat([df25, df26], ignore_index=True)
    del df25, df26
    print(f"出走馬データ結合完了: 計 {len(df_target)} 行")

    print(f"レース情報データ ロード中: {bac_path}...")
    df_bac = pd.read_csv(bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)

    # RACE_ID 付与
    target_rkeys = df_target["KYI_RACE_KEY"].astype(str).str.zfill(10)
    df_target["RACE_ID"] = target_rkeys.str[:8]

    # レース単位の条件クラス & 確定フラグ判定
    race_class_max = df_target.groupby("RACE_ID")["KYI_条件クラス"].max()
    race_is_done = df_target.groupby("RACE_ID")["Target_着順"].apply(lambda s: s.notna().any())

    target_race_ids = set(race_class_max[race_class_max.isin([1, 2, 3, 9])].index) & set(race_is_done[race_is_done == True].index)

    # BACとマージ
    merged_all = pd.merge(df_target[df_target["RACE_ID"].isin(target_race_ids)], df_bac, on="RACE_ID", how="inner")
    merged_all["年月日_int"] = pd.to_numeric(merged_all["年月日"], errors="coerce").fillna(0).astype(int)

    # 2025年1月1日以降の芝・ダート確定レース
    merged_filtered = merged_all[
        (merged_all["年月日_int"] >= 20250101) &
        (merged_all["芝ダ障害コード"].isin([1, 2]))
    ].copy().reset_index(drop=True)

    all_target_races = sorted(list(merged_filtered["RACE_ID"].unique()))
    all_dates = sorted([int(x) for x in merged_filtered["年月日_int"].unique()])
    print(f"【抽出完了】期間: {all_dates[0]} 〜 {all_dates[-1]} (計 {len(all_dates)} 開催日, 対象 {len(all_target_races)} レース)")

    # -------------------------------------------------------------
    # ダート推論
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("【ダート 1勝クラス以上】BESTモデル (Model 24 + Model 49) 推論中...")
    print("-" * 80)
    dirt_models = load_dirt_models(CURRENT_DIR, device)
    with open(os.path.join(CURRENT_DIR, "artifacts_dirt_1win", "dirt_1win_preprocessor_stats.pkl"), "rb") as f:
        d_stats = pickle.load(f)
    d_means = d_stats["means"]
    d_stds = d_stats["stds"]
    d_medians = d_stats["medians"]
    d_all_cols = d_stats["all_feature_cols"]
    d_col_to_idx = d_stats["col_to_idx"]
    d_bac_cols = d_stats["bac_feature_cols"]

    f24_cols = list(d_bac_cols) + dirt_models["m24_yearly"]
    f49_cols = list(d_bac_cols) + dirt_models["m49_yearly"]
    f24_idx = [d_col_to_idx[c] for c in f24_cols]
    f49_idx = [d_col_to_idx[c] for c in f49_cols]

    dirt_target_races = [rid for rid in all_target_races if (merged_filtered[merged_filtered["RACE_ID"] == rid]["芝ダ障害コード"].iloc[0] == 2)]
    merged_dirt = merged_filtered[merged_filtered["RACE_ID"].isin(dirt_target_races)].copy().reset_index(drop=True)
    print(f"ダート対象レース数: {len(dirt_target_races)} レース (計 {len(merged_dirt)} 頭)")

    mat_dirt = np.zeros((len(merged_dirt), len(d_all_cols)), dtype=np.float32)
    for i, c in enumerate(d_all_cols):
        if c in merged_dirt.columns:
            mat_dirt[:, i] = pd.to_numeric(merged_dirt[c], errors="coerce").fillna(np.nan).values
        else:
            mat_dirt[:, i] = np.nan
    for col_i in range(len(d_all_cols)):
        nan_m = np.isnan(mat_dirt[:, col_i])
        if np.any(nan_m):
            mat_dirt[nan_m, col_i] = d_medians[col_i]
    mat_dirt = np.clip((mat_dirt - d_means) / d_stds, -10.0, 10.0)
    mat_dirt = np.nan_to_num(mat_dirt, nan=0.0)

    max_horses = 18
    dirt_race_results = []
    for rid in dirt_target_races:
        race_mask_rows = (merged_dirt["RACE_ID"] == rid)
        race_df = merged_dirt[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat_dirt[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        ymd = int(r_row["年月日_int"])
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
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
            actual_rank = h_row.get("Target_着順", np.nan)
            final_odds = h_row.get("Target_確定単勝オッズ", np.nan)
            final_pop = h_row.get("Target_確定単勝人気順位", np.nan)
            win_payout = h_row.get("Target_単勝", np.nan)
            place_payout = h_row.get("Target_複勝", np.nan)
            std_odds = float(h_row.get("KYI_基準オッズ", 0.0))

            slot = h_num - 1
            p_best = float(probs_best[slot]) if 0 <= slot < max_horses else 0.0
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
                "ev_std": ev_std
            })

        horses_data_sorted = sorted(horses_data, key=lambda x: x["prob"], reverse=True)
        top1 = horses_data_sorted[0]

        dirt_race_results.append({
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
        })

    df_res_dirt = pd.DataFrame(dirt_race_results)
    print(f"ダート推論完了: {len(df_res_dirt)} レース")

    # -------------------------------------------------------------
    # 芝推論
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("【芝 1勝クラス以上】神アンサンブル (Model 11 + Model 25 + Model 35) 推論中...")
    print("-" * 80)
    turf_models = load_turf_models(CURRENT_DIR, device)
    with open(os.path.join(CURRENT_DIR, "artifacts_turf_1win", "turf_1win_preprocessor_stats.pkl"), "rb") as f:
        t_stats = pickle.load(f)
    t_means = t_stats["means"]
    t_stds = t_stats["stds"]
    t_medians = t_stats["medians"]
    t_all_cols = t_stats["all_feature_cols"]
    t_col_to_idx = t_stats["col_to_idx"]
    t_bac_cols = t_stats["bac_feature_cols"]

    f11_cols = list(t_bac_cols) + turf_models["m11_yearly"]
    f25_cols = list(t_bac_cols) + turf_models["m25_yearly"]
    f35_cols = list(t_bac_cols) + turf_models["m35_yearly"]
    f11_idx = [t_col_to_idx[c] for c in f11_cols]
    f25_idx = [t_col_to_idx[c] for c in f25_cols]
    f35_idx = [t_col_to_idx[c] for c in f35_cols]

    turf_target_races = [rid for rid in all_target_races if (merged_filtered[merged_filtered["RACE_ID"] == rid]["芝ダ障害コード"].iloc[0] == 1)]
    merged_turf = merged_filtered[merged_filtered["RACE_ID"].isin(turf_target_races)].copy().reset_index(drop=True)
    print(f"芝対象レース数: {len(turf_target_races)} レース (計 {len(merged_turf)} 頭)")

    mat_turf = np.zeros((len(merged_turf), len(t_all_cols)), dtype=np.float32)
    for i, c in enumerate(t_all_cols):
        if c in merged_turf.columns:
            mat_turf[:, i] = pd.to_numeric(merged_turf[c], errors="coerce").fillna(np.nan).values
        else:
            mat_turf[:, i] = np.nan
    for col_i in range(len(t_all_cols)):
        nan_m = np.isnan(mat_turf[:, col_i])
        if np.any(nan_m):
            mat_turf[nan_m, col_i] = t_medians[col_i]
    mat_turf = np.clip((mat_turf - t_means) / t_stds, -10.0, 10.0)
    mat_turf = np.nan_to_num(mat_turf, nan=0.0)

    turf_race_results = []
    for rid in turf_target_races:
        race_mask_rows = (merged_turf["RACE_ID"] == rid)
        race_df = merged_turf[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat_turf[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        ymd = int(r_row["年月日_int"])
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
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
            actual_rank = h_row.get("Target_着順", np.nan)
            final_odds = h_row.get("Target_確定単勝オッズ", np.nan)
            final_pop = h_row.get("Target_確定単勝人気順位", np.nan)
            win_payout = h_row.get("Target_単勝", np.nan)
            place_payout = h_row.get("Target_複勝", np.nan)
            std_odds = float(h_row.get("KYI_基準オッズ", 0.0))

            slot = h_num - 1
            p_ens = float(probs_ens[slot]) if 0 <= slot < max_horses else 0.0
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
                "ev_std": ev_std
            })

        horses_data_sorted = sorted(horses_data, key=lambda x: x["prob"], reverse=True)
        top1 = horses_data_sorted[0]

        turf_race_results.append({
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
        })

    df_res_turf = pd.DataFrame(turf_race_results)
    print(f"芝推論完了: {len(df_res_turf)} レース")

    # 統合DataFrame
    df_combined = pd.concat([df_res_dirt, df_res_turf], ignore_index=True)
    df_eval = df_combined[df_combined["top1_actual_rank"].notna()].copy()
    print(f"\n全確定レース数: {len(df_eval)} レース (ダート {len(df_res_dirt)}R, 芝 {len(df_res_turf)}R)")

    # -------------------------------------------------------------
    # 日別成績集計
    # -------------------------------------------------------------
    daily_rows = []
    for ymd, g in df_eval.groupby("ymd"):
        # 総合
        n = len(g)
        w_hits = int((g["top1_actual_rank"] == 1).sum())
        w_pay = float(g["top1_win_payout"].fillna(0).sum())
        p_hits = int((g["top1_actual_rank"] <= 3).sum())
        p_pay = float(g["top1_place_payout"].fillna(0).sum())

        w_rate = (w_hits / n) * 100.0 if n > 0 else 0.0
        w_roi = (w_pay / (n * 100.0)) * 100.0 if n > 0 else 0.0
        p_rate = (p_hits / n) * 100.0 if n > 0 else 0.0
        p_roi = (p_pay / (n * 100.0)) * 100.0 if n > 0 else 0.0

        # ダート
        gd = g[g["category"] == "ダート"]
        nd = len(gd)
        wd_hits = int((gd["top1_actual_rank"] == 1).sum()) if nd > 0 else 0
        wd_pay = float(gd["top1_win_payout"].fillna(0).sum()) if nd > 0 else 0.0
        pd_hits = int((gd["top1_actual_rank"] <= 3).sum()) if nd > 0 else 0
        pd_pay = float(gd["top1_place_payout"].fillna(0).sum()) if nd > 0 else 0.0

        wd_rate = (wd_hits / nd) * 100.0 if nd > 0 else np.nan
        wd_roi = (wd_pay / (nd * 100.0)) * 100.0 if nd > 0 else np.nan
        pd_rate = (pd_hits / nd) * 100.0 if nd > 0 else np.nan
        pd_roi = (pd_pay / (nd * 100.0)) * 100.0 if nd > 0 else np.nan

        # 芝
        gt = g[g["category"] == "芝"]
        nt = len(gt)
        wt_hits = int((gt["top1_actual_rank"] == 1).sum()) if nt > 0 else 0
        wt_pay = float(gt["top1_win_payout"].fillna(0).sum()) if nt > 0 else 0.0
        pt_hits = int((gt["top1_actual_rank"] <= 3).sum()) if nt > 0 else 0
        pt_pay = float(gt["top1_place_payout"].fillna(0).sum()) if nt > 0 else 0.0

        wt_rate = (wt_hits / nt) * 100.0 if nt > 0 else np.nan
        wt_roi = (wt_pay / (nt * 100.0)) * 100.0 if nt > 0 else np.nan
        pt_rate = (pt_hits / nt) * 100.0 if nt > 0 else np.nan
        pt_roi = (pt_pay / (nt * 100.0)) * 100.0 if nt > 0 else np.nan

        daily_rows.append({
            "ymd": ymd,
            "total_races": n,
            "total_win_hits": w_hits,
            "total_win_rate": w_rate,
            "total_win_pay": w_pay,
            "total_win_roi": w_roi,
            "total_place_hits": p_hits,
            "total_place_rate": p_rate,
            "total_place_pay": p_pay,
            "total_place_roi": p_roi,

            "dirt_races": nd,
            "dirt_win_hits": wd_hits,
            "dirt_win_rate": wd_rate,
            "dirt_win_pay": wd_pay,
            "dirt_win_roi": wd_roi,
            "dirt_place_hits": pd_hits,
            "dirt_place_rate": pd_rate,
            "dirt_place_pay": pd_pay,
            "dirt_place_roi": pd_roi,

            "turf_races": nt,
            "turf_win_hits": wt_hits,
            "turf_win_rate": wt_rate,
            "turf_win_pay": wt_pay,
            "turf_win_roi": wt_roi,
            "turf_place_hits": pt_hits,
            "turf_place_rate": pt_rate,
            "turf_place_pay": pt_pay,
            "turf_place_roi": pt_roi,
        })

    df_daily = pd.DataFrame(daily_rows)

    # -------------------------------------------------------------
    # 統計量（平均・分散・標準偏差）の算出
    # -------------------------------------------------------------
    def calc_stats_block(series: pd.Series, name: str):
        valid = series.dropna()
        n = len(valid)
        mean_val = float(valid.mean())
        # 標本分散 (ddof=0) と 不偏分散 (ddof=1)
        var_p = float(valid.var(ddof=0))
        var_s = float(valid.var(ddof=1))
        std_p = float(valid.std(ddof=0))
        std_s = float(valid.std(ddof=1))
        min_val = float(valid.min())
        q25 = float(valid.quantile(0.25))
        med_val = float(valid.median())
        q75 = float(valid.quantile(0.75))
        max_val = float(valid.max())

        return {
            "name": name,
            "days": n,
            "mean": mean_val,
            "var_p": var_p,
            "var_s": var_s,
            "std_p": std_p,
            "std_s": std_s,
            "min": min_val,
            "median": med_val,
            "max": max_val,
        }

    metrics = [
        ("総合 日別単勝的中率 (%)", df_daily["total_win_rate"]),
        ("総合 日別単勝回収率 (%)", df_daily["total_win_roi"]),
        ("総合 日別複勝的中率 (%)", df_daily["total_place_rate"]),
        ("総合 日別複勝回収率 (%)", df_daily["total_place_roi"]),

        ("ダート 日別単勝的中率 (%)", df_daily["dirt_win_rate"]),
        ("ダート 日別単勝回収率 (%)", df_daily["dirt_win_roi"]),
        ("ダート 日別複勝的中率 (%)", df_daily["dirt_place_rate"]),
        ("ダート 日別複勝回収率 (%)", df_daily["dirt_place_roi"]),

        ("芝 日別単勝的中率 (%)", df_daily["turf_win_rate"]),
        ("芝 日別単勝回収率 (%)", df_daily["turf_win_roi"]),
        ("芝 日別複勝的中率 (%)", df_daily["turf_place_rate"]),
        ("芝 日別複勝回収率 (%)", df_daily["turf_place_roi"]),
    ]

    stats_list = [calc_stats_block(s, name) for name, s in metrics]
    df_stats = pd.DataFrame(stats_list)

    # コンソール表示
    print("\n" + "=" * 105)
    print("【各日毎の的中率・回収率 統計分析結果 (平均・分散・標準偏差)】")
    print(f"対象期間: 2025/01/05 〜 2026/11/01 (計 {len(df_daily)} 開催日, 全 {len(df_eval)} レース)")
    print("=" * 105)

    print(f"{'指標名':<22} | {'日数':<4} | {'平均(Mean)':<10} | {'不偏分散(s²)':<12} | {'標準偏差(s)':<10} | {'最小':<7} | {'中央値':<7} | {'最大'}")
    print("-" * 105)
    for st in stats_list:
        print(
            f"{st['name']:<22} | {st['days']:3d}日 | {st['mean']:9.2f}% | "
            f"{st['var_s']:11.2f}  | {st['std_s']:9.2f}% | "
            f"{st['min']:6.1f}% | {st['median']:6.1f}% | {st['max']:6.1f}%"
        )

    # 通算回収率（加重平均）との比較
    tot_n = len(df_eval)
    tot_w_pay = df_eval["top1_win_payout"].fillna(0).sum()
    tot_p_pay = df_eval["top1_place_payout"].fillna(0).sum()
    tot_w_hits = (df_eval["top1_actual_rank"] == 1).sum()
    tot_p_hits = (df_eval["top1_actual_rank"] <= 3).sum()

    print("\n" + "=" * 105)
    print("【参考：全レース通算（総投資に対する総回収）】※レース数加重平均")
    print("-" * 105)
    print(f"総合計 ({tot_n}R):")
    print(f"  ■ 単勝: 的中 {tot_w_hits}/{tot_n} ({tot_w_hits/tot_n*100:5.2f}%) | 投資 {tot_n*100:>7}円 | 払戻 {int(tot_w_pay):>7}円 | 通算回収率: {tot_w_pay/(tot_n*100)*100:6.2f}%")
    print(f"  ■ 複勝: 的中 {tot_p_hits}/{tot_n} ({tot_p_hits/tot_n*100:5.2f}%) | 投資 {tot_n*100:>7}円 | 払戻 {int(tot_p_pay):>7}円 | 通算回収率: {tot_p_pay/(tot_n*100)*100:6.2f}%")

    # ダート通算
    nd = len(df_res_dirt)
    wd_pay = df_res_dirt["top1_win_payout"].fillna(0).sum()
    pd_pay = df_res_dirt["top1_place_payout"].fillna(0).sum()
    wd_hits = (df_res_dirt["top1_actual_rank"] == 1).sum()
    pd_hits = (df_res_dirt["top1_actual_rank"] <= 3).sum()
    print(f"\nダート ({nd}R):")
    print(f"  ■ 単勝: 的中 {wd_hits}/{nd} ({wd_hits/nd*100:5.2f}%) | 投資 {nd*100:>7}円 | 払戻 {int(wd_pay):>7}円 | 通算回収率: {wd_pay/(nd*100)*100:6.2f}%")
    print(f"  ■ 複勝: 的中 {pd_hits}/{nd} ({pd_hits/nd*100:5.2f}%) | 投資 {nd*100:>7}円 | 払戻 {int(pd_pay):>7}円 | 通算回収率: {pd_pay/(nd*100)*100:6.2f}%")

    # 芝通算
    nt = len(df_res_turf)
    wt_pay = df_res_turf["top1_win_payout"].fillna(0).sum()
    pt_pay = df_res_turf["top1_place_payout"].fillna(0).sum()
    wt_hits = (df_res_turf["top1_actual_rank"] == 1).sum()
    pt_hits = (df_res_turf["top1_actual_rank"] <= 3).sum()
    print(f"\n芝 ({nt}R):")
    print(f"  ■ 単勝: 的中 {wt_hits}/{nt} ({wt_hits/nt*100:5.2f}%) | 投資 {nt*100:>7}円 | 払戻 {int(wt_pay):>7}円 | 通算回収率: {wt_pay/(nt*100)*100:6.2f}%")
    print(f"  ■ 複勝: 的中 {pt_hits}/{nt} ({pt_hits/nt*100:5.2f}%) | 投資 {nt*100:>7}円 | 払戻 {int(pt_pay):>7}円 | 通算回収率: {pt_pay/(nt*100)*100:6.2f}%")

    # CSV保存
    daily_csv_path = os.path.join(CURRENT_DIR, "daily_stats_2025_2026.csv")
    df_daily.to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n全 {len(df_daily)} 開催日の日別的中率・回収率CSV保存完了: {daily_csv_path}")

    stats_csv_path = os.path.join(CURRENT_DIR, "stats_summary_2025_2026.csv")
    df_stats.to_csv(stats_csv_path, index=False, encoding="utf-8-sig")
    print(f"統計分析サマリーCSV保存完了: {stats_csv_path}")

    races_csv_path = os.path.join(CURRENT_DIR, "race_predictions_2025_2026.csv")
    df_combined.to_csv(races_csv_path, index=False, encoding="utf-8-sig")
    print(f"全 {len(df_combined)} レース詳細予想結果CSV保存完了: {races_csv_path}")

if __name__ == "__main__":
    main()
