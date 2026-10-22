"""
eval_predict_turf_ensemble_20261002.py
========================================================================
2026年10月2日以降の芝1勝クラス以上レースに対する
神アンサンブル [Model 11 + Model 25 + Model 35] 回収率・的中率 検証プログラム
========================================================================
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = "/Users/hino/Program/Python/KeibaData/HorseRaceAI"
sys.path.append(CURRENT_DIR)

from src.utils.helpers import get_device
from src.models.base_nn import RaceLevelStage1NN
from predict_turf_ensemble_m11_m25_m35 import load_turf_models, PLACE_DICT, CLASS_DICT

def evaluate():
    device = get_device()
    print(f"使用デバイス: {device}")

    # 1. 芝正規化統計のロード
    stats_path = os.path.join(CURRENT_DIR, "artifacts_turf_1win/turf_1win_preprocessor_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    all_feature_cols = stats["all_feature_cols"]
    col_to_idx = stats["col_to_idx"]
    bac_feature_cols = stats["bac_feature_cols"]
    medians = stats["medians"]
    means = stats["means"]
    stds = stats["stds"]

    # 2. モデルロード (M11, M25, M35)
    models = load_turf_models(CURRENT_DIR, device)
    f11_cols = list(bac_feature_cols) + models["m11_yearly"]
    f25_cols = list(bac_feature_cols) + models["m25_yearly"]
    f35_cols = list(bac_feature_cols) + models["m35_yearly"]

    f11_indices = [col_to_idx[c] for c in f11_cols]
    f25_indices = [col_to_idx[c] for c in f25_cols]
    f35_indices = [col_to_idx[c] for c in f35_cols]

    # 3. データロード
    target_path = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/2026.csv"
    bac_path = "/Users/hino/Program/Python/KeibaData/Data/RESULT/MAIN/BAC_KAB.csv"

    print(f"Loading {target_path}...")
    df_target = pd.read_csv(target_path, low_memory=False)

    print(f"Loading {bac_path}...")
    df_bac = pd.read_csv(bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)

    # RACE_ID 付与
    target_rkeys = df_target["KYI_RACE_KEY"].astype(str).str.zfill(10)
    df_target["RACE_ID"] = target_rkeys.str[:8]
    df_target["KYI_RACE_KEY"] = target_rkeys

    # BAC とマージ
    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_target.columns]
    merged = pd.merge(df_target, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")

    # 4. フィルタリング: 2026年10月2日以降 & 芝1勝クラス以上
    merged["年月日_int"] = pd.to_numeric(merged["年月日"], errors="coerce").fillna(0).astype(int)
    date_mask = merged["年月日_int"] >= 20261002
    turf_1win_mask = (merged["芝ダ障害コード"] == 1) & (merged["KYI_条件クラス"].isin([1, 2, 3, 9]))
    
    merged_sub = merged[date_mask & turf_1win_mask].copy().reset_index(drop=True)
    all_target_races = merged_sub["RACE_ID"].drop_duplicates().tolist()
    print(f"\n【フィルタ適用】2026/10/02以降の芝1勝クラス以上レース数: {len(all_target_races)} レース (計 {len(merged_sub)} 頭)")

    # 5. 特徴量抽出 & 正規化
    mat = np.zeros((len(merged_sub), len(all_feature_cols)), dtype=np.float32)
    for i, c in enumerate(all_feature_cols):
        if c in merged_sub.columns:
            mat[:, i] = pd.to_numeric(merged_sub[c], errors="coerce").fillna(np.nan).values
        else:
            mat[:, i] = np.nan

    for col_i in range(len(all_feature_cols)):
        nan_mask = np.isnan(mat[:, col_i])
        if np.any(nan_mask):
            mat[nan_mask, col_i] = medians[col_i]

    mat = np.clip((mat - means) / stds, -10.0, 10.0)
    mat = np.nan_to_num(mat, nan=0.0)

    # 6. 推論ループ
    max_horses = 18
    race_results = []
    horse_predictions = []

    for rid in all_target_races:
        race_mask_rows = (merged_sub["RACE_ID"] == rid)
        race_df = merged_sub[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        ymd = int(r_row.get("年月日_int", 0))
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
        place_name = PLACE_DICT.get(place_code, f"場{place_code}")
        race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
        distance = int(r_row.get("距離", 0))
        class_code = int(r_row.get("KYI_条件クラス", 0))
        class_str = CLASS_DICT.get(class_code, f"クラス{class_code}")

        # テンソル構築
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
            logits11 = models["net11"](X11_flat, mask=mask_t)
            logits25 = models["net25"](X25_flat, mask=mask_t)
            logits35 = models["net35"](X35_flat, mask=mask_t)

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
            
            # 着順・払戻・オッズ
            actual_rank = h_row.get("Target_着順", np.nan)
            final_odds = h_row.get("Target_確定単勝オッズ", np.nan)
            final_pop = h_row.get("Target_確定単勝人気順位", np.nan)
            win_payout = h_row.get("Target_単勝", np.nan)
            place_payout = h_row.get("Target_複勝", np.nan)

            std_odds = float(h_row.get("KYI_基準オッズ", 0.0))
            std_pop = int(h_row.get("KYI_基準人気順位", 0)) if not pd.isna(h_row.get("KYI_基準人気順位", 0)) else 0

            slot = h_num - 1
            p_ens = float(probs_ens[slot]) if 0 <= slot < max_horses else 0.0
            p_m11 = float(probs11[slot]) if 0 <= slot < max_horses else 0.0
            p_m25 = float(probs25[slot]) if 0 <= slot < max_horses else 0.0
            p_m35 = float(probs35[slot]) if 0 <= slot < max_horses else 0.0

            ev_std = p_ens * std_odds if std_odds > 0 else 0.0
            ev_final = p_ens * float(final_odds) if pd.notna(final_odds) and float(final_odds) > 0 else 0.0

            h_dict = {
                "race_id": rid,
                "ymd": ymd,
                "place": place_name,
                "race_num": race_num,
                "distance": distance,
                "class_name": class_str,
                "horse_num": h_num,
                "horse_name": h_name,
                "jockey": jockey,
                "actual_rank": actual_rank,
                "final_odds": final_odds,
                "final_pop": final_pop,
                "win_payout": win_payout,
                "place_payout": place_payout,
                "std_odds": std_odds,
                "std_pop": std_pop,
                "prob_ens": p_ens,
                "prob_m11": p_m11,
                "prob_m25": p_m25,
                "prob_m35": p_m35,
                "ev_std": ev_std,
                "ev_final": ev_final
            }
            horses_data.append(h_dict)
            horse_predictions.append(h_dict)

        # アンサンブル勝率順
        horses_data.sort(key=lambda x: x["prob_ens"], reverse=True)
        top1 = horses_data[0]
        
        # M11 1位, M25 1位, M35 1位
        top_m11 = max(horses_data, key=lambda x: x["prob_m11"])
        top_m25 = max(horses_data, key=lambda x: x["prob_m25"])
        top_m35 = max(horses_data, key=lambda x: x["prob_m35"])

        # 推奨度判定
        if top1["prob_ens"] >= 0.25 and top1["ev_std"] >= 1.2:
            recom = "★★★ 激熱鉄板 (勝率25%+ & EV1.2+)"
        elif top1["prob_ens"] >= 0.25:
            recom = "★★☆ 有力本命 (勝率25%+)"
        elif top1["ev_std"] >= 2.0:
            recom = "★★★ 爆穴妙味 (EV2.0+)"
        elif top1["ev_std"] >= 1.2:
            recom = "★★☆ 妙味中穴 (EV1.2+)"
        else:
            recom = "★☆☆ 通常"

        race_results.append({
            "race_id": rid,
            "ymd": ymd,
            "place": place_name,
            "race_num": race_num,
            "distance": distance,
            "class_name": class_str,
            "top1_horse_num": top1["horse_num"],
            "top1_horse_name": top1["horse_name"],
            "top1_prob": top1["prob_ens"],
            "top1_prob11": top1["prob_m11"],
            "top1_prob25": top1["prob_m25"],
            "top1_prob35": top1["prob_m35"],
            "top1_std_odds": top1["std_odds"],
            "top1_ev_std": top1["ev_std"],
            "top1_actual_rank": top1["actual_rank"],
            "top1_final_odds": top1["final_odds"],
            "top1_final_pop": top1["final_pop"],
            "top1_win_payout": top1["win_payout"],
            "top1_place_payout": top1["place_payout"],
            "recom": recom,
            "top_m11_num": top_m11["horse_num"],
            "top_m11_rank": top_m11["actual_rank"],
            "top_m11_win_payout": top_m11["win_payout"],
            "top_m25_num": top_m25["horse_num"],
            "top_m25_rank": top_m25["actual_rank"],
            "top_m25_win_payout": top_m25["win_payout"],
            "top_m35_num": top_m35["horse_num"],
            "top_m35_rank": top_m35["actual_rank"],
            "top_m35_win_payout": top_m35["win_payout"]
        })

    df_eval = pd.DataFrame(race_results)
    n_races = len(df_eval)

    # 全体成績 (神アンサンブル 本命◎)
    win_hits = (df_eval["top1_actual_rank"] == 1).sum()
    win_hit_rate = win_hits / n_races if n_races > 0 else 0.0
    total_invest = n_races * 100
    total_win_payout = df_eval["top1_win_payout"].fillna(0.0).sum()
    win_roi = (total_win_payout / total_invest) * 100.0 if total_invest > 0 else 0.0

    # 複勝
    place_hits = (df_eval["top1_actual_rank"] <= 3).sum()
    place_hit_rate = place_hits / n_races if n_races > 0 else 0.0
    total_place_payout = df_eval["top1_place_payout"].fillna(0.0).sum()
    place_roi = (total_place_payout / total_invest) * 100.0 if total_invest > 0 else 0.0

    print("=" * 85)
    print("【2026年10月2日以降 芝1勝クラス以上 神アンサンブル (M11+M25+M35) 成績サマリー】")
    print(f"対象確定レース数: {n_races} レース")
    print(f"■ 単勝成績: 的中 {win_hits}/{n_races} ({win_hit_rate*100:.2f}%) | 投資 {total_invest}円 | 払戻 {int(total_win_payout)}円 | 回収率: {win_roi:.2f}%")
    print(f"■ 複勝成績: 的中 {place_hits}/{n_races} ({place_hit_rate*100:.2f}%) | 投資 {total_invest}円 | 払戻 {int(total_place_payout)}円 | 回収率: {place_roi:.2f}%")
    print("=" * 85)

    # 日別成績
    print("\n【日別 成績】")
    for ymd, g in df_eval.groupby("ymd"):
        g_n = len(g)
        g_win_hits = (g["top1_actual_rank"] == 1).sum()
        g_win_pay = g["top1_win_payout"].fillna(0.0).sum()
        g_win_roi = (g_win_pay / (g_n * 100)) * 100.0
        g_pl_hits = (g["top1_actual_rank"] <= 3).sum()
        g_pl_pay = g["top1_place_payout"].fillna(0.0).sum()
        g_pl_roi = (g_pl_pay / (g_n * 100)) * 100.0
        print(f"日付: {ymd} ({g_n}R) | 単勝的中: {g_win_hits}/{g_n} ({g_win_hits/g_n*100:5.1f}%) 回収率: {g_win_roi:6.1f}% | 複勝的中: {g_pl_hits}/{g_n} ({g_pl_hits/g_n*100:5.1f}%) 回収率: {g_pl_roi:6.1f}%")

    # モデル比較
    m11_win_hits = (df_eval["top_m11_rank"] == 1).sum()
    m11_win_pay = df_eval["top_m11_win_payout"].fillna(0.0).sum()
    m11_win_roi = (m11_win_pay / total_invest) * 100.0

    m25_win_hits = (df_eval["top_m25_rank"] == 1).sum()
    m25_win_pay = df_eval["top_m25_win_payout"].fillna(0.0).sum()
    m25_win_roi = (m25_win_pay / total_invest) * 100.0

    m35_win_hits = (df_eval["top_m35_rank"] == 1).sum()
    m35_win_pay = df_eval["top_m35_win_payout"].fillna(0.0).sum()
    m35_win_roi = (m35_win_pay / total_invest) * 100.0

    print("\n【モデル比較 (単勝)】")
    print(f"・Model 11 単体 : 的中 {m11_win_hits}/{n_races} ({m11_win_hits/n_races*100:.1f}%) | 払戻 {int(m11_win_pay)}円 | 回収率: {m11_win_roi:.2f}%")
    print(f"・Model 25 単体 : 的中 {m25_win_hits}/{n_races} ({m25_win_hits/n_races*100:.1f}%) | 払戻 {int(m25_win_pay)}円 | 回収率: {m25_win_roi:.2f}%")
    print(f"・Model 35 単体 : 的中 {m35_win_hits}/{n_races} ({m35_win_hits/n_races*100:.1f}%) | 払戻 {int(m35_win_pay)}円 | 回収率: {m35_win_roi:.2f}%")
    print(f"・神アンサンブル : 的中 {win_hits}/{n_races} ({win_hit_rate*100:.1f}%) | 払戻 {int(total_win_payout)}円 | 回収率: {win_roi:.2f}%")

    # 推奨度別
    print("\n【推奨度別 成績 (単勝)】")
    for recom, g in df_eval.groupby("recom"):
        g_n = len(g)
        g_hits = (g["top1_actual_rank"] == 1).sum()
        g_pay = g["top1_win_payout"].fillna(0.0).sum()
        g_roi = (g_pay / (g_n * 100)) * 100.0
        print(f"・{recom:<30}: {g_hits}/{g_n} ({g_hits/g_n*100:5.1f}%) | 払戻 {int(g_pay):>5}円 | 回収率: {g_roi:6.1f}%")

    # EV閾値別
    print("\n【期待値(EV_std) フィルタ別 成績 (単勝)】")
    for ev_th in [0.0, 0.8, 1.0, 1.2, 1.5, 2.0]:
        g = df_eval[df_eval["top1_ev_std"] >= ev_th]
        g_n = len(g)
        if g_n > 0:
            g_hits = (g["top1_actual_rank"] == 1).sum()
            g_pay = g["top1_win_payout"].fillna(0.0).sum()
            g_roi = (g_pay / (g_n * 100)) * 100.0
            print(f"・EV >= {ev_th:.1f}: {g_hits}/{g_n} ({g_hits/g_n*100:5.1f}%) | 払戻 {int(g_pay):>5}円 | 回収率: {g_roi:6.1f}%")

    # 勝率閾値別
    print("\n【予測勝率(prob_ens) フィルタ別 成績 (単勝)】")
    for p_th in [0.15, 0.20, 0.25, 0.30]:
        g = df_eval[df_eval["top1_prob"] >= p_th]
        g_n = len(g)
        if g_n > 0:
            g_hits = (g["top1_actual_rank"] == 1).sum()
            g_pay = g["top1_win_payout"].fillna(0.0).sum()
            g_roi = (g_pay / (g_n * 100)) * 100.0
            print(f"・勝率 >= {p_th*100:.0f}%: {g_hits}/{g_n} ({g_hits/g_n*100:5.1f}%) | 払戻 {int(g_pay):>5}円 | 回収率: {g_roi:6.1f}%")

    # 全31レース詳細テーブル出力
    print("\n【確定済み 31レース詳細結果】")
    for _, r in df_eval.iterrows():
        hit_mark = "★的中" if r["top1_actual_rank"] == 1 else ("◯複勝" if r["top1_actual_rank"] <= 3 else "×ハズレ")
        pay_str = f"単{int(r['top1_win_payout'])}円" if r["top1_actual_rank"] == 1 else ""
        print(f"{r['ymd']} {r['place']}{r['race_num']:2d}R 芝{r['distance']}m | 本命: {r['top1_horse_num']:2d}番 {r['top1_horse_name'][:8]} (勝率{r['top1_prob']*100:4.1f}%, 基準オッズ{r['top1_std_odds']:4.1f}, EV{r['top1_ev_std']:4.2f}) | 確定{int(r['top1_final_pop'])}人気 {r['top1_final_odds']:4.1f}倍 | 結果: {int(r['top1_actual_rank']):2d}着 {hit_mark} {pay_str}")

    # CSV保存
    out_csv = "/Users/hino/Program/Python/KeibaData/HorseRaceAI/eval_turf_ensemble_m11_m25_m35_results_20261002.csv"
    df_eval.to_csv(out_csv, index=False)
    print(f"\n結果CSV保存完了: {out_csv}")

if __name__ == "__main__":
    evaluate()
