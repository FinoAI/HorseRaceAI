"""
predict_best_cascade.py
========================================================================
BESTチェーン式モデル (Cascade Chain: Step 1 -> Step 2) 予想プログラム
------------------------------------------------------------------------
- 対象レース: JRA ダート 1勝クラス以上 (1勝/2勝/3勝/オープン/重賞) のみ
- 推論方式: カスケードチェーン式 (Cascade Chain)
  * Step 1: 起点モデル (200特徴量) で一次予測
  * Step 2: 次段モデル (200特徴量 + Step1予測値 = 201特徴量) で最終予測
- デフォルト最強チェーン: M2 -> M27 (テスト回収率 115.77% 歴代最高)
  (オプションで M49->M47, M2->M29 なども指定可能)
- 入力データ:
  * target_v2.csv (出走馬情報)
  * BAC_KAB.csv (レース情報)
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

# パス追加
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

# 利用可能な有力チェーンモデルの定義
AVAILABLE_CHAINS = {
    "M24->M49": {
        "name": "M24 -> M49 (BESTモデル M24起点 -> M49次段 カスケードチェーン)",
        "m1": 24,
        "m2": 49,
        "m1_weights": "artifacts_dirt_1win/model_24.pth",
        "m2_weights": "artifacts_dirt_1win/cascade_M24_to_M49.pth",
        "m1_features": "artifacts_dirt_1win/features_24.json",
        "m2_features": "artifacts_dirt_1win/features_49.json",
        "test_roi": 0.7972,
        "val_roi": 0.8003,
        "hit_rate": 0.2250,
        "mean_odds": 3.54
    },
    "M49->M24": {
        "name": "M49 -> M24 (BESTモデル M49起点 -> M24次段 カスケードチェーン)",
        "m1": 49,
        "m2": 24,
        "m1_weights": "artifacts_dirt_1win/model_49.pth",
        "m2_weights": "artifacts_dirt_1win/cascade_M49_to_M24.pth",
        "m1_features": "artifacts_dirt_1win/features_49.json",
        "m2_features": "artifacts_dirt_1win/features_24.json",
        "test_roi": 0.8507,
        "val_roi": 0.7336,
        "hit_rate": 0.2304,
        "mean_odds": 3.69
    },
    "M2->M27": {
        "name": "M2 -> M27 (最高テスト回収率 115.77%)",
        "m1": 2,
        "m2": 27,
        "m1_weights": "artifacts_dirt_1win/model_2.pth",
        "m2_weights": "artifacts_cascade_4origins_exhaustive/test100_M2_to_M27.pth",
        "m1_features": "artifacts_dirt_1win/features_2.json",
        "m2_features": "artifacts_dirt_1win/features_27.json",
        "test_roi": 1.1577,
        "val_roi": 0.8136,
        "hit_rate": 0.1997,
        "mean_odds": 5.80
    },
    "M49->M47": {
        "name": "M49 -> M47 (両方向上・テスト黒字 106.43%)",
        "m1": 49,
        "m2": 47,
        "m1_weights": "artifacts_dirt_1win/model_49.pth",
        "m2_weights": "artifacts_cascade_continue_2086/found_M49_to_M47.pth",
        "m1_features": "artifacts_dirt_1win/features_49.json",
        "m2_features": "artifacts_dirt_1win/features_47.json",
        "test_roi": 1.0643,
        "val_roi": 0.8771,
        "hit_rate": 0.1984,
        "mean_odds": 5.36
    },
    "M2->M29": {
        "name": "M2 -> M29 (テスト回収率 104.25% / 高的中率 24.23%)",
        "m1": 2,
        "m2": 29,
        "m1_weights": "artifacts_dirt_1win/model_2.pth",
        "m2_weights": "artifacts_cascade_4origins_exhaustive/test100_M2_to_M29.pth",
        "m1_features": "artifacts_dirt_1win/features_2.json",
        "m2_features": "artifacts_dirt_1win/features_29.json",
        "test_roi": 1.0425,
        "val_roi": 0.7654,
        "hit_rate": 0.2423,
        "mean_odds": 4.30
    }
}

def load_chain_models(chain_key: str, base_dir: str, device: torch.device):
    """指定されたチェーンモデル (Step 1 & Step 2) をロード"""
    cfg = AVAILABLE_CHAINS[chain_key]
    max_horses = 18

    # 特徴量定義ロード
    with open(os.path.join(base_dir, cfg["m1_features"]), "r", encoding="utf-8") as f:
        f1_data = json.load(f)
    with open(os.path.join(base_dir, cfg["m2_features"]), "r", encoding="utf-8") as f:
        f2_data = json.load(f)

    # Step 1 モデル (200 特徴量 * 18 頭 = 3,600)
    net1 = RaceLevelStage1NN(input_dim=max_horses * 200, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w1_path = os.path.join(base_dir, cfg["m1_weights"])
    net1.load_state_dict(torch.load(w1_path, map_location=device))
    net1.eval()

    # Step 2 モデル (201 特徴量 * 18 頭 = 3,618)
    net2 = RaceLevelStage1NN(input_dim=max_horses * 201, max_horses=max_horses, multiplier=2, reduction_ratio=0.5, dropout=0.15).to(device)
    w2_path = os.path.join(base_dir, cfg["m2_weights"])
    net2.load_state_dict(torch.load(w2_path, map_location=device))
    net2.eval()

    return {
        "cfg": cfg,
        "net1": net1,
        "net2": net2,
        "m1_yearly": f1_data["yearly_features"],
        "m2_yearly": f2_data["yearly_features"]
    }

def main():
    parser = argparse.ArgumentParser(description="JRA BEST Cascade Chain Prediction (Dirt 1-Win+ Only)")
    parser.add_argument("--chain", type=str, default="M24->M49", choices=list(AVAILABLE_CHAINS.keys()), help="Chain model key")
    parser.add_argument("--target_path", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/target_v2.csv", help="Path to target_v2.csv")
    parser.add_argument("--bac_path", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/BAC_KAB.csv", help="Path to BAC_KAB.csv")
    parser.add_argument("--output_dir", type=str, default="/Users/hino/Program/Python/KeibaData/Data/PREDICT/", help="Output directory")
    args = parser.parse_args()

    print("=" * 85)
    print("  ★ JRA ダート1勝クラス以上専用 BESTチェーン式予想プログラム ★")
    print(f"  使用チェーンモデル: {AVAILABLE_CHAINS[args.chain]['name']}")
    print("=" * 85)

    device = get_device()
    print(f"使用デバイス: {device}")

    # 1. 前処理正規化統計量のロード
    stats_path = os.path.join(CURRENT_DIR, "artifacts_dirt_1win/dirt_1win_preprocessor_stats.pkl")
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} が見つかりません。")
        sys.exit(1)

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    all_feature_cols = stats["all_feature_cols"]
    col_to_idx = stats["col_to_idx"]
    bac_feature_cols = stats["bac_feature_cols"]
    medians = stats["medians"]
    means = stats["means"]
    stds = stats["stds"]

    # 2. チェーンモデルのロード
    chain_model = load_chain_models(args.chain, CURRENT_DIR, device)
    cfg = chain_model["cfg"]

    # 特徴量列の特定 (BAC 36 + 各モデル 164 = 200)
    m1_cols = list(bac_feature_cols) + chain_model["m1_yearly"]
    m2_cols = list(bac_feature_cols) + chain_model["m2_yearly"]
    m1_indices = [col_to_idx[c] for c in m1_cols]
    m2_indices = [col_to_idx[c] for c in m2_cols]

    # 3. 入力データロード
    df_target = pd.read_csv(args.target_path, low_memory=False)
    df_bac = pd.read_csv(args.bac_path, dtype={"RACE_ID": str}, low_memory=False)
    df_bac["RACE_ID"] = df_bac["RACE_ID"].astype(str).str.zfill(8)
    df_bac = df_bac.drop_duplicates(subset=["RACE_ID"]).reset_index(drop=True)

    target_rkeys = df_target["KYI_RACE_KEY"].astype(str).str.zfill(10)
    df_target["RACE_ID"] = target_rkeys.str[:8]
    df_target["KYI_RACE_KEY"] = target_rkeys

    bac_cols_to_merge = [c for c in df_bac.columns if c == "RACE_ID" or c not in df_target.columns]
    merged = pd.merge(df_target, df_bac[bac_cols_to_merge], on="RACE_ID", how="left")

    # 4. 【絶対ルール】ダート 1勝クラス以上のレースのみに厳密フィルタ
    # 芝ダ障害コード == 2 (ダート) かつ KYI_条件クラス in [1, 2, 3, 9]
    dirt_1win_mask = (merged["芝ダ障害コード"] == 2) & (merged["KYI_条件クラス"].isin([1, 2, 3, 9]))
    merged_dirt = merged[dirt_1win_mask].copy().reset_index(drop=True)

    target_races = merged_dirt["RACE_ID"].drop_duplicates().tolist()
    print(f"\n【フィルタ適用】全24レース中、対象の『ダート1勝クラス以上』レース数: {len(target_races)} レース (計 {len(merged_dirt)} 頭)")

    if len(target_races) == 0:
        print("対象となるダート1勝クラス以上のレースが存在しませんでした。")
        return

    # 5. 特徴量行列化 & 正規化
    mat = np.zeros((len(merged_dirt), len(all_feature_cols)), dtype=np.float32)
    for i, c in enumerate(all_feature_cols):
        if c in merged_dirt.columns:
            mat[:, i] = pd.to_numeric(merged_dirt[c], errors="coerce").fillna(np.nan).values
        else:
            mat[:, i] = np.nan

    # median 補完
    for col_i in range(len(all_feature_cols)):
        nan_mask = np.isnan(mat[:, col_i])
        if np.any(nan_mask):
            mat[nan_mask, col_i] = medians[col_i]

    # 標準化
    mat = np.clip((mat - means) / stds, -10.0, 10.0)
    mat = np.nan_to_num(mat, nan=0.0)

    # 6. レースごとのカスケード推論
    max_horses = 18
    all_horses_results = []
    race_summaries = []

    for rid in target_races:
        race_mask_rows = (merged_dirt["RACE_ID"] == rid)
        race_df = merged_dirt[race_mask_rows].copy().reset_index(drop=True)
        race_mat = mat[race_mask_rows]
        n_horses = len(race_df)

        r_row = race_df.iloc[0]
        place_code = str(r_row.get("KYI_場コード", str(r_row.get("場コード", "")))).zfill(2)
        place_name = PLACE_DICT.get(place_code, f"場{place_code}")
        race_num = int(r_row.get("KYI_R", r_row.get("R", 0)))
        distance = int(r_row.get("距離", 0))
        class_code = int(r_row.get("KYI_条件クラス", 0))
        class_str = CLASS_DICT.get(class_code, f"クラス{class_code}")

        # === Step 1: 起点モデル (M1) による一次推論 ===
        X1 = np.zeros((1, max_horses, 200), dtype=np.float32)
        mask = np.zeros((1, max_horses), dtype=np.float32)

        for h_i in range(min(n_horses, max_horses)):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            slot = h_num - 1
            if 0 <= slot < max_horses:
                X1[0, slot] = race_mat[h_i, m1_indices]
                mask[0, slot] = 1.0

        X1_flat = torch.tensor(X1.reshape(1, -1), dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits1 = chain_model["net1"](X1_flat, mask=mask_t)
            probs1 = torch.softmax(logits1, dim=-1).cpu().numpy()[0]  # shape: (18,)

        # === Step 2: 次段モデル (M2) によるカスケード推論 ===
        # M2 の 200 特徴量に、Step 1 の予測勝率 probs1 を 201 番目の特徴量として結合
        X2 = np.zeros((1, max_horses, 201), dtype=np.float32)
        for h_i in range(min(n_horses, max_horses)):
            h_row = race_df.iloc[h_i]
            h_num = int(h_row.get("KYI_馬番", h_i + 1))
            slot = h_num - 1
            if 0 <= slot < max_horses:
                X2[0, slot, :200] = race_mat[h_i, m2_indices]
                X2[0, slot, 200] = probs1[slot]  # カスケード特徴量！

        X2_flat = torch.tensor(X2.reshape(1, -1), dtype=torch.float32, device=device)

        with torch.no_grad():
            logits2 = chain_model["net2"](X2_flat, mask=mask_t)
            probs2 = torch.softmax(logits2, dim=-1).cpu().numpy()[0]  # 最終予測勝率！

        # 各馬のデータ整理
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
            p_s1 = float(probs1[slot]) if 0 <= slot < max_horses else 0.0
            p_s2 = float(probs2[slot]) if 0 <= slot < max_horses else 0.0
            delta_p = p_s2 - p_s1
            ev = p_s2 * std_odds if std_odds > 0 else 0.0

            horses_data.append({
                "race_id": rid,
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
                "step1_prob": p_s1,
                "final_prob": p_s2,
                "delta_prob": delta_p,
                "ev": ev
            })

        # 最終勝率 (Step 2) 順にソート
        horses_data.sort(key=lambda x: x["final_prob"], reverse=True)
        marks = ["◎", "◯", "▲", "△", "△", "★"]
        for rank_idx, h in enumerate(horses_data):
            h["pred_rank"] = rank_idx + 1
            h["mark"] = marks[rank_idx] if rank_idx < len(marks) else "  "
            all_horses_results.append(h)

        top1 = horses_data[0]
        top2 = horses_data[1] if len(horses_data) > 1 else None
        gap = top1["final_prob"] - top2["final_prob"] if top2 else top1["final_prob"]

        race_summaries.append({
            "race_id": rid,
            "place": place_name,
            "race_num": race_num,
            "distance": distance,
            "class_name": class_str,
            "top1_num": top1["horse_num"],
            "top1_name": top1["horse_name"],
            "top1_step1_prob": top1["step1_prob"],
            "top1_final_prob": top1["final_prob"],
            "top1_delta_prob": top1["delta_prob"],
            "top1_odds": top1["std_odds"],
            "top1_ev": top1["ev"],
            "gap": gap,
            "horses": horses_data
        })

    # 7. コンソール表示 (ダート1勝クラス以上のみ！)
    print("\n" + "=" * 105)
    print(f"  ★ JRA ダート1勝クラス以上専用 レース別 予想サマリー (チェーン: {args.chain}) ★")
    print("=" * 105)
    print(f"{'場・R':<8} | {'距離・クラス':<20} | {'本命◎ (馬番・馬名)':<24} | {'Step1勝率':<9} | {'最終勝率':<9} | {'(変化量)':<8} | {'基準オッズ':<8} | {'期待値(EV)':<9} | {'推奨度'}")
    print("-" * 105)

    for r in race_summaries:
        r_str = f"{r['place']}{r['race_num']:2d}R"
        c_str = f"ダ{r['distance']}m ({r['class_name'][:6]})"
        top_str = f"◎ {r['top1_num']:2d} {r['top1_name'][:9]}"
        s1_str = f"{r['top1_step1_prob']*100:5.1f}%"
        s2_str = f"{r['top1_final_prob']*100:5.1f}%"
        d_str = f"{r['top1_delta_prob']*100:+5.1f}pt"
        odds_str = f"{r['top1_odds']:4.1f}倍"
        ev_str = f"{r['top1_ev']:4.2f}"

        if r["top1_final_prob"] >= 0.25 and r["top1_ev"] >= 1.2:
            recom = "★★★ 激熱鉄板 (勝率25%+ & EV1.2+)"
        elif r["top1_final_prob"] >= 0.25:
            recom = "★★☆ 有力本命 (勝率25%+)"
        elif r["top1_ev"] >= 2.0:
            recom = "★★★ 爆穴妙味 (EV2.0+ 超高期待値)"
        elif r["top1_ev"] >= 1.2:
            recom = "★★☆ 妙味中穴 (EV1.2+)"
        else:
            recom = "★☆☆ 通常"

        print(f"{r_str:<8} | {c_str:<20} | {top_str:<24} | {s1_str:<9} | {s2_str:<9} | {d_str:<8} | {odds_str:<8} | {ev_str:<9} | {recom}")

    # 8. ファイル保存
    date_tag = datetime.now().strftime("%Y%m%d")

    # CSV 出力
    df_out = pd.DataFrame(all_horses_results)
    out_csv_path = os.path.join(args.output_dir, f"predict_cascade_{args.chain.replace('->', '_')}_{date_tag}.csv")
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    # Markdown 出力
    out_md_path = os.path.join(args.output_dir, f"predict_cascade_{args.chain.replace('->', '_')}_{date_tag}.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(f"# JRA ダート1勝クラス以上専用 BESTチェーン式予想レポート\n\n")
        f.write(f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 使用チェーン: **{cfg['name']}** (テスト回収率 {cfg['test_roi']*100:.2f}%, 検証回収率 {cfg['val_roi']*100:.2f}%)\n")
        f.write(f"- 推論構造: **Step 1 (M{cfg['m1']}: 200特徴量) $\\to$ Step 2 (M{cfg['m2']}: 201特徴量)** カスケード接続\n")
        f.write(f"- 対象条件: **ダート 1勝クラス以上 (1勝/2勝/3勝/オープン/重賞) 限定** (計 {len(target_races)} レース)\n\n")

        f.write(f"## 1. 対象レース 予想サマリー\n\n")
        f.write(f"| レース | 距離・クラス | 本命馬 (馬番・馬名) | Step1勝率 (M{cfg['m1']}) | 最終勝率 (M{cfg['m2']}) | チェーン効果 | 基準オッズ | 期待値 (EV) | 推奨度 |\n")
        f.write(f"| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |\n")

        for r in race_summaries:
            if r["top1_final_prob"] >= 0.25 and r["top1_ev"] >= 1.2:
                recom = "★★★ 激熱鉄板"
            elif r["top1_final_prob"] >= 0.25:
                recom = "★★☆ 有力本命"
            elif r["top1_ev"] >= 2.0:
                recom = "★★★ 爆穴妙味"
            elif r["top1_ev"] >= 1.2:
                recom = "★★☆ 妙味中穴"
            else:
                recom = "★☆☆ 通常"

            f.write(f"| **{r['place']}{r['race_num']}R** | ダ{r['distance']}m ({r['class_name']}) | **◎ {r['top1_num']} {r['top1_name']}** | {r['top1_step1_prob']*100:.1f}% | **{r['top1_final_prob']*100:.1f}%** | **{r['top1_delta_prob']*100:+5.1f}pt** | {r['top1_odds']:.1f}倍 | **{r['top1_ev']:.2f}** | {recom} |\n")

        f.write(f"\n## 2. 各レース全頭 詳細予想表\n\n")
        for r in race_summaries:
            f.write(f"### {r['place']} {r['race_num']}R: ダート{r['distance']}m {r['class_name']}\n\n")
            f.write(f"| 印 | 順位 | 馬番 | 馬名 | 斤量 | 騎手 | Step1勝率 | 最終勝率 (チェーン後) | 勝率変化 | 基準オッズ | 期待値 (EV) |\n")
            f.write(f"| :---: | :---: | :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for h in r["horses"]:
                f.write(f"| {h['mark']} | {h['pred_rank']} | {h['horse_num']} | {h['horse_name']} | {h['weight']:.1f} | {h['jockey']} | {h['step1_prob']*100:.1f}% | **{h['final_prob']*100:.1f}%** | {h['delta_prob']*100:+5.1f}pt | {h['std_odds']:.1f}倍 | **{h['ev']:.2f}** |\n")
            f.write(f"\n")

    print("\n" + "=" * 85)
    print(f"  ダート1勝クラス以上限定のチェーン式予想が正常に完了しました！")
    print(f"  Markdown レポート: {out_md_path}")
    print(f"  CSV 出力ファイル : {out_csv_path}")
    print("=" * 85)

if __name__ == "__main__":
    main()
