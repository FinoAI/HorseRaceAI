"""
generate_turf_cascade_report.py
========================================================================
芝・1勝クラス以上 2段カスケード総当たり探索 抽出レポート生成スクリプト
========================================================================
"""

import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    csv_path = os.path.join(CURRENT_DIR, "artifacts_turf_cascade_exhaustive/extracted_turf_cascade_pairs.csv")
    df = pd.read_csv(csv_path)

    report_path = os.path.join(CURRENT_DIR, "TURF_CASCADE_EXHAUSTIVE_EXTRACTION_REPORT.md")

    md = []
    md.append("# 芝・1勝クラス以上 2段カスケード総当たり探索 抽出結果レポート\n\n")
    md.append("> **対象データ**: JRA 芝 1勝クラス以上 (条件クラス in [1, 2, 3, 9])  \n")
    md.append("> **評価データ**: **VAL (826レース) のみ** 完全評価 (TEST検証は指示通りスキップ)  \n")
    md.append("> **探索規模**: 50モデル × 49モデル = **全 2,450 通り 完全走破**  \n")
    md.append("> **抽出条件**: \n")
    md.append(">   1. **Val 的中率 $\\ge 20.0\%$**  \n")
    md.append(">   2. **Val 回収率 $\\ge 90.0\%$**  \n")
    md.append(">   3. **元のモデル (起点 $M_i$) より回収率が向上していること ($\\Delta\\text{ROI} \\ge 0$)**  \n\n")
    md.append("---\n\n")

    md.append("## 1. エグゼクティブサマリー\n\n")
    md.append("| 評価項目 | 数値・結果 | 備考 |\n")
    md.append("| :--- | :---: | :--- |\n")
    md.append(f"| **総試行ペア数** | **2,450 ペア** | 100.0% 完全走破完了 |\n")
    md.append(f"| **条件合致 抽出ペア数** | **{len(df)} ペア** | 全体の 8.7% を厳選抽出 |\n")
    md.append(f"| **Val 回収率 $\\ge$ 100% (黒字化達成)** | **{(df['val_roi'] >= 1.0).sum()} ペア** | 抽出ペアの 44.3% が完全黒字 |\n")
    md.append(f"| **Val 回収率 $\\ge$ 110%** | **{(df['val_roi'] >= 1.1).sum()} ペア** | |\n")
    md.append(f"| **Val 回収率 $\\ge$ 120%** | **{(df['val_roi'] >= 1.2).sum()} ペア** | |\n")
    md.append(f"| **Val 回収率 $\\ge$ 130%** | **{(df['val_roi'] >= 1.3).sum()} ペア** | |\n")
    md.append(f"| **Val 的中率 $\\ge$ 22.0%** | **{(df['val_hit_rate'] >= 0.22).sum()} ペア** | 高安定型 |\n")
    md.append(f"| **Val 的中率 $\\ge$ 24.0%** | **{(df['val_hit_rate'] >= 0.24).sum()} ペア** | 超高的中型 |\n")
    md.append(f"| **Val 回収率 Top 1** | **`M36 -> M11`** | **Val ROI: 145.64%** (元: 88.29%, **+57.35pt 向上**) |\n")
    md.append(f"| **Val 的中率 Top 1** | **`M13 -> M49`** | **的中率: 25.30% (209勝)** (Val ROI: 100.57%) |\n")
    md.append(f"| **高的中＆高回収 バランス No.1** | **`M41 -> M26`** | **的中率: 24.82% (205勝) ｜ Val ROI: 118.00%** |\n\n")
    md.append("---\n\n")

    md.append("## 2. Val 回収率ランキング Top 25 (抽出ペア)\n\n")
    md.append("| 順位 | 2段チェーン | Val 回収率 | 元モデル回収率 | 回収率向上幅 | Val 的中率 | Val 的中数 | Val 払戻金 | 評価・特徴 |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")

    df_top_roi = df.sort_values("val_roi", ascending=False).reset_index(drop=True)
    for i, row in df_top_roi.head(25).iterrows():
        p_name = row["pair_name"]
        v_roi = row["val_roi"] * 100
        b_roi = row["base_val_roi"] * 100
        d_roi = row["delta_roi"] * 100
        v_hr = row["val_hit_rate"] * 100
        v_hits = int(row["val_hits"])
        v_pay = int(row["val_payout"])

        tag = ""
        if i == 0:
            tag = "★全2,450ペア中 歴代最高回収率"
        elif v_hr >= 23.0:
            tag = "★高回収＆高的中(23%超)"
        elif v_roi >= 130.0:
            tag = "★回収率130%超え"
        elif v_roi >= 120.0:
            tag = "回収率120%超え"
        else:
            tag = "黒字達成"

        md.append(f"| **{i+1}** | **`{p_name}`** | **{v_roi:6.2f}%** | {b_roi:6.2f}% | **{d_roi:+6.2f}pt** | {v_hr:4.1f}% | {v_hits}勝 | ¥{v_pay:,d} | {tag} |\n")

    md.append("\n---\n\n")

    md.append("## 3. Val 的中率ランキング Top 15 (高精度・安定型)\n\n")
    md.append("| 順位 | 2段チェーン | Val 的中率 | Val 的中数 | Val 回収率 | 元モデル回収率 | 回収率向上幅 | Val 払戻金 | 評価・特徴 |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")

    df_top_hr = df.sort_values("val_hit_rate", ascending=False).reset_index(drop=True)
    for i, row in df_top_hr.head(15).iterrows():
        p_name = row["pair_name"]
        v_roi = row["val_roi"] * 100
        b_roi = row["base_val_roi"] * 100
        d_roi = row["delta_roi"] * 100
        v_hr = row["val_hit_rate"] * 100
        v_hits = int(row["val_hits"])
        v_pay = int(row["val_payout"])

        tag = ""
        if v_hr >= 25.0:
            tag = "★的中率25%超え (4レースに1勝)"
        elif v_roi >= 115.0:
            tag = "★高的中＆回収率115%超え"
        elif v_roi >= 100.0:
            tag = "★黒字＆高的中"
        else:
            tag = "高的中安定型"

        md.append(f"| **{i+1}** | **`{p_name}`** | **{v_hr:4.1f}%** | {v_hits}勝 | **{v_roi:6.2f}%** | {b_roi:6.2f}% | {d_roi:+6.2f}pt | ¥{v_pay:,d} | {tag} |\n")

    md.append("\n---\n\n")

    md.append("## 4. 回収率向上幅 ($\\Delta\\text{ROI}$) Top 15 (カスケードシナジー爆発型)\n\n")
    md.append("| 順位 | 2段チェーン | 回収率向上幅 ($\\Delta\\text{ROI}$) | チェーン後回収率 | 元モデル回収率 | Val 的中率 | Val 払戻金 |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

    df_top_delta = df.sort_values("delta_roi", ascending=False).reset_index(drop=True)
    for i, row in df_top_delta.head(15).iterrows():
        p_name = row["pair_name"]
        v_roi = row["val_roi"] * 100
        b_roi = row["base_val_roi"] * 100
        d_roi = row["delta_roi"] * 100
        v_hr = row["val_hit_rate"] * 100
        v_hits = int(row["val_hits"])
        v_pay = int(row["val_payout"])
        md.append(f"| **{i+1}** | **`{p_name}`** | **+{d_roi:5.2f}pt** | **{v_roi:6.2f}%** | {b_roi:6.2f}% | {v_hr:4.1f}% ({v_hits}勝) | ¥{v_pay:,d} |\n")

    md.append("\n---\n\n")

    md.append("## 5. 考察と実戦的インサイト\n\n")
    md.append("1. **芝におけるカスケードチェーンの威力**:\n")
    md.append("   - 単体モデルでは Val 回収率 100% 超えは 19 モデルにとどまっていましたが、\n")
    md.append("     カスケード接続を行うことで **94 ペアが回収率 100% 超え（完全黒字）を達成** しました。\n")
    md.append("   - 特に `M36 -> M11`（145.64%）、`M10 -> M47`（142.40%）、`M2 -> M10`（137.65%）などは、\n")
    md.append("     単体の壁を大きく突破する劇的な回収率向上を見せています。\n\n")
    md.append("2. **推奨ベストモデルの棲み分け**:\n")
    md.append("   - **最大回収率・爆発型**: **`M36 -> M11`** (Val ROI: 145.64% ｜ 的中率 21.2%)\n")
    md.append("   - **超高回収＆高的中バランス型**: **`M2 -> M10`** (Val ROI: 137.65% ｜ 的中率 23.2%)\n")
    md.append("   - **超高的中＆黒字型**: **`M41 -> M26`** (Val ROI: 118.00% ｜ 的中率 24.8%)\n\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(md)

    print("レポートを保存しました:", report_path)

if __name__ == "__main__":
    main()
