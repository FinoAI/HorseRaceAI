import os
import pandas as pd

def generate_report():
    csv_path = "HorseRaceAI/artifacts_cascade_step3_m28_m26/cascade_step3_m28_m26_all_48.csv"
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    lines = []
    lines.append("# 3段カスケード（M28 -> M26 -> Mx）残り48モデル総当たり探索 最終評価レポート")
    lines.append("")
    lines.append("> **ベースライン (Step 2)**: `M28 -> M26`（Val ROI: 114.57%, Test ROI: 104.30%, Double 100% 達成ペア）  ")
    lines.append("> **探索対象**: 残り全48モデル（50モデル中、M28・M26を除く全モデル）  ")
    lines.append("> **学習条件**: 2009〜2024年 (14,742R) のみで新規NN再学習 $\\to$ Val (前半751R) で最良モデル選定 $\\to$ Test (後半751R) で完全ブラインド評価")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. エグゼクティブサマリー")
    lines.append("")
    lines.append("| 評価項目 | 結果 | 備考 |")
    lines.append("| :--- | :---: | :--- |")
    lines.append("| **総当たり試行モデル数** | **48 モデル** | 100.0% 完全走破完了 (所要時間: 71.6分) |")
    lines.append("| **Val 回収率 $\\ge$ 100% (黒字)** | **4 モデル** | M40 (106.7%), M37 (105.6%), M32 (105.5%), M31 (105.0%) |")
    lines.append("| **Test 回収率 $\\ge$ 90% (高精度・準黒字)** | **3 モデル** | M45 (97.84%), M8 (94.53%), M43 (90.95%) |")
    lines.append("| **総合バランス No.1** | **`M28 -> M26 -> M27`** | **総合 90.46%** (Val: 92.12%, Test: 88.80%) |")
    lines.append("| **テスト回収率 Top 1** | **`M28 -> M26 -> M45`** | **Test: 97.84%**, 払戻金 ¥73,480 |")
    lines.append("| **高精度・高的中 Top 1** | **`M28 -> M26 -> M43`** | **Test: 90.95%**, Val: 86.36%, Test的中率: **22.9%** (172勝) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 3段カスケード ベスト組み合わせランキング")
    lines.append("")
    lines.append("### 【総合・安定度トップ】総合回収率（Val + Test 平均）ランキング Top 10")
    lines.append("")
    lines.append("| 順位 | 3段チェーン | 総合回収率 | Val 回収率 | Test 回収率 | Val 的中率 | Test 的中率 | Test 払戻金 | 特徴・評価 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    df["comb_roi"] = (df["s3_val_roi"] + df["s3_test_roi"]) / 2
    top_comb = df.sort_values(by="comb_roi", ascending=False).head(10).reset_index(drop=True)
    for i, r in top_comb.iterrows():
        tag = "★Val黒字" if r["s3_val_roi"] >= 1.0 else "高バランス"
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['comb_roi']*100:.2f}%** | {r['s3_val_roi']*100:.2f}% | {r['s3_test_roi']*100:.2f}% | {r['val_hit_rate']*100:.1f}% ({int(r['val_hits'])}勝) | {r['test_hit_rate']*100:.1f}% ({int(r['test_hits'])}勝) | ¥{r['test_payout']:,.0f} | {tag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 【Test 回収率重視】テストデータ回収率ランキング Top 10")
    lines.append("")
    lines.append("| 順位 | 3段チェーン | Test 回収率 | $\\Delta$Test | Val 回収率 | Test 的中率 | Test 平均オッズ | Test 払戻金 | 評価 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    top_test = df.sort_values(by="s3_test_roi", ascending=False).head(10).reset_index(drop=True)
    for i, r in top_test.iterrows():
        d_te = f"{r['delta_test_roi']*100:+.2f}pt"
        tag = "★Test最高峰 (約98%)" if i == 0 else ("Test 90%超" if r["s3_test_roi"] >= 0.90 else "高水準")
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['s3_test_roi']*100:.2f}%** | {d_te} | {r['s3_val_roi']*100:.2f}% | {r['test_hit_rate']*100:.1f}% ({int(r['test_hits'])}勝) | {r['test_mean_odds']:.1f}倍 | ¥{r['test_payout']:,.0f} | {tag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 【Val 回収率重視】検証データ黒字化モデル（Val $\\ge$ 100% 全4モデル）")
    lines.append("")
    lines.append("| 順位 | 3段チェーン | Val 回収率 | Test 回収率 | Val 的中率 | Val 払戻金 | 評価 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :--- |")

    top_val = df[df["s3_val_roi"] >= 1.0].sort_values(by="s3_val_roi", ascending=False).reset_index(drop=True)
    for i, r in top_val.iterrows():
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['s3_val_roi']*100:.2f}%** | {r['s3_test_roi']*100:.2f}% | {r['val_hit_rate']*100:.1f}% ({int(r['val_hits'])}勝) | ¥{r['val_payout']:,.0f} | ★Val黒字達成 |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 全48モデル完全記録一覧表")
    lines.append("")
    lines.append("| モデル | 3段チェーン名 | Val 回収率 | Test 回収率 | $\\Delta$Val | $\\Delta$Test | Test 的中率 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    for i, r in df.sort_values(by="step3_model").iterrows():
        lines.append(f"| M{int(r['step3_model']):02d} | `{r['chain_name']}` | {r['s3_val_roi']*100:.2f}% | {r['s3_test_roi']*100:.2f}% | {r['delta_val_roi']*100:+.2f}pt | {r['delta_test_roi']*100:+.2f}pt | {r['test_hit_rate']*100:.1f}% |")

    out_path = "HorseRaceAI/CASCADE_STEP3_M28_M26_REPORT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report saved successfully to:", out_path)

if __name__ == "__main__":
    generate_report()
