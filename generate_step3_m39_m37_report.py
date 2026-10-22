import os
import pandas as pd

def generate_report():
    csv_path = "HorseRaceAI/artifacts_cascade_step3_m39_m37/cascade_step3_m39_m37_all_48.csv"
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)

    lines = []
    lines.append("# 3段カスケード（M39 -> M37 -> Mx）残り48モデル総当たり探索 最終評価レポート")
    lines.append("")
    lines.append("> **ベースライン (Step 2)**: `M39 -> M37`（Val ROI: 110.28%, Test ROI: 126.95%, 全2,450ペア中 歴代最高峰ペア）  ")
    lines.append("> **探索対象**: 残り全48モデル（50モデル中、M39・M37を除く全モデル）  ")
    lines.append("> **学習条件**: 2009〜2024年 (14,742R) のみで新規NN再学習 $\\to$ Val (前半751R) で最良モデル選定 $\\to$ Test (後半751R) で完全ブラインド評価")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. エグゼクティブサマリー")
    lines.append("")
    lines.append("| 評価項目 | 結果 | 備考 |")
    lines.append("| :--- | :---: | :--- |")
    lines.append("| **総当たり試行モデル数** | **48 モデル** | 100.0% 完全走破完了 (所要時間: 73.4分) |")
    lines.append("| **Val 回収率 $\\ge$ 100% (黒字)** | **4 モデル** | M16 (116.2%), M32 (110.9%), M34 (100.7%), M46 (100.0%) |")
    lines.append("| **Val 回収率がベースラインより向上** | **2 モデル** | M16 (+5.87pt), M32 (+0.63pt) |")
    lines.append("| **総合バランス No.1** | **`M39 -> M37 -> M46`** | **総合 96.99%** (Val: 100.01%, Test: 93.97%) |")
    lines.append("| **テスト回収率 Top 1** | **`M39 -> M37 -> M24`** | **Test: 98.03%**, Test的中率: 22.0% (165勝) |")
    lines.append("| **Val 回収率 Top 1** | **`M39 -> M37 -> M16`** | **Val: 116.15% (+5.87pt)**, 払戻金 ¥87,230 |")
    lines.append("| **超高的中率 Top 1** | **`M39 -> M37 -> M10`** | **的中率: 23.6% (177勝)**, Test: 87.42%, Val: 89.25% |")
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
        tag = "★総合トップ＆Val黒字" if i == 0 else ("★Val黒字" if r["s3_val_roi"] >= 1.0 else "高バランス")
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['comb_roi']*100:.2f}%** | {r['s3_val_roi']*100:.2f}% | {r['s3_test_roi']*100:.2f}% | {r['val_hit_rate']*100:.1f}% ({int(r['val_hits'])}勝) | {r['test_hit_rate']*100:.1f}% ({int(r['test_hits'])}勝) | ¥{r['test_payout']:,.0f} | {tag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 【Test 回収率重視】テストデータ回収率ランキング Top 10")
    lines.append("")
    lines.append("| 順位 | 3段チェーン | Test 回収率 | Val 回収率 | Test 的中率 | Test 平均オッズ | Test 払戻金 | 評価 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    top_test = df.sort_values(by="s3_test_roi", ascending=False).head(10).reset_index(drop=True)
    for i, r in top_test.iterrows():
        tag = "★Test最高峰 (約98%)" if i == 0 else ("Test 90%超" if r["s3_test_roi"] >= 0.90 else "高水準")
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['s3_test_roi']*100:.2f}%** | {r['s3_val_roi']*100:.2f}% | {r['test_hit_rate']*100:.1f}% ({int(r['test_hits'])}勝) | {r['test_mean_odds']:.1f}倍 | ¥{r['test_payout']:,.0f} | {tag} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### 【Val 回収率重視】検証データ黒字化モデル（Val $\\ge$ 100% 全4モデル）")
    lines.append("")
    lines.append("| 順位 | 3段チェーン | Val 回収率 | $\\Delta$Val | Test 回収率 | Val 的中率 | Val 払戻金 | 評価 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    top_val = df[df["s3_val_roi"] >= 1.0].sort_values(by="s3_val_roi", ascending=False).reset_index(drop=True)
    for i, r in top_val.iterrows():
        d_val = f"{r['delta_val_roi']*100:+.2f}pt"
        lines.append(f"| **{i+1}** | **`{r['chain_name']}`** | **{r['s3_val_roi']*100:.2f}%** | {d_val} | {r['s3_test_roi']*100:.2f}% | {r['val_hit_rate']*100:.1f}% ({int(r['val_hits'])}勝) | ¥{r['val_payout']:,.0f} | ★Val黒字達成 |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 全48モデル完全記録一覧表")
    lines.append("")
    lines.append("| モデル | 3段チェーン名 | Val 回収率 | Test 回収率 | $\\Delta$Val | $\\Delta$Test | Test 的中率 |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    for i, r in df.sort_values(by="step3_model").iterrows():
        lines.append(f"| M{int(r['step3_model']):02d} | `{r['chain_name']}` | {r['s3_val_roi']*100:.2f}% | {r['s3_test_roi']*100:.2f}% | {r['delta_val_roi']*100:+.2f}pt | {r['delta_test_roi']*100:+.2f}pt | {r['test_hit_rate']*100:.1f}% |")

    out_path = "HorseRaceAI/CASCADE_STEP3_M39_M37_REPORT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report saved successfully to:", out_path)

if __name__ == "__main__":
    generate_report()
