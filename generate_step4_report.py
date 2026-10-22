import pandas as pd

df_ext = pd.read_csv('HorseRaceAI/artifacts_turf_cascade_step4/extracted_turf_step4_quads.csv')
df_all = pd.read_csv('HorseRaceAI/artifacts_turf_cascade_step4/all_turf_3680_step4_results.csv')

hr25_df = df_ext[df_ext['is_hit_rate_25'] == True].sort_values(by=['s4_val_hit_rate', 's4_val_roi'], ascending=[False, False])
improved_df = df_ext[df_ext['is_improved'] == True].sort_values(by='s4_val_roi', ascending=False)

def make_table(subset_df):
    lines = []
    lines.append('| 順位 | クアッド構成 (4段チェーン) | Val 的中率 | 的中数 | Val 回収率 | 払戻総額 | 黒字判定 | 元3段 的中率 | 的中率向上 (ΔHit) | 元3段 回収率 | 回収率向上 (ΔROI) |')
    lines.append('|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')
    for rank, (_, row) in enumerate(subset_df.iterrows(), 1):
        kuroji = '★黒字' if row['s4_val_roi'] >= 1.0 else '-'
        lines.append(
            f"| {rank} | **{row['quad_name']}** | **{row['s4_val_hit_rate']*100:.2f}%** | "
            f"{int(row['s4_val_hits'])}/826 | **{row['s4_val_roi']*100:.2f}%** | ¥{int(row['s4_val_payout']):,} | "
            f"{kuroji} | {row['s3_val_hit_rate']*100:.2f}% | +{row['delta_hit_rate']*100:.2f}pt | "
            f"{row['s3_val_roi']*100:.2f}% | {row['delta_roi']*100:+.2f}pt |"
        )
    return '\n'.join(lines)

table_hr25 = make_table(hr25_df)
table_imp = make_table(improved_df)

report = f"""# 芝（Turf）4段カスケード総当たり探索 完了レポート（案1）

## 1. エグゼクティブサマリー

芝（1勝クラス以上・VAL 826レース）において、3段カスケード探索で抽出された**的中率23.0%以上の80トリプレット**を起点とし、残る候補46モデル（`M45`除外）との**4段カスケード総当たり探索（計 80 × 46 = 3,680 通り）**を完全走破いたしました。

### 📊 探索および抽出統計
- **総試行数**: **3,680 通り（100.0% 完全走破完了）**
- **的中率 25.0% 以上達成モデル**: **16 クアッド**
  - 最高的中率: **26.63%** (`M24 -> M16 -> M46 -> M11`)
- **的中率 26.0% 以上達成モデル**: **4 クアッド**
- **「的中率 ≥ 25%」かつ「完全黒字（回収率 ≥ 100%）」の最高峰モデル**: **4 クアッド**
  - うち1モデル（`M43 -> M11 -> M49 -> M50`）は、**的中率 25.06% ＆ 回収率 125.33%（元3段からダブル向上）** を達成！
- **モデル重み保存数**: 抽出された **21基のクアッド重み（.pth）すべて保存完了**

---

## 2. 注目すべき最高峰モデル（神クアッド）

### 👑 【超高回収 ＆ 的中率25%超 ダブル向上 No.1】 `M43 -> M11 -> M49 -> M50`
- **Val 的中率**: **25.06%** (207勝 / 826R、元3段 24.94% から向上)
- **Val 回収率**: **125.33%** (払戻金 ¥103,520、元3段 100.42% から **+24.91pt 爆上げ**)
- **純利益**: **+20,920 円** (投資82,600円)
- **特徴**: 的中率25%の超高頻度的中を維持しながら、4段目に M50 を接続したことで回収率が125.3%まで跳ね上がった、本探索屈指の攻守兼備最強モデル。

### 🎯 【最高的中率 ＆ 黒字 No.1】 `M13 -> M37 -> M31 -> M17`
- **Val 的中率**: **26.51%** (219勝 / 826R、約3.7レースに1勝ペース！)
- **Val 回収率**: **108.91%** (払戻金 ¥89,960 / 黒字達成)
- **特徴**: 826レース中219勝という圧倒的な的中力を誇り、かつ回収率も108.9%と完全黒字を確保。

### 🏹 【歴代最高的中率 No.1】 `M24 -> M16 -> M46 -> M11`
- **Val 的中率**: **26.63%** (220勝 / 826R、元3段 23.00% から **+3.63pt 向上**)
- **Val 回収率**: **91.10%** (払戻金 ¥75,250)
- **特徴**: 全3,680通り中トップの的中率 26.63% を記録。

### ⚡ 【バランス黒字 ＆ ダブル向上 No.2】 `M6 -> M30 -> M5 -> M50`
- **Val 的中率**: **25.18%** (208勝 / 826R、元3段 23.73% から **+1.45pt 向上**)
- **Val 回収率**: **104.54%** (払戻金 ¥86,350、元3段 93.66% から **+10.88pt 向上**)

---

## 3. 的中率 25.0% 以上達成モデル 一覧（全16クアッド）

{table_hr25}

---

## 4. チェーン元より「回収率・的中率が両方向上した黒字モデル」一覧（全8クアッド）

{table_imp}

---

## 5. 保存アーティファクト

- **全3,680通りログCSV**: `HorseRaceAI/artifacts_turf_cascade_step4/all_turf_3680_step4_results.csv`
- **厳選21クアッドログCSV**: `HorseRaceAI/artifacts_turf_cascade_step4/extracted_turf_step4_quads.csv`
- **学習済みモデル重み**: `HorseRaceAI/artifacts_turf_cascade_step4/found_M*.pth` (全21基保存済み)
"""

with open('HorseRaceAI/TURF_CASCADE_STEP4_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)

print('Successfully generated HorseRaceAI/TURF_CASCADE_STEP4_REPORT.md')
