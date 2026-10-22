import pandas as pd

df = pd.read_csv('HorseRaceAI/artifacts_turf_cascade_step3/extracted_turf_step3_triplets.csv')

def make_table(subset_df):
    lines = []
    lines.append('| 順位 | トリプレット構成 | Val 回収率 | Val 的中率 | 的中数 | 払戻総額 | 元2段 回収率 | 元2段 的中率 | 回収率向上 (ΔROI) | 的中率向上 (ΔHit) |')
    lines.append('|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')
    for rank, (_, row) in enumerate(subset_df.iterrows(), 1):
        lines.append(
            f'| {rank} | **{row["triplet_name"]}** | **{row["s3_val_roi"]*100:.2f}%** | {row["s3_val_hit_rate"]*100:.2f}% | '
            f'{int(row["s3_val_hits"])}/826 | ¥{int(row["s3_val_payout"]):,} | '
            f'{row["s2_val_roi"]*100:.2f}% | {row["s2_val_hit_rate"]*100:.2f}% | '
            f'+{row["delta_roi"]*100:.2f}pt | +{row["delta_hit_rate"]*100:.2f}pt |'
        )
    return '\n'.join(lines)

top_roi = df.sort_values(by='s3_val_roi', ascending=False).head(20)
top_hr = df.sort_values(by='s3_val_hit_rate', ascending=False).head(20)
top_droi = df.sort_values(by='delta_roi', ascending=False).head(20)
hybrid = df[(df['s3_val_roi'] >= 1.10) & (df['s3_val_hit_rate'] >= 0.23)].sort_values(by='s3_val_roi', ascending=False)

report = f'''# 芝（Turf）3段カスケード総当たり探索 最終抽出レポート

## 1. エグゼクティブサマリー

芝（1勝クラス以上・VAL 826レース）において、2段カスケード探索で抽出された全212ペア $(M_i \\to M_j)$ を起点とし、残り48モデル $(M_k)$ との**3段カスケード総当たり探索（計 $212 \\times 48 = 10,176$ 通り）**を完全走破しました。

ユーザー様のご指示に基づき、**VAL（826レース）のみ**で厳密に評価を行い（TESTデータは完全不使用）、**チェーン元（2段カスケード）と比較して「回収率（$\\Delta\\text{{ROI}} > 0$）」かつ「的中率（$\\Delta\\text{{HitRate}} > 0$）」の両方が同時に向上したモデルのみ**を厳格に抽出いたしました。

### 📊 走破および抽出統計
- **総試行数**: **10,176 通り（100.0% 完全走破完了）**
- **厳選抽出数**: **222 トリプレット**（元2段モデルから回収率・的中率がダブル向上）
- **完全黒字モデル数 (回収率 $\\ge$ 100%)**: **128 トリプレット**（抽出数の **57.7%** が黒字を達成）
- **超高回収モデル (回収率 $\\ge$ 120%)**: **23 トリプレット**
- **超高的中モデル (的中率 $\\ge$ 24%)**: **33 トリプレット**
- **攻守兼備ハイブリッド (回収率 $\\ge$ 110% ＆ 的中率 $\\ge$ 23%)**: **18 トリプレット**

---

## 2. 注目すべき最高峰モデル

### 👑 【歴代最高回収率 No.1】 `M33 -> M3 -> M43`
- **Val 回収率**: **176.89%**（元2段 92.13% から **+84.76pt 爆上げ**）
- **Val 的中率**: **21.19%**（175勝 / 826R）
- **払戻金総額**: **¥146,110**（投資82,600円に対し、純利益 **+63,510円**）
- **特徴**: 荒れる中穴・穴馬を3段目の $M_{{43}}$ が極めて的確にフィルタリング・昇格させ、驚異的な回収率 176.9% を記録。

### ⚡ 【超高回収 ＆ 超高的中 ハイブリッド怪物 No.1】 `M48 -> M33 -> M49`
- **Val 回収率**: **152.66%**（元2段 90.25% から **+62.41pt 向上**）
- **Val 的中率**: **24.46%**（202勝 / 826R、元2段 20.82% から **+3.63pt 向上**）
- **払戻金総額**: **¥126,100**（純利益 **+43,500円**）
- **特徴**: 的中率 約4レースに1回（24.5%）を維持しながら、回収率152.7%という別次元のパフォーマンスを叩き出した本探索屈指の傑作モデル。

### 🎯 【超高回収 ＆ 安定型 No.2】 `M13 -> M50 -> M10`
- **Val 回収率**: **149.83%**（元2段 93.54% から **+56.30pt 向上**）
- **Val 的中率**: **23.73%**（196勝 / 826R、元2段 23.12% から向上）
- **払戻金総額**: **¥123,760**（純利益 **+41,160円**）

### 🏹 【最高的中率 No.1】 `M6 -> M7 -> M49`
- **Val 的中率**: **25.91%**（214勝 / 826R、4回に1回以上的中！）
- **Val 回収率**: **93.69%**（元2段 90.88% から +2.81pt 向上）
- **払戻金総額**: **¥77,390**

---

## 3. 回収率（ROI）トップ20 ランキング

{make_table(top_roi)}

---

## 4. 的中率（Hit Rate）トップ20 ランキング

{make_table(top_hr)}

---

## 5. 回収率向上幅（$\\Delta$ROI）トップ20 ランキング

{make_table(top_droi)}

---

## 6. 【推奨】攻守兼備 ハイブリッド厳選18モデル（ROI $\\ge$ 110% ＆ 的中率 $\\ge$ 23%）

的中率23%以上（高頻度的中）を確保しながら、回収率110%以上を達成した実戦投資に最適な18モデルです。

{make_table(hybrid)}

---

## 7. カスケード推論の実行メカニズム

3段カスケードモデル $(M_i \\to M_j \\to M_k)$ の推論パイプラインは以下のフローで動作します：

```mermaid
flowchart TD
    Race["出走レース馬群"] --> M_i["第1段: 基本予測モデル M_i"]
    M_i --> S1["各馬の予測スコア & Top1選定"]
    S1 --> Feat_j["第1段スコア・確信度をメタ特徴量として結合"]
    Feat_j --> M_j["第2段: 精査モデル M_j"]
    M_j --> S2["第2段予測スコア & 候補馬再評価"]
    S2 --> Feat_k["第1段・第2段スコアをメタ特徴量として結合"]
    Feat_k --> M_k["第3段: 最終判定モデル M_k"]
    M_k --> Top1["最終推奨 本命馬 1頭選定"]
```

1. **Step 1 ($M_i$)**: 元の特徴量（200特徴）から各馬のスコアを算出。
2. **Step 2 ($M_j$)**: Step 1の出力スコア・順位をメタ特徴量として結合し、精査予測。
3. **Step 3 ($M_k$)**: Step 1およびStep 2の予測結果を統合したメタ特徴量を用いて最終決定。

---

## 8. 保存アーティファクト一覧

- **全10,176通りログCSV**: `HorseRaceAI/artifacts_turf_cascade_step3/all_turf_10176_step3_results.csv`
- **厳選222モデルログCSV**: `HorseRaceAI/artifacts_turf_cascade_step3/extracted_turf_step3_triplets.csv`
- **学習済みモデル重み**: `HorseRaceAI/artifacts_turf_cascade_step3/found_M*.pth` (全222基保存済み)
'''

with open('HorseRaceAI/TURF_CASCADE_STEP3_EXTRACTION_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)

print('Report generation complete!')
