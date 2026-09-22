# 競馬AI 勝ち馬予測・高配当/回収率向上モデル (Race-Level Deep Stacking AI)

本システムは、JRDBの開催年別競走馬データ（約1,380列）およびレース条件データ（`BAC_KAB.csv`）を統合し、**単勝的中率20%超**および**回収率100%超**を達成するために構築された**レース単位入力型アンサンブル・ディープラーニング競馬AI**です。

---

## 1. システム構成・アーキテクチャ概要

### 1.1 レース単位テンソル表現とマスキング
- **最大出走頭数**: $N_{\max} = 18$ 頭（JRAフルゲート）。
- **入力形式**: 1レースあたり **$18 \times D$ 次元**（$D$: 特徴量数）。各出走馬を馬番（1〜18番）順のスロットに配置。
  - 出走頭数が14頭のレースの場合、スロット15〜18には $0.0$ をパディング。
  - 同時に出走馬マスク `(18,)`（出走馬: 1.0、未出走: 0.0）を保持。
- **全馬共通特徴量**: 各モデルでランダムサンプリングされた $D$ 個の特徴量（BAC_KAB 全量 36列 ＋ 開催年ランダム列）は、**そのレース内の全出走馬で完全に共通のセット**として整列されます。

### 1.2 ネットワーク構造 (32倍入力層 ＆ 最大1/2段階的減衰ピラミッド構造)
- **Stage 1: 前段ニューラルネットワーク (10モデル)**
  - 入力第1層: **$(18 \times D) \times 32$ ユニット**（例: 50特徴量の場合、第1層は $900 \times 32 = 28,800$ ユニット）。
  - **隠れ層の減衰ペース**: 急激な圧縮（1/8や1/10）を排し、**各層で最大1/2（半分）ずつ段階的に縮小**する 8〜10層のピラミッド型ディープ構造。
  - 各ブロック: LayerNorm + GELU + Dropout + 残差接続（Residual connection）。
  - 出力層: 18次元（各馬番のロジット）。非出走馬スロットに $-10^9$ を加算してマスキングすることで、**出走馬間だけで勝率の合計が 1.0 となる Softmax 分布**を出力。
- **Stage 2: 後段メタニューラルネットワーク (最終予想モデル)**
  - 前段10モデルの予測勝率（$18 \times 10 = 180$ 次元）＋各馬番ごとのアンサンブル統計量（平均、標準偏差、最大値、最小値 = 72次元）の計 **252次元** を入力。
  - 入力第1層: **$252 \times 32 = 8,064$ ユニット**。
  - 最大1/2ずつ段階的に縮小するディープメタNN。
  - 出力: 18頭分の最終勝率分布（マスク付きSoftmax）。

### 1.3 データ分割と過学習防止 (Train / Val / Test)
- **Train**: 2018年〜2023年
- **Val / Test**: 2024年と2025年の全レースを結合し、レースID単位でシャッフル分割。
  - Stage 2 の過学習を防ぐため、Valデータをさらにレース単位で **`Val_Train` (Stage 2 メタ学習用)** と **`Val_Eval` (Stage 2 Early Stopping用)** に分割。未見の検証データで Early Stopping を判定します。

---

## 2. 払戻金（高配当）最大化評価オプション

`configs/config.yaml` にて評価指標および馬券選定方式を切り替え可能です:

### 2.1 評価関数（Validationモデル選択基準）
- `eval_metric: "payout"`:
  - 選択した馬の払戻金（**`Target_単勝 * (Target_着順 == 1)`**、着外なら 0）を集計。
  - 回収率（総払戻額 / 総投資額）が最大のエポックを最良モデルとして自動保存します。
- `eval_metric: "hybrid"`: 回収率 × 的中率20%制約ペナルティ。
- `eval_metric: "hit_rate"`: 単勝的中率重視。

### 2.2 馬券・予想選定モード (`selection_mode`)
1. **`prob` (予測勝率最大)**: レース内で最も勝率 $P$ が高いと推計された馬を選択。
2. **`ev` (期待値最大 ★高配当狙い)**: $\text{EV} = P \times \text{確定オッズ}$ が最大の馬を選択。
3. **`ev_filtered` (推奨 ★的中率20%担保＋高配当回収)**: 最低勝率 $P \ge 10\%$ を満たす馬の中で $\text{EV}$ が最大の馬を選択。

---

## 3. セットアップと実行方法

### 3.1 必要環境
Python 3.9〜3.12 (GPU環境推奨)

```bash
pip install -r requirements.txt
```

### 3.2 疎通テスト（レース単位データ変換・モデル構造・マスキング検証）
```bash
python test_pipeline_components.py
```

### 3.3 本番モデル学習・評価・シミュレーション実行
```bash
python run_pipeline.py --config configs/config.yaml
```

---

## 4. CUDA to Metal C言語 高速推論エンジン (Apple Metal GPU)

PyTorchで学習したレース単位モデルを、**CUDAカーネルからApple Metal Shading Language (MSL) へ変換したコンピュートシェーダ**および**純粋なC言語インターフェース**を用いて、MacのGPU上で直接高速推論できるネイティブエンジンを搭載しています。

### 4.1 CUDA to Metal 変換対応表
| 概念 | CUDA | Metal (MSL) |
| :--- | :--- | :--- |
| カーネル関数宣言 | `__global__ void kernel(...)` | `kernel void kernel(...)` |
| メモリアドレス空間 | グローバルポインタ | `device float* [[buffer(0)]]` |
| 共有メモリ | `__shared__ float s_mem[N];` | `threadgroup float s_mem[N];` |
| スレッドID | `threadIdx.x` | `uint tid [[thread_position_in_threadgroup]]` |
| グリッド内グローバルID | `blockIdx.x * blockDim.x + tid` | `uint gid [[thread_position_in_grid]]` |
| スレッドグループ同期 | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |

### 4.2 C言語 API (`include/horse_race_metal.h`)
```c
#include "horse_race_metal.h"

// 1. モデルとMetalシェーダの初期化
HorseRaceMetalContext* ctx = horse_race_metal_init("artifacts_models/model_weights_test.bin", "src/metal/kernels.metal");

// 2. レース特徴量と出走馬マスクを渡してGPU推論
float out_probs[18];
horse_race_metal_predict(ctx, race_features, horse_mask, out_probs);

// 3. リソース解放
horse_race_metal_free(ctx);
```

### 4.3 C言語エンジンのビルド & 実行
```bash
# 1. コンパイル
make

# 2. 推論実行
./bin/horse_race_metal_cli artifacts_models/model_weights_test.bin src/metal/kernels.metal

# 3. PyTorchとMetalの数値完全一致検証
python tools/verify_metal_vs_pytorch.py
```
