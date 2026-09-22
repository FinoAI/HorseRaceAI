#ifndef HORSE_RACE_METAL_H
#define HORSE_RACE_METAL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define MAX_HORSES_PER_RACE 18

/**
 * 競馬AI Metal 推論コンテキスト不透明ポインタ
 */
typedef struct HorseRaceMetalContext HorseRaceMetalContext;

/**
 * 競馬AI Metal 推論エンジンの初期化
 * 
 * @param weights_path        エクスポートされた重みバイナリ (model_weights.bin) のパス
 * @param metal_source_path   Metal シェーダソース (kernels.metal) のパス (NULL時は組み込みデフォルト)
 * @return 成功時はコンテキストポインタ、失敗時は NULL
 */
HorseRaceMetalContext* horse_race_metal_init(const char* weights_path, const char* metal_source_path);

/**
 * レース単位の勝ち馬予想推論を実行
 * 
 * @param ctx            初期化済みコンテキスト
 * @param race_features  入力レース特徴量 (18頭 × 特徴量数 D のフラット配列)
 * @param horse_mask     出走馬マスク (18次元: 出走=1.0f, 非出走=0.0f)
 * @param out_probs      出力予測確率バッファ (18次元、各馬の勝率、合計1.0)
 * @return 成功時 0、エラー時 非0
 */
int horse_race_metal_predict(
    HorseRaceMetalContext* ctx,
    const float* race_features,
    const float* horse_mask,
    float* out_probs
);

/**
 * 入力特徴量数 (D) を取得
 */
int horse_race_metal_get_feature_dim(const HorseRaceMetalContext* ctx);

/**
 * コンテキストの解放とリソース破棄
 */
void horse_race_metal_free(HorseRaceMetalContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* HORSE_RACE_METAL_H */
