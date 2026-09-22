#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "horse_race_metal.h"

int main(int argc, char* argv[]) {
    printf("=================================================================\n");
    printf("     ★ 競馬AI 勝ち馬予想推論エンジン (CUDA to Metal C Edition) ★\n");
    printf("=================================================================\n");

    const char* weights_path = "artifacts_models/model_weights_test.bin";
    const char* metal_source = "src/metal/kernels.metal";

    if (argc > 1) {
        weights_path = argv[1];
    }
    if (argc > 2) {
        metal_source = argv[2];
    }

    printf("[Main] Loading model weights: %s\n", weights_path);
    printf("[Main] Loading Metal shader : %s\n", metal_source);

    // 1. Metal 推論エンジンの初期化
    HorseRaceMetalContext* ctx = horse_race_metal_init(weights_path, metal_source);
    if (!ctx) {
        fprintf(stderr, "[Main] Failed to initialize HorseRaceMetal engine.\n");
        return 1;
    }

    int feat_dim = horse_race_metal_get_feature_dim(ctx);
    int total_input_dim = MAX_HORSES_PER_RACE * feat_dim;
    printf("[Main] Feature count per horse (D): %d | Total race input dim: %d\n", feat_dim, total_input_dim);

    // 2. サンプルレースデータの作成 (18頭枠、うち14頭立てのレースをシミュレート)
    float* race_features = (float*)calloc(total_input_dim, sizeof(float));
    float horse_mask[MAX_HORSES_PER_RACE] = {0};
    int running_horses = 14; // 14頭出走

    srand(42);
    for (int h = 0; h < MAX_HORSES_PER_RACE; ++h) {
        if (h < running_horses) {
            horse_mask[h] = 1.0f; // 出走
            // 各馬の特徴量をダミー設定 (正規化済み数値を模倣)
            for (int f = 0; f < feat_dim; ++f) {
                race_features[h * feat_dim + f] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
        } else {
            horse_mask[h] = 0.0f; // 未出走 (パディング枠)
            for (int f = 0; f < feat_dim; ++f) {
                race_features[h * feat_dim + f] = 0.0f;
            }
        }
    }

    // 3. Metal GPU 推論の実行
    float out_probs[MAX_HORSES_PER_RACE] = {0};
    printf("\n[Main] Executing Race-Level Inference on Apple Metal GPU...\n");

    clock_t start = clock();
    int ret = horse_race_metal_predict(ctx, race_features, horse_mask, out_probs);
    clock_t end = clock();

    if (ret != 0) {
        fprintf(stderr, "[Main] Inference failed with error code: %d\n", ret);
        free(race_features);
        horse_race_metal_free(ctx);
        return 1;
    }

    double elapsed_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("[Main] GPU Inference Completed in %.2f ms!\n", elapsed_ms);

    // 4. 推論結果の表示
    printf("\n-----------------------------------------------------------------\n");
    printf(" 馬番 | 出走状況 | 予想勝率 (Probability) | グラフ表示\n");
    printf("-----------------------------------------------------------------\n");

    int best_horse = -1;
    float max_prob = -1.0f;
    float sum_probs = 0.0f;

    for (int h = 0; h < MAX_HORSES_PER_RACE; ++h) {
        int horse_num = h + 1;
        float p = out_probs[h];
        sum_probs += p;

        if (horse_mask[h] > 0.5f) {
            if (p > max_prob) {
                max_prob = p;
                best_horse = horse_num;
            }
            int bar_len = (int)(p * 100.0f / 2.0f); // 50% = 25 chars
            char bar[64];
            memset(bar, '#', bar_len);
            bar[bar_len] = '\0';

            printf("  %2d  |   出走   |       %6.2f%%         | %s\n", horse_num, p * 100.0f, bar);
        } else {
            printf("  %2d  |   取消   |        0.00%%         | (未出走枠・マスク済み)\n", horse_num);
        }
    }

    printf("-----------------------------------------------------------------\n");
    printf(" 出走馬の勝率合計: %.4f (1.0000 に正規化)\n", sum_probs);
    printf(" ★ AI予想 本命最有力馬: 【 馬番 %d 】 (推定勝率: %.2f%%)\n", best_horse, max_prob * 100.0f);
    printf("=================================================================\n\n");

    // クリーンアップ
    free(race_features);
    horse_race_metal_free(ctx);
    printf("[Main] Metal resources successfully released.\n");
    return 0;
}
