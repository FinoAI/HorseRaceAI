#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "horse_race_metal.h"

// 高精度ミリ秒タイマー (POSIX clock_gettime)
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("========================================================================================\n");
    printf("        ★ 競馬AI (RaceLevelStage1NN) C言語/Apple Metal GPU ベンチマーク測定 ★\n");
    printf("========================================================================================\n");

    const char* weights_path = "artifacts_models/model_1.bin";
    const char* metal_source = "src/metal/kernels.metal";

    if (argc > 1) weights_path = argv[1];
    if (argc > 2) metal_source = argv[2];

    // ファイルサイズの取得
    struct stat st;
    if (stat(weights_path, &st) != 0) {
        fprintf(stderr, "Error: Could not stat weights file: %s\n", weights_path);
        return 1;
    }
    double file_size_mb = (double)st.st_size / (1024.0 * 1024.0);

    // 1. モデルロード & GPU初期化時間計測
    printf("[1/4] Loading model weights & compiling Metal Shaders...\n");
    double t_load_start = get_time_ms();
    HorseRaceMetalContext* ctx = horse_race_metal_init(weights_path, metal_source);
    double t_load_end = get_time_ms();

    if (!ctx) {
        fprintf(stderr, "Error: Failed to initialize HorseRaceMetal engine.\n");
        return 1;
    }
    double load_time_ms = t_load_end - t_load_start;

    int feat_dim = horse_race_metal_get_feature_dim(ctx);
    int total_input_dim = MAX_HORSES_PER_RACE * feat_dim;

    // 2. テスト入力データの生成 (16頭出走、2頭取消のレース)
    float* race_features = (float*)calloc(total_input_dim, sizeof(float));
    float horse_mask[MAX_HORSES_PER_RACE] = {0};
    int running_horses = 16;

    srand(12345);
    for (int h = 0; h < MAX_HORSES_PER_RACE; ++h) {
        if (h < running_horses) {
            horse_mask[h] = 1.0f;
            for (int f = 0; f < feat_dim; ++f) {
                race_features[h * feat_dim + f] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
            }
        } else {
            horse_mask[h] = 0.0f;
        }
    }

    float out_probs[MAX_HORSES_PER_RACE] = {0};

    // 3. 初回推論 (Cold Start) の計測
    printf("[2/4] Measuring Cold-Start inference latency...\n");
    double t_cold_start = get_time_ms();
    horse_race_metal_predict(ctx, race_features, horse_mask, out_probs);
    double t_cold_end = get_time_ms();
    double cold_latency_ms = t_cold_end - t_cold_start;

    // 4. ウォームアップ後連続推論 (Warm Run: 10回) の計測
    printf("[3/4] Measuring Warm-Run inference latency across 10 iterations...\n");
    int iterations = 10;
    double latencies[iterations];
    double total_warm_ms = 0.0;
    double min_lat_ms = 1e9;
    double max_lat_ms = 0.0;

    for (int i = 0; i < iterations; ++i) {
        double t0 = get_time_ms();
        horse_race_metal_predict(ctx, race_features, horse_mask, out_probs);
        double t1 = get_time_ms();
        
        double lat = t1 - t0;
        latencies[i] = lat;
        total_warm_ms += lat;
        if (lat < min_lat_ms) min_lat_ms = lat;
        if (lat > max_lat_ms) max_lat_ms = lat;
    }

    double avg_warm_ms = total_warm_ms / (double)iterations;
    double throughput_races_per_sec = 1000.0 / avg_warm_ms;

    // 5. 結果サマリーの出力
    printf("\n========================================================================================\n");
    printf("                         ★ ベンチマーク測定結果レポート ★\n");
    printf("========================================================================================\n");
    printf("【1. モデルサイズ・構造仕様】\n");
    printf("  ・対象モデル                 : Stage 1 (前段モデル 10モデル中の1モデル)\n");
    printf("  ・1馬あたりの入力特徴量数 (D) : %d カラム (BAC全量 + 開催年ランダム抽出)\n", feat_dim);
    printf("  ・レース総入力次元           : %d 次元 (出走頭数 %d 頭 × %d 特徴量)\n", total_input_dim, MAX_HORSES_PER_RACE, feat_dim);
    printf("  ・第1層ユニット数 (x32)      : %d ユニット\n", total_input_dim * 32);
    printf("  ・隠れ層の減衰設計           : 最大1/2ずつ段階的に縮小するディープ残差ピラミッド\n");
    printf("  ・ディスク保存容量 (.bin)    : %.2f MB (%.2f GB / %lld bytes)\n", file_size_mb, file_size_mb / 1024.0, (long long)st.st_size);
    printf("  ・生FP32パラメータ数 (総計)   : 約 %lld パラメータ (%.2f M)\n", (long long)(st.st_size / sizeof(float)), (double)(st.st_size / sizeof(float)) / 1000000.0);
    printf("----------------------------------------------------------------------------------------\n");
    printf("【2. GPU推論処理速度 (Apple Metal GPU / C言語)】\n");
    printf("  ・GPUデバイス名              : Apple Metal GPU (macOS)\n");
    printf("  ・モデルロード & GPU初期化時間 : %.2f ms (%.2f 秒)\n", load_time_ms, load_time_ms / 1000.0);
    printf("  ・初回推論時間 (Cold Start)   : %.2f ms\n", cold_latency_ms);
    printf("  ・平均推論時間 (Warm Run)     : %.2f ms / レース\n", avg_warm_ms);
    printf("  ・最速推論時間 (Min Latency)  : %.2f ms\n", min_lat_ms);
    printf("  ・最遅推論時間 (Max Latency)  : %.2f ms\n", max_lat_ms);
    printf("  ・推論スループット (Throughput): %.2f レース / 秒 (1秒間に %.1f レースの予想が可能)\n", throughput_races_per_sec, throughput_races_per_sec);
    printf("----------------------------------------------------------------------------------------\n");
    printf("【3. 予想出力サンプル (16頭出走・2頭取消レース)】\n");
    int best_h = 1;
    float max_p = out_probs[0];
    for (int h = 0; h < MAX_HORSES_PER_RACE; ++h) {
        if (out_probs[h] > max_p) {
            max_p = out_probs[h];
            best_h = h + 1;
        }
    }
    printf("  ・本命最有力馬               : 【 馬番 %2d 】 (推定勝率: %5.2f%%)\n", best_h, max_p * 100.0f);
    printf("  ・取消馬 (馬番17, 18) の勝率 : %5.2f%% (GPUカーネル内で完全に 0.00%% にマスク)\n", out_probs[16] * 100.0f);
    printf("========================================================================================\n\n");

    free(race_features);
    horse_race_metal_free(ctx);
    return 0;
}
