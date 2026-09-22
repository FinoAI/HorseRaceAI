#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "horse_race_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    id<MTLBuffer> linear_w;
    id<MTLBuffer> linear_b;
    id<MTLBuffer> norm_w;
    id<MTLBuffer> norm_b;
    id<MTLBuffer> res_w;
    id<MTLBuffer> res_b;
} BlockWeights;

struct HorseRaceMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    
    // Compute Pipeline States (CUDA kernels converted to Metal)
    id<MTLComputePipelineState> pso_gemv;
    id<MTLComputePipelineState> pso_gelu;
    id<MTLComputePipelineState> pso_layernorm;
    id<MTLComputePipelineState> pso_residual_add;
    id<MTLComputePipelineState> pso_masked_softmax;
    
    uint32_t input_dim;
    uint32_t max_horses;
    uint32_t num_blocks;
    uint32_t first_layer_dim;
    
    // 入力層
    id<MTLBuffer> in_linear_w;
    id<MTLBuffer> in_linear_b;
    id<MTLBuffer> in_norm_w;
    id<MTLBuffer> in_norm_b;
    
    // 中間残差ブロック
    BlockWeights* blocks;
    
    // 最終出力ヘッド
    uint32_t head_in_dim;
    id<MTLBuffer> head_w;
    id<MTLBuffer> head_b;
    
    // 順伝播用中間テンソルバッファ (Ping-Pong buffers)
    id<MTLBuffer> buf_a;
    id<MTLBuffer> buf_b;
    id<MTLBuffer> buf_res;
    id<MTLBuffer> buf_in;
    id<MTLBuffer> buf_mask;
    id<MTLBuffer> buf_probs;
};

// ヘルパー: 重みバッファの読み込みと確保
static id<MTLBuffer> read_and_create_buffer(id<MTLDevice> device, FILE* f, size_t num_elements) {
    size_t bytes = num_elements * sizeof(float);
    id<MTLBuffer> buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (fread([buf contents], sizeof(float), num_elements, f) != num_elements) {
        fprintf(stderr, "[HorseRaceMetal] Error reading weights tensor (%zu elements)\n", num_elements);
        return nil;
    }
    return buf;
}

HorseRaceMetalContext* horse_race_metal_init(const char* weights_path, const char* metal_source_path) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[HorseRaceMetal] Error: No Metal capable device found.\n");
            return NULL;
        }
        
        printf("[HorseRaceMetal] Initialized Metal Device: %s\n", [[device name] UTF8String]);
        
        // 1. Metal シェーダソースのロードとランタイムコンパイル
        const char* m_path = metal_source_path ? metal_source_path : "src/metal/kernels.metal";
        NSError* error = nil;
        NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:m_path]
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
        if (!src) {
            fprintf(stderr, "[HorseRaceMetal] Failed to load Metal source from %s: %s\n",
                    m_path, [[error localizedDescription] UTF8String]);
            return NULL;
        }
        
        id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
        if (!library) {
            fprintf(stderr, "[HorseRaceMetal] Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }
        
        // パイプライン状態の生成 (CUDAカーネルに対応)
        id<MTLFunction> fn_gemv = [library newFunctionWithName:@"gemv_bias_kernel"];
        id<MTLFunction> fn_gelu = [library newFunctionWithName:@"gelu_kernel"];
        id<MTLFunction> fn_ln   = [library newFunctionWithName:@"layernorm_kernel"];
        id<MTLFunction> fn_res  = [library newFunctionWithName:@"residual_add_kernel"];
        id<MTLFunction> fn_sm   = [library newFunctionWithName:@"masked_softmax_kernel"];
        
        id<MTLComputePipelineState> pso_gemv = [device newComputePipelineStateWithFunction:fn_gemv error:&error];
        id<MTLComputePipelineState> pso_gelu = [device newComputePipelineStateWithFunction:fn_gelu error:&error];
        id<MTLComputePipelineState> pso_ln   = [device newComputePipelineStateWithFunction:fn_ln error:&error];
        id<MTLComputePipelineState> pso_res  = [device newComputePipelineStateWithFunction:fn_res error:&error];
        id<MTLComputePipelineState> pso_sm   = [device newComputePipelineStateWithFunction:fn_sm error:&error];
        
        if (!pso_gemv || !pso_gelu || !pso_ln || !pso_res || !pso_sm) {
            fprintf(stderr, "[HorseRaceMetal] Failed to create compute pipeline states: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }
        
        // 2. 重みバイナリのロード
        FILE* f = fopen(weights_path, "rb");
        if (!f) {
            fprintf(stderr, "[HorseRaceMetal] Failed to open weights file: %s\n", weights_path);
            return NULL;
        }
        
        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "HRM1", 4) != 0) {
            fprintf(stderr, "[HorseRaceMetal] Invalid weights format (magic mismatch)\n");
            fclose(f);
            return NULL;
        }
        
        uint32_t input_dim, max_horses, num_blocks;
        fread(&input_dim, sizeof(uint32_t), 1, f);
        fread(&max_horses, sizeof(uint32_t), 1, f);
        fread(&num_blocks, sizeof(uint32_t), 1, f);
        
        uint32_t in_d0, first_layer_dim;
        fread(&in_d0, sizeof(uint32_t), 1, f);
        fread(&first_layer_dim, sizeof(uint32_t), 1, f);
        
        HorseRaceMetalContext* ctx = (HorseRaceMetalContext*)calloc(1, sizeof(HorseRaceMetalContext));
        ctx->device = device;
        ctx->queue = [device newCommandQueue];
        ctx->pso_gemv = pso_gemv;
        ctx->pso_gelu = pso_gelu;
        ctx->pso_layernorm = pso_ln;
        ctx->pso_residual_add = pso_res;
        ctx->pso_masked_softmax = pso_sm;
        
        ctx->input_dim = input_dim;
        ctx->max_horses = max_horses;
        ctx->num_blocks = num_blocks;
        ctx->first_layer_dim = first_layer_dim;
        
        printf("[HorseRaceMetal] Model Architecture: input_dim=%u, 1st_layer=%u (x%u), num_blocks=%u, max_horses=%u\n",
               input_dim, first_layer_dim, first_layer_dim / input_dim, num_blocks, max_horses);
        
        // 各ブロックの次元読み取り
        ctx->blocks = (BlockWeights*)calloc(num_blocks, sizeof(BlockWeights));
        for (uint32_t i = 0; i < num_blocks; ++i) {
            fread(&ctx->blocks[i].in_dim, sizeof(uint32_t), 1, f);
            fread(&ctx->blocks[i].out_dim, sizeof(uint32_t), 1, f);
        }
        
        uint32_t head_in, head_out;
        fread(&head_in, sizeof(uint32_t), 1, f);
        fread(&head_out, sizeof(uint32_t), 1, f);
        ctx->head_in_dim = head_in;
        
        // 重みテンソルのロード
        // (1) 入力層
        ctx->in_linear_w = read_and_create_buffer(device, f, first_layer_dim * input_dim);
        ctx->in_linear_b = read_and_create_buffer(device, f, first_layer_dim);
        ctx->in_norm_w   = read_and_create_buffer(device, f, first_layer_dim);
        ctx->in_norm_b   = read_and_create_buffer(device, f, first_layer_dim);
        
        // (2) 残差ブロック
        for (uint32_t i = 0; i < num_blocks; ++i) {
            uint32_t in_d = ctx->blocks[i].in_dim;
            uint32_t out_d = ctx->blocks[i].out_dim;
            
            ctx->blocks[i].linear_w = read_and_create_buffer(device, f, out_d * in_d);
            ctx->blocks[i].linear_b = read_and_create_buffer(device, f, out_d);
            ctx->blocks[i].norm_w   = read_and_create_buffer(device, f, out_d);
            ctx->blocks[i].norm_b   = read_and_create_buffer(device, f, out_d);
            ctx->blocks[i].res_w    = read_and_create_buffer(device, f, out_d * in_d);
            ctx->blocks[i].res_b    = read_and_create_buffer(device, f, out_d);
        }
        
        // (3) 出力ヘッド
        ctx->head_w = read_and_create_buffer(device, f, max_horses * head_in);
        ctx->head_b = read_and_create_buffer(device, f, max_horses);
        
        fclose(f);
        
        // 3. 中間バッファのアロケート (最大隠れ層次元)
        size_t max_buf_bytes = first_layer_dim * sizeof(float);
        ctx->buf_a = [device newBufferWithLength:max_buf_bytes options:MTLResourceStorageModeShared];
        ctx->buf_b = [device newBufferWithLength:max_buf_bytes options:MTLResourceStorageModeShared];
        ctx->buf_res = [device newBufferWithLength:max_buf_bytes options:MTLResourceStorageModeShared];
        
        ctx->buf_in = [device newBufferWithLength:input_dim * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_mask = [device newBufferWithLength:max_horses * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_probs = [device newBufferWithLength:max_horses * sizeof(float) options:MTLResourceStorageModeShared];
        
        printf("[HorseRaceMetal] All model weights and GPU buffers initialized successfully.\n");
        return ctx;
    }
}

int horse_race_metal_predict(
    HorseRaceMetalContext* ctx,
    const float* race_features,
    const float* horse_mask,
    float* out_probs
) {
    if (!ctx || !race_features || !horse_mask || !out_probs) return -1;
    
    @autoreleasepool {
        // 入力データとマスクを GPU バッファへコピー
        memcpy([ctx->buf_in contents], race_features, ctx->input_dim * sizeof(float));
        memcpy([ctx->buf_mask contents], horse_mask, ctx->max_horses * sizeof(float));
        
        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        
        float eps = 1e-5f;
        
        // 1. 入力層: gemv (x -> buf_a) + LayerNorm + GELU
        // (a) GEMV
        [enc setComputePipelineState:ctx->pso_gemv];
        [enc setBuffer:ctx->in_linear_w offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_in offset:0 atIndex:1];
        [enc setBuffer:ctx->in_linear_b offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_a offset:0 atIndex:3];
        [enc setBytes:&ctx->first_layer_dim length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&ctx->input_dim length:sizeof(uint32_t) atIndex:5];
        
        MTLSize grid_gemv = MTLSizeMake(ctx->first_layer_dim, 1, 1);
        MTLSize thread_group_gemv = MTLSizeMake(MIN(ctx->first_layer_dim, 256), 1, 1);
        [enc dispatchThreads:grid_gemv threadsPerThreadgroup:thread_group_gemv];
        
        // (b) LayerNorm
        [enc setComputePipelineState:ctx->pso_layernorm];
        [enc setBuffer:ctx->buf_a offset:0 atIndex:0];
        [enc setBuffer:ctx->in_norm_w offset:0 atIndex:1];
        [enc setBuffer:ctx->in_norm_b offset:0 atIndex:2];
        [enc setBytes:&ctx->first_layer_dim length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        
        MTLSize grid_ln = MTLSizeMake(1, 1, 1); // 1 threadgroup
        MTLSize thread_group_ln = MTLSizeMake(256, 1, 1);
        [enc dispatchThreadgroups:grid_ln threadsPerThreadgroup:thread_group_ln];
        
        // (c) GELU
        [enc setComputePipelineState:ctx->pso_gelu];
        [enc setBuffer:ctx->buf_a offset:0 atIndex:0];
        [enc setBytes:&ctx->first_layer_dim length:sizeof(uint32_t) atIndex:1];
        [enc dispatchThreads:grid_gemv threadsPerThreadgroup:thread_group_gemv];
        
        // 2. ピラミッド型残差ブロックの順伝播
        id<MTLBuffer> cur_in = ctx->buf_a;
        id<MTLBuffer> cur_out = ctx->buf_b;
        
        for (uint32_t b = 0; b < ctx->num_blocks; ++b) {
            uint32_t in_d = ctx->blocks[b].in_dim;
            uint32_t out_d = ctx->blocks[b].out_dim;
            
            // 主分岐: Linear (cur_in -> cur_out)
            [enc setComputePipelineState:ctx->pso_gemv];
            [enc setBuffer:ctx->blocks[b].linear_w offset:0 atIndex:0];
            [enc setBuffer:cur_in offset:0 atIndex:1];
            [enc setBuffer:ctx->blocks[b].linear_b offset:0 atIndex:2];
            [enc setBuffer:cur_out offset:0 atIndex:3];
            [enc setBytes:&out_d length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&in_d length:sizeof(uint32_t) atIndex:5];
            
            MTLSize grid_b = MTLSizeMake(out_d, 1, 1);
            MTLSize tg_b = MTLSizeMake(MIN(out_d, 256), 1, 1);
            [enc dispatchThreads:grid_b threadsPerThreadgroup:tg_b];
            
            // LayerNorm
            [enc setComputePipelineState:ctx->pso_layernorm];
            [enc setBuffer:cur_out offset:0 atIndex:0];
            [enc setBuffer:ctx->blocks[b].norm_w offset:0 atIndex:1];
            [enc setBuffer:ctx->blocks[b].norm_b offset:0 atIndex:2];
            [enc setBytes:&out_d length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:grid_ln threadsPerThreadgroup:thread_group_ln];
            
            // GELU
            [enc setComputePipelineState:ctx->pso_gelu];
            [enc setBuffer:cur_out offset:0 atIndex:0];
            [enc setBytes:&out_d length:sizeof(uint32_t) atIndex:1];
            [enc dispatchThreads:grid_b threadsPerThreadgroup:tg_b];
            
            // 残差分岐: Linear (cur_in -> buf_res)
            [enc setComputePipelineState:ctx->pso_gemv];
            [enc setBuffer:ctx->blocks[b].res_w offset:0 atIndex:0];
            [enc setBuffer:cur_in offset:0 atIndex:1];
            [enc setBuffer:ctx->blocks[b].res_b offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_res offset:0 atIndex:3];
            [enc setBytes:&out_d length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&in_d length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:grid_b threadsPerThreadgroup:tg_b];
            
            // 残差加算: cur_out += buf_res
            [enc setComputePipelineState:ctx->pso_residual_add];
            [enc setBuffer:cur_out offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_res offset:0 atIndex:1];
            [enc setBytes:&out_d length:sizeof(uint32_t) atIndex:2];
            [enc dispatchThreads:grid_b threadsPerThreadgroup:tg_b];
            
            // Ping-Pong 交換
            id<MTLBuffer> tmp = cur_in;
            cur_in = cur_out;
            cur_out = tmp;
        }
        
        // 3. 最終出力ヘッド: Linear (cur_in -> buf_res[18])
        [enc setComputePipelineState:ctx->pso_gemv];
        [enc setBuffer:ctx->head_w offset:0 atIndex:0];
        [enc setBuffer:cur_in offset:0 atIndex:1];
        [enc setBuffer:ctx->head_b offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_res offset:0 atIndex:3];
        [enc setBytes:&ctx->max_horses length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&ctx->head_in_dim length:sizeof(uint32_t) atIndex:5];
        
        MTLSize grid_head = MTLSizeMake(ctx->max_horses, 1, 1);
        MTLSize tg_head = MTLSizeMake(ctx->max_horses, 1, 1);
        [enc dispatchThreads:grid_head threadsPerThreadgroup:tg_head];
        
        // 4. マスク付き Softmax (buf_res[18] + mask -> buf_probs[18])
        [enc setComputePipelineState:ctx->pso_masked_softmax];
        [enc setBuffer:ctx->buf_res offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_mask offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_probs offset:0 atIndex:2];
        [enc setBytes:&ctx->max_horses length:sizeof(uint32_t) atIndex:3];
        
        MTLSize grid_sm = MTLSizeMake(1, 1, 1);
        MTLSize tg_sm = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid_sm threadsPerThreadgroup:tg_sm];
        
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        
        // 結果を取り出し
        memcpy(out_probs, [ctx->buf_probs contents], ctx->max_horses * sizeof(float));
        return 0;
    }
}

int horse_race_metal_get_feature_dim(const HorseRaceMetalContext* ctx) {
    if (!ctx || ctx->max_horses == 0) return 0;
    return ctx->input_dim / ctx->max_horses;
}

void horse_race_metal_free(HorseRaceMetalContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        if (ctx->blocks) free(ctx->blocks);
        free(ctx);
    }
}
