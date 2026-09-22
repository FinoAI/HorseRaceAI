#include <metal_stdlib>
using namespace metal;

/*
 * CUDA to Metal Code Conversion (HorseRaceAI GPU Inference Kernels)
 * 
 * -----------------------------------------------------------------------------
 * CUDA Concept                       | Metal (MSL) Equivalent
 * -----------------------------------------------------------------------------
 * __global__ void kernel(...)        | kernel void kernel(...)
 * __shared__ float s_mem[N];         | threadgroup float s_mem[N];
 * blockIdx.x                         | [[threadgroup_position_in_grid]]
 * threadIdx.x                        | [[thread_position_in_threadgroup]]
 * blockDim.x                         | [[threads_per_threadgroup]]
 * gridDim.x                          | [[threadgroups_per_grid]]
 * blockIdx.x * blockDim.x + tid      | [[thread_position_in_grid]]
 * __syncthreads()                    | threadgroup_barrier(mem_flags::mem_threadgroup)
 * -----------------------------------------------------------------------------
 */

// 1. 行列ベクトル積 + バイアス: y = W * x + b
// W: (M, K) row-major, x: (K), b: (M), y: (M)
kernel void gemv_bias_kernel(
    device const float* W          [[buffer(0)]],
    device const float* x          [[buffer(1)]],
    device const float* b          [[buffer(2)]],
    device float*       y          [[buffer(3)]],
    constant uint&      M          [[buffer(4)]],
    constant uint&      K          [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]] // CUDA: blockIdx.x * blockDim.x + threadIdx.x
) {
    if (gid >= M) return;

    float sum = (b != nullptr) ? b[gid] : 0.0f;
    device const float* row = W + gid * K;

    // 内積計算 (SIMD loop)
    for (uint k = 0; k < K; ++k) {
        sum += row[k] * x[k];
    }
    y[gid] = sum;
}

// 2. GELU 活性化関数 (Elementwise GELU)
// CUDA: elementwise 1D kernel
kernel void gelu_kernel(
    device float*  data            [[buffer(0)]],
    constant uint& size            [[buffer(1)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = data[gid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float cdf = 0.5f * (1.0f + metal::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    data[gid] = x * cdf;
}

// 3. レイヤー正規化 (LayerNorm)
// CUDA: __shared__ reduction kernel
kernel void layernorm_kernel(
    device float*       x          [[buffer(0)]],
    device const float* gamma      [[buffer(1)]],
    device const float* beta       [[buffer(2)]],
    constant uint&      dim        [[buffer(3)]],
    constant float&     eps        [[buffer(4)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint threads_per_group         [[threads_per_threadgroup]]
) {
    // 1スレッドグループで1つのレイヤーノルムを実行 (dim <= 32768)
    // ステップ1: 平均の計算
    float local_sum = 0.0f;
    for (uint i = tid; i < dim; i += threads_per_group) {
        local_sum += x[i];
    }

    threadgroup float s_mean[256];
    s_mean[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup); // CUDA: __syncthreads()

    for (uint s = threads_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = s_mean[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ステップ2: 分散の計算
    float local_var = 0.0f;
    for (uint i = tid; i < dim; i += threads_per_group) {
        float diff = x[i] - mean;
        local_var += diff * diff;
    }

    threadgroup float s_var[256];
    s_var[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = threads_per_group / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_var[tid] += s_var[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float var = s_var[0] / float(dim);
    float inv_std = metal::rsqrt(var + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ステップ3: 正規化とスケール/シフト
    for (uint i = tid; i < dim; i += threads_per_group) {
        float val = (x[i] - mean) * inv_std;
        if (gamma != nullptr) val *= gamma[i];
        if (beta != nullptr)  val += beta[i];
        x[i] = val;
    }
}

// 4. 残差加算 (Residual Add: y = x1 + x2)
kernel void residual_add_kernel(
    device float*       y          [[buffer(0)]],
    device const float* res        [[buffer(1)]],
    constant uint&      size       [[buffer(2)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    y[gid] += res[gid];
}

// 5. 出走馬マスキング付き Softmax (Masked Softmax for 18 Horses)
// logits: (18), mask: (18), probs: (18)
kernel void masked_softmax_kernel(
    device const float* logits     [[buffer(0)]],
    device const float* mask       [[buffer(1)]],
    device float*       probs      [[buffer(2)]],
    constant uint&      num_horses [[buffer(3)]],
    uint tid                       [[thread_position_in_threadgroup]]
) {
    // 1スレッドグループ (18スレッド) で Softmax を完了
    threadgroup float s_logits[32];
    threadgroup float s_exp[32];

    if (tid < num_horses) {
        float m = mask[tid];
        // 非出走馬に -1e9f を加算してマスク
        s_logits[tid] = (m > 0.5f) ? logits[tid] : -1e9f;
    } else {
        s_logits[tid] = -1e9f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 最大値探索 (オーバーフロー防止)
    float max_val = -1e9f;
    for (uint i = 0; i < num_horses; ++i) {
        if (s_logits[i] > max_val) {
            max_val = s_logits[i];
        }
    }

    // exp 計算
    float exp_val = 0.0f;
    if (tid < num_horses) {
        if (mask[tid] > 0.5f) {
            exp_val = metal::exp(s_logits[tid] - max_val);
        } else {
            exp_val = 0.0f;
        }
    }
    s_exp[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 合計値計算
    float sum_exp = 0.0f;
    for (uint i = 0; i < num_horses; ++i) {
        sum_exp += s_exp[i];
    }
    if (sum_exp < 1e-12f) sum_exp = 1.0f;

    // 確率出力
    if (tid < num_horses) {
        probs[tid] = s_exp[tid] / sum_exp;
    }
}
