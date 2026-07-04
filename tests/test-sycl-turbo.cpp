#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#define GGML_COMMON_DECL_SYCL
#include "../ggml/src/ggml-common.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>

extern "C" {
    void quantize_row_turbo2_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo2_0(const void * x, float * y, int64_t k);
    void quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
    void quantize_row_turbo4_0_ref(const float * x, void * y, int64_t k);
    void dequantize_row_turbo4_0(const void * x, float * y, int64_t k);
    
    void quantize_row_tq3_1s_ref(const float * x, void * y, int64_t k);
    void dequantize_row_tq3_1s(const void * x, float * y, int64_t k);
    void quantize_row_tq4_1s_ref(const float * x, void * y, int64_t k);
    void dequantize_row_tq4_1s(const void * x, float * y, int64_t k);

    void turbo_cpu_fwht(float * x, int group_size);
}

static bool run_test(ggml_backend_t backend, ggml_type type, const char * name, 
                    void (*quant_ref)(const float *, void *, int64_t),
                    void (*dequant_ref)(const void *, float *, int64_t)) {
    const int d = 128;
    printf("Testing %s quantization on SYCL...\n", name);

    struct ggml_init_params params_sycl = { 2 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx_sycl = ggml_init(params_sycl);

    // 1. Create tensors in context
    struct ggml_tensor * input   = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_F32, d);
    struct ggml_tensor * output  = ggml_new_tensor_1d(ctx_sycl, type, d);
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_I32, 1);
    
    // 2. Build graph (creates views)
    struct ggml_tensor * view = ggml_set_rows(ctx_sycl, output, input, indices);

    // 3. Allocate tensors on backend
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_sycl, backend);

    // 4. Set input data
    std::vector<float> host_input(d);
    for (int i = 0; i < d; i++) {
        host_input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
    }
    ggml_backend_tensor_set(input, host_input.data(), 0, d * sizeof(float));

    int32_t host_indices[1] = { 0 };
    ggml_backend_tensor_set(indices, host_indices, 0, sizeof(int32_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx_sycl);
    ggml_build_forward_expand(gf, view);

    // 5. Compute
    ggml_backend_graph_compute(backend, gf);

    // 6. Copy back
    std::vector<char> sycl_data(ggml_nbytes(output));
    ggml_backend_tensor_get(output, sycl_data.data(), 0, sycl_data.size());

    // 7. Ref
    std::vector<char> ref_data(ggml_nbytes(output));
    quant_ref(host_input.data(), ref_data.data(), d);

    // 8. Compare
    std::vector<float> sycl_float(d);
    std::vector<float> ref_float(d);

    dequant_ref(sycl_data.data(), sycl_float.data(), d);
    dequant_ref(ref_data.data(), ref_float.data(), d);

    float mse = 0;
    float cosv = 0;
    float ni = 0;
    float no = 0;
    for (int i = 0; i < d; i++) {
        float s = sycl_float[i];
        float r = ref_float[i];
        mse += (s - r) * (s - r);
        cosv += s * r;
        ni += r * r;
        no += s * s;
    }

    float cosine = cosv / (sqrtf(ni) * sqrtf(no));
    printf("  MSE: %.8f, Cosine: %.6f\n", mse / d, cosine);
    bool pass = !(std::isnan(mse) || std::isnan(cosine)) && (mse / d <= 1e-5 && cosine >= 0.99);
    if (!pass) {
        printf("  FAILED\n");
        printf("  First 5 values: SYCL vs REF:\n");
        for (int i = 0; i < std::min(5, d); i++) {
            printf("    [%d] SYCL: %f, REF: %f\n", i, sycl_float[i], ref_float[i]);
        }
    } else {
        printf("  PASSED\n");
    }

    ggml_free(ctx_sycl);
    ggml_backend_buffer_free(buffer);
    return pass;
}

static bool run_weight_test(ggml_backend_t backend, ggml_type type, const char * name,
                           void (*quant_ref)(const float *, void *, int64_t),
                           void (*dequant_ref)(const void *, float *, int64_t)) {
    const int M = 32; // rows
    const int K = 128; // cols
    printf("Testing %s weight multiplication on SYCL...\n", name);

    struct ggml_init_params params_sycl = { 2 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx_sycl = ggml_init(params_sycl);

    // 1. Create tensors
    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx_sycl, type, K, M);
    struct ggml_tensor * input   = ggml_new_tensor_1d(ctx_sycl, GGML_TYPE_F32, K);
    struct ggml_tensor * result  = ggml_mul_mat(ctx_sycl, weights, input);

    // 2. Allocate
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_sycl, backend);

    // 3. Set data
    std::vector<float> host_weights(M * K);
    for (int i = 0; i < M * K; i++) host_weights[i] = (float)((i % 7 - 3) * 0.1);
    
    std::vector<char> quantized_weights(ggml_nbytes(weights));
    quant_ref(host_weights.data(), quantized_weights.data(), M * K);
    ggml_backend_tensor_set(weights, quantized_weights.data(), 0, quantized_weights.size());

    std::vector<float> host_input(K);
    for (int i = 0; i < K; i++) host_input[i] = (float)((i % 5 - 2) * 0.2);
    ggml_backend_tensor_set(input, host_input.data(), 0, K * sizeof(float));

    // 4. Compute
    struct ggml_cgraph * gf = ggml_new_graph(ctx_sycl);
    ggml_build_forward_expand(gf, result);
    ggml_backend_graph_compute(backend, gf);

    // 5. Copy back
    std::vector<float> sycl_result(M);
    ggml_backend_tensor_get(result, sycl_result.data(), 0, M * sizeof(float));

    // 6. Compute CPU reference
    // W_orig @ input
    std::vector<float> ref_result(M);
    size_t row_size = ggml_row_size(type, K);
    for (int i = 0; i < M; i++) {
        float sum = 0;
        // TQ weights are already dequantized to W_orig by dequant_ref
        std::vector<float> dequant_weights(K);
        dequant_ref(quantized_weights.data() + i * row_size, dequant_weights.data(), K);
        
        for (int j = 0; j < K; j++) {
            sum += dequant_weights[j] * host_input[j];
        }
        ref_result[i] = sum;
    }

    // 7. Compare
    float mse = 0;
    float cosv = 0, ni = 0, no = 0;
    for (int i = 0; i < M; i++) {
        float s = sycl_result[i], r = ref_result[i];
        mse += (s-r)*(s-r); cosv += s*r; ni += r*r; no += s*s;
    }
    float cosine = cosv / (sqrtf(ni) * sqrtf(no));
    printf("  MSE: %.8f, Cosine: %.6f\n", mse / M, cosine);
    bool pass = !(std::isnan(mse) || std::isnan(cosine)) && (mse / M <= 1e-4 && cosine >= 0.99);
    if (!pass) {
        printf("  FAILED\n");
        printf("  First 5 values: SYCL vs REF:\n");
        for (int i = 0; i < std::min(5, M); i++) {
            printf("    [%d] SYCL: %f, REF: %f\n", i, sycl_result[i], ref_result[i]);
        }
    } else {
        printf("  PASSED\n");
    }

    ggml_free(ctx_sycl);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// Reference sign array for the TQ weight (group_size == 32) WHT rotation.
// Values copied from ggml-turbo-quant.c's TQ3_0_SIGNS / ggml-cuda's TQ_WEIGHT_SIGNS /
// ggml-sycl's TQ_SIGNS — all three must be identical for the rotation to be correct.
static const float kTqSigns32[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
};

// CPU reference for GGML_OP_TURBO_WHT at group_size == 32: sign flip -> butterfly -> normalize.
// Mirrors tq3_0_rht_forward() in ggml-turbo-quant.c and the group_size==32 branch of
// ggml-cuda/turbo-wht.cu (this is the exact case that regressed in the SYCL port: the
// SYCL kernel used to reuse the 128-element KV sign tables and apply signs twice).
static void turbo_wht32_forward_ref(float * x) {
    for (int i = 0; i < 32; i++) x[i] *= kTqSigns32[i];
    for (int h = 1; h < 32; h <<= 1) {
        for (int i = 0; i < 32; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int i = 0; i < 32; i++) x[i] *= inv_sqrt32;
}

// Regression test for the group_size==32 GGML_OP_TURBO_WHT sign-table bug: builds a
// real GGML_OP_TURBO_WHT graph node on the SYCL backend and compares against the CPU
// reference rotation used at TQ3_1S/TQ4_1S quantization time.
static bool run_wht32_test(ggml_backend_t backend) {
    printf("Testing GGML_OP_TURBO_WHT group_size=32 on SYCL...\n");

    struct ggml_init_params params = { 2 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    struct ggml_tensor * out   = ggml_turbo_wht(ctx, input, /*direction=*/0, /*group_size=*/32, /*scale=*/nullptr);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host_input(32);
    for (int i = 0; i < 32; i++) host_input[i] = sinf(i * 0.37f + 0.2f) * 1.5f;
    ggml_backend_tensor_set(input, host_input.data(), 0, 32 * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> sycl_out(32);
    ggml_backend_tensor_get(out, sycl_out.data(), 0, 32 * sizeof(float));

    std::vector<float> ref_out = host_input;
    turbo_wht32_forward_ref(ref_out.data());

    float mse = 0, cosv = 0, ni = 0, no = 0;
    for (int i = 0; i < 32; i++) {
        float s = sycl_out[i], r = ref_out[i];
        mse += (s - r) * (s - r); cosv += s * r; ni += r * r; no += s * s;
    }
    float cosine = cosv / (sqrtf(ni) * sqrtf(no));
    printf("  MSE: %.8f, Cosine: %.6f\n", mse / 32, cosine);
    bool pass = !(std::isnan(mse) || std::isnan(cosine)) && (mse / 32 <= 1e-8 && cosine >= 0.9999f);
    if (!pass) {
        printf("  FAILED\n");
        for (int i = 0; i < 8; i++) {
            printf("    [%d] SYCL: %f, REF: %f\n", i, sycl_out[i], ref_out[i]);
        }
    } else {
        printf("  PASSED\n");
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// Regression test for the flash-attention turbo KV-cache gate: ggml_sycl_flash_attn_ext_supported()
// used to hard-reject any K/V of a TURBO type before even checking kernel availability, even
// though the underlying tile kernel is fully implemented. This builds a real
// GGML_OP_FLASH_ATTN_EXT graph with TURBO3_0 K/V and checks it actually executes and produces
// finite output, instead of aborting or being silently skipped.
static bool run_fattn_turbo_smoke_test(ggml_backend_t backend) {
    printf("Testing Flash-Attention with TURBO3_0 KV cache on SYCL...\n");

    const int D     = 128; // head dim, matches QK_TURBO3 block size
    const int n_kv   = 256; // must be a multiple of FATTN_KQ_STRIDE (256)
    const int n_head = 1;

    struct ggml_init_params params = { 16 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, 1, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO3_0, D, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_TURBO3_0, D, n_kv, n_head, 1);

    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr,
                                                    1.0f / sqrtf((float) D), 0.0f, 0.0f);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host_q(D);
    for (int i = 0; i < D; i++) host_q[i] = sinf(i * 0.1f + 0.3f);
    ggml_backend_tensor_set(q, host_q.data(), 0, D * sizeof(float));

    std::vector<float> host_kv(D * n_kv);
    for (int i = 0; i < D * n_kv; i++) host_kv[i] = sinf(i * 0.05f + 0.7f) * 0.5f;

    std::vector<char> quantized_kv(ggml_nbytes(k));
    quantize_row_turbo3_0_ref(host_kv.data(), quantized_kv.data(), D * n_kv);
    ggml_backend_tensor_set(k, quantized_kv.data(), 0, quantized_kv.size());
    ggml_backend_tensor_set(v, quantized_kv.data(), 0, quantized_kv.size());

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_status status = ggml_backend_graph_compute(backend, gf);

    bool pass = status == GGML_STATUS_SUCCESS;
    if (pass) {
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        for (float val : result) {
            if (std::isnan(val) || std::isinf(val)) { pass = false; break; }
        }
    }
    printf("  graph_compute status: %d, %s\n", (int) status, pass ? "PASSED" : "FAILED");

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

int main() {
    // Structural assertions for TurboQuant block layouts (TDD validation)
    printf("Validating TurboQuant block layouts...\n");
    printf("  block_turbo2_0: %zu bytes\n", sizeof(block_turbo2_0));
    printf("  block_turbo3_0: %zu bytes\n", sizeof(block_turbo3_0));
    printf("  block_turbo4_0: %zu bytes\n", sizeof(block_turbo4_0));
    printf("  block_tq3_1s:   %zu bytes\n", sizeof(block_tq3_1s));
    printf("  block_tq4_1s:   %zu bytes\n", sizeof(block_tq4_1s));

    if (sizeof(block_turbo2_0) != 34) {
        fprintf(stderr, "FAIL: block_turbo2_0 size mismatch (expected 34)\n");
        return 1;
    }
    if (sizeof(block_turbo3_0) != 50) {
        fprintf(stderr, "FAIL: block_turbo3_0 size mismatch (expected 50)\n");
        return 1;
    }
    if (sizeof(block_turbo4_0) != 68) {
        fprintf(stderr, "FAIL: block_turbo4_0 size mismatch (expected 68)\n");
        return 1;
    }
    if (sizeof(block_tq3_1s) != 16) {
        fprintf(stderr, "FAIL: block_tq3_1s size mismatch (expected 16)\n");
        return 1;
    }
    if (sizeof(block_tq4_1s) != 20) {
        fprintf(stderr, "FAIL: block_tq4_1s size mismatch (expected 20)\n");
        return 1;
    }
    printf("  All layouts structurally valid.\n\n");

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize SYCL backend\n");
        return 1;
    }

    bool success = true;
    success &= run_test(backend, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, dequantize_row_turbo3_0);
    success &= run_test(backend, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, dequantize_row_turbo2_0);
    success &= run_test(backend, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, dequantize_row_turbo4_0);

    success &= run_weight_test(backend, GGML_TYPE_TQ3_1S, "TQ3_1S", quantize_row_tq3_1s_ref, dequantize_row_tq3_1s);
    success &= run_weight_test(backend, GGML_TYPE_TQ4_1S, "TQ4_1S", quantize_row_tq4_1s_ref, dequantize_row_tq4_1s);

    success &= run_wht32_test(backend);
    success &= run_fattn_turbo_smoke_test(backend);

    ggml_backend_free(backend);
    return success ? 0 : 1;
}
