#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"
#define GGML_COMMON_DECL_SYCL
#include "../ggml/src/ggml-common.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

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
    void turbo_cpu_fwht_inverse(float * x, int group_size);
}

// TURBO BEGIN - precise-K quant refs for asymmetric mixed-KV parity. K stays high-precision
// (q8_0 or f16) while V is turbo-compressed; these wrap the public ggml-base quantizers so the
// golden helper can quantize K and V by independent types (signature matches the turbo refs).
static void quantize_row_q8_0_ref_wrap(const float * x, void * y, int64_t k) {
    ggml_quantize_chunk(GGML_TYPE_Q8_0, x, y, 0, 1, k, nullptr);
}
static void quantize_row_q4_0_ref_wrap(const float * x, void * y, int64_t k) {
    ggml_quantize_chunk(GGML_TYPE_Q4_0, x, y, 0, 1, k, nullptr);
}
static void quantize_row_f16_ref_wrap(const float * x, void * y, int64_t k) {
    ggml_fp32_to_fp16_row(x, (ggml_fp16_t *) y, k);
}
// TURBO END

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
// ggml-sycl's TQ_SIGNS - all three must be identical for the rotation to be correct.
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

// Round-trip identity check for GGML_OP_TURBO_WHT at the real KV group sizes (128 and 64):
// forward then inverse must reconstruct the input (the +/-1 sign diagonals and (1/N)*H*H are
// mutually inverse). Sign-table-free: does not need the 128/64 sign arrays exposed to the test,
// unlike run_wht32_test which compares against a CPU reference. Catches a broken inverse or a
// group-size/normalization mismatch in the SYCL WHT op.
static bool run_wht_roundtrip_test(ggml_backend_t backend, int gs) {
    printf("Testing GGML_OP_TURBO_WHT fwd->inv round-trip, group_size=%d on SYCL...\n", gs);

    struct ggml_init_params params = { 4 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gs);
    struct ggml_tensor * fwd   = ggml_turbo_wht(ctx, input, /*direction=*/0, /*group_size=*/gs, /*scale=*/nullptr);
    struct ggml_tensor * inv   = ggml_turbo_wht(ctx, fwd,   /*direction=*/1, /*group_size=*/gs, /*scale=*/nullptr);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host_input(gs);
    for (int i = 0; i < gs; i++) host_input[i] = sinf(i * 0.23f + 0.11f) * 2.0f - 0.5f;
    ggml_backend_tensor_set(input, host_input.data(), 0, gs * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, inv);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> sycl_out(gs);
    ggml_backend_tensor_get(inv, sycl_out.data(), 0, gs * sizeof(float));

    float mse = 0, maxabs = 0;
    for (int i = 0; i < gs; i++) {
        float diff = sycl_out[i] - host_input[i];
        mse += diff * diff;
        maxabs = std::max(maxabs, std::fabs(diff));
    }
    mse /= gs;
    bool pass = !std::isnan(mse) && mse <= 1e-8f && maxabs <= 1e-3f;
    printf("  MSE: %.10f, max|diff|: %.8f, %s\n", mse, maxabs, pass ? "PASSED" : "FAILED");
    if (!pass) {
        for (int i = 0; i < 8; i++) printf("    [%d] SYCL: %f, IN: %f\n", i, sycl_out[i], host_input[i]);
    }

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// Value-level flash-attention parity oracle for turbo KV. Unlike run_fattn_turbo_smoke_test
// (which builds FA with a RAW Q and only checks finiteness), this reproduces the real-model
// data flow: the graph forward-WHT-rotates Q via GGML_OP_TURBO_WHT, then runs FA over a
// turbo-quantized K/V cache. The CPU golden computes attention in the SAME rotated domain the
// kernel works in (Qr = fwht(Q); scores against dequant(quant(K)); P.V against dequant(quant(V))),
// so no output inverse-WHT is needed on either side.
//
// This test FAILS on the current code (the kernel ALSO rotates Q inline -> double rotation; and
// the active TILE path reads turbo V as raw half2) and must PASS once the FA path converges on
// the CUDA architecture (graph op = single rotation site, VEC kernel dequants V).
static bool run_fattn_turbo_golden_test_nq(ggml_backend_t backend, ggml_type type_K, ggml_type type_V, const char * name,
                                           void (*quant_ref_K)(const float *, void *, int64_t),
                                           void (*quant_ref_V)(const float *, void *, int64_t), int n_q,
                                           int n_kv = 256, int D = 128, float logit_softcap = 0.0f,
                                           float sink_val = NAN, int n_seq = 1) {
    // n_seq > 1 gives the tensors a fourth dimension, which the kernel turns into
    // sequence = group(0)/ne02 and then offsets K/V/Q by nb03/nb13/nb23 (fattn-vec.hpp:133-137).
    // Reachable by default: kv_unified is false (common/common.h:571), so n_stream = n_seq_max
    // and any -np N gives K->ne[3] = N. Never covered until now, and derived indices under an
    // untested dimension are exactly the shape of the D=256 defects.
    // sink_val not NaN attaches an attention sink: a per-head logit that competes in the softmax
    // denominator without owning a V row, so it pulls probability mass away and shrinks the output.
    // gpt-oss models use them. The vec kernel handles sinks at fattn-vec.hpp:473-501 and no test had
    // ever set one. Authoritative formula from ggml-cpu/ops.cpp:8628-8642 - the sink participates in
    // the running max and adds exp(sink - M) to the denominator, leaving the numerator alone.
    // logit_softcap != 0 selects the use_logit_softcap kernel instantiation, which applies
    // softcap * tanh(score) to every KQ score before the mask is added (fattn-vec.hpp). Half of the
    // compiled vec instances carry that template parameter and, until this parameter existed, none
    // of them had ever been executed by a test. Gemma-2 and Grok set it in production.
    // The kernel refuses softcap outside D in {128,256} (fattn-vec.hpp early-out), so do not pass a
    // non-zero value at any other head dim.
    // head dim. 128 is one QK_TURBO block per row; 256 is two, which is the shape that made a
    // head_dim-256 model lose SYCL flash-attention entirely and fall back to the CPU.
    // The WHT group is fixed at 128 (QK_TURBO3_GROUP), so D=256 is two independent rotations and
    // the CPU golden below needs no special casing - it quantizes the whole D-length row with the
    // same reference quantizer, which walks block by block.
    // n_kv must be a multiple of FATTN_KQ_STRIDE. The default 256 gives ntiles_KQ = 2, which barely
    // engages the split-KV path; pass a deep n_kv to force parallel_blocks > 1 in launch_fattn and
    // exercise flash_attn_combine_results.
    const float scale = 1.0f / sqrtf((float) D);
    printf("Testing FA turbo golden parity: %s (D=%d, n_kv=%d, n_q=%d)...\n", name, D, n_kv, n_q);
    // n_q>1 exercises the vec kernel's cols_per_block>=2 (prefill) path, which only turbo hits
    // (non-turbo quantized prefill routes to the tile kernel). Quantize refs default the WHT group
    // to 128 for a 128-aligned row, matching the head dim.

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, 1, n_seq);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, type_K, D, n_kv, 1, n_seq);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, type_V, D, n_kv, 1, n_seq);

    // Graph: rotate Q with the SAME forward WHT the real model inserts, then flash-attention.
    // group_size 0 = auto, which ggml_turbo_wht resolves to 128 for any 128-aligned ne[0]
    // (ggml.c) - the same value llama-graph.cpp derives. The WHT group is a property of the
    // turbo format, not of head_dim, so D=256 is two independent 128-rotations per row.
    struct ggml_tensor * qr  = ggml_turbo_wht(ctx, q, /*direction=*/0, /*group_size=*/0, /*scale=*/nullptr);
    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, qr, k, v, /*mask=*/nullptr, scale, 0.0f, logit_softcap);

    struct ggml_tensor * sinks = nullptr;
    if (!std::isnan(sink_val)) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);  // one head in this harness
        ggml_flash_attn_ext_add_sinks(out, sinks);
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host_q((size_t) D * n_q * n_seq);
    // Distinct data per sequence, so a kernel that ignored nb03 would mix them and be caught.
    for (int sq = 0; sq < n_seq; sq++)
        for (int c = 0; c < n_q; c++)
            for (int i = 0; i < D; i++)
                host_q[((size_t) sq*n_q + c)*D + i] = sinf(i * 0.1f + 0.3f + c * 0.9f + sq * 2.7f);
    ggml_backend_tensor_set(q, host_q.data(), 0, host_q.size() * sizeof(float));
    if (sinks) { ggml_backend_tensor_set(sinks, &sink_val, 0, sizeof(float)); }

    const size_t kv_elems = (size_t) D * n_kv * n_seq;
    std::vector<float> host_kv(kv_elems);
    // Offset per sequence so sequence 1's K/V is not a copy of sequence 0's.
    for (int sq = 0; sq < n_seq; sq++)
        for (int i = 0; i < D * n_kv; i++)
            host_kv[(size_t) sq*D*n_kv + i] = sinf(i * 0.05f + 0.7f + sq * 1.3f) * 0.5f;

    std::vector<char> quant_k(ggml_nbytes(k)), quant_v(ggml_nbytes(v));
    quant_ref_K(host_kv.data(), quant_k.data(), (int64_t) kv_elems);
    quant_ref_V(host_kv.data(), quant_v.data(), (int64_t) kv_elems);
    ggml_backend_tensor_set(k, quant_k.data(), 0, quant_k.size());
    ggml_backend_tensor_set(v, quant_v.data(), 0, quant_v.size());

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  graph_compute FAILED with status %d\n", (int) status);
        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
        return false;
    }

    std::vector<float> sycl_out(ggml_nelements(out));
    ggml_backend_tensor_get(out, sycl_out.data(), 0, ggml_nbytes(out));

    // --- CPU golden in the rotated domain (K and V dequant by independent types) ---
    const size_t rb_k = ggml_row_size(type_K, D);
    const size_t rb_v = ggml_row_size(type_V, D);
    const ggml_type_traits * tr_k = ggml_get_type_traits(type_K);
    const ggml_type_traits * tr_v = ggml_get_type_traits(type_V);
    auto dequant_row_k = [&](int row, std::vector<float> & dst) {
        tr_k->to_float(quant_k.data() + (size_t) row * rb_k, dst.data(), D);
    };
    auto dequant_row_v = [&](int row, std::vector<float> & dst) {
        tr_v->to_float(quant_v.data() + (size_t) row * rb_v, dst.data(), D);
    };

    // Device-rotated Q read back from the graph WHT node (n_q columns, each its own 128-group).
    std::vector<float> qr_ref((size_t) D * n_q * n_seq);
    ggml_backend_tensor_get(qr, qr_ref.data(), 0, qr_ref.size() * sizeof(float));

    // Pre-dequantize every K/V row of every sequence once. Sequence sq owns rows
    // [sq*n_kv, (sq+1)*n_kv) because the tensor is contiguous in ne[3].
    const int n_rows_all = n_kv * n_seq;
    std::vector<std::vector<float>> k_rows(n_rows_all, std::vector<float>(D)),
                                    v_rows(n_rows_all, std::vector<float>(D));
    for (int j = 0; j < n_rows_all; j++) { dequant_row_k(j, k_rows[j]); dequant_row_v(j, v_rows[j]); }

    // FA output layout is {DV, n_head=1, n_q, n_seq}: sequence sq column c at (sq*n_q + c)*D.
    bool pass = true;
    float worst_cos = 1.0f, worst_relmse = 0.0f;
    std::vector<float> scores(n_kv), ref_out(D);
    for (int cc = 0; cc < n_q * n_seq; cc++) {
        const int sq = cc / n_q;          // sequence index
        const int c  = cc % n_q;          // column within the sequence
        const int kv0 = sq * n_kv;        // first K/V row belonging to this sequence
        const float * qc = qr_ref.data() + (size_t) cc * D;
        for (int j = 0; j < n_kv; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += qc[d] * k_rows[kv0 + j][d];
            // ggml divides the scale by the softcap up front (ggml-cpu/ops.cpp: `scale /=
            // logit_softcap` before the loop, then `s = s*scale` and `s = logit_softcap*tanhf(s)`),
            // so the score is softcap*tanh(dot*scale/softcap), not softcap*tanh(dot*scale). Getting
            // that wrong makes tanh saturate and looks exactly like a kernel defect: cosine stays
            // near 0.99 while magnitudes diverge several-fold.
            scores[j] = logit_softcap != 0.0f
                      ? logit_softcap * tanhf(dot * scale / logit_softcap)
                      : dot * scale;
        }
        float mx = scores[0];
        for (int j = 1; j < n_kv; j++) mx = std::max(mx, scores[j]);
        // The sink joins the running max and contributes exp(sink - M) to the denominator only,
        // never to the numerator - it has no V row (ggml-cpu/ops.cpp:8628-8642).
        if (!std::isnan(sink_val)) mx = std::max(mx, sink_val);
        float sum = 0.0f;
        for (int j = 0; j < n_kv; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        if (!std::isnan(sink_val)) sum += expf(sink_val - mx);
        float inv_sum = 1.0f / sum;
        std::fill(ref_out.begin(), ref_out.end(), 0.0f);
        for (int j = 0; j < n_kv; j++) {
            float p = scores[j] * inv_sum;
            for (int d = 0; d < D; d++) ref_out[d] += p * v_rows[kv0 + j][d];
        }
        const float * sc = sycl_out.data() + (size_t) cc * D;
        float mse = 0, cosv = 0, nr = 0, no = 0;
        for (int d = 0; d < D; d++) {
            float s = sc[d], r = ref_out[d];
            mse += (s - r) * (s - r); cosv += s * r; nr += r * r; no += s * s;
        }
        float cosine = cosv / (sqrtf(nr) * sqrtf(no) + 1e-20f);
        float rel_mse = mse / (nr / D + 1e-20f);
        worst_cos = std::min(worst_cos, cosine);
        worst_relmse = std::max(worst_relmse, rel_mse);
        // The achievable rel-MSE depends on which device path this shape takes.
        //  - f16 TILE, taken by ggml_sycl_flash_attn_ext_turbo_prefill (symmetric turbo above 2 Q
        //    columns, asymmetric at 16 or more): carries f16 KV precision, ~2^-9 relative.
        //  - turbo VEC with a quantized NON-turbo K (i.e. q8_0): fattn-vec.hpp:124 sets
        //    Q_q8_1 = type_K != F16 && !K_is_turbo, so Q itself is quantized to q8_1 for the KQ
        //    product. That costs a few percent per element and is upstream's design, not a turbo
        //    defect. f16-K and turbo-K both skip it, which is why only q8_0-K needs the wider bound.
        //    Cosine still has to clear 0.999, which is what actually catches structural breakage.
        const bool k_is_turbo = type_K == GGML_TYPE_TURBO2_0 || type_K == GGML_TYPE_TURBO3_0 ||
                                type_K == GGML_TYPE_TURBO4_0;
        const bool v_is_turbo = type_V == GGML_TYPE_TURBO2_0 || type_V == GGML_TYPE_TURBO3_0 ||
                                type_V == GGML_TYPE_TURBO4_0;
        // Mirrors ggml_sycl_flash_attn_ext_turbo_prefill: turbo V, 16 columns, and mixed turbo widths
        // only for the curated turbo4-K pairs. Keep in lockstep with the allowlist in fattn.cpp - a
        // mismatch here selects the TIGHTER bound, so it surfaces as a failure rather than a false
        // pass. Do not "fix" such a failure by widening the tolerance.
        const bool mixed_ok   = type_K == GGML_TYPE_TURBO4_0 &&
                                (type_V == GGML_TYPE_TURBO3_0 || type_V == GGML_TYPE_TURBO2_0);
        // Mirrors the shim: turbo K at D>=256 stays on VEC (the f16 TILE drifts there).
        const bool uses_tile  = v_is_turbo && !(k_is_turbo && type_K != type_V && !mixed_ok)
                                && !(k_is_turbo && D >= 256) && n_q >= 16;
        const bool q_is_q8_1  = !uses_tile && type_K != GGML_TYPE_F16 && !k_is_turbo;
        // A q8_0 V also takes a different dequant path in the vec kernel than a turbo V, and the
        // AOT (bmg-g21) compiler schedules it with slightly wider error than the JIT: the same
        // turbo-K x q8_0-V cases measure 0.0 under build-sync and 1.2-1.4e-3 under build-perf,
        // with cosine holding at 0.999995. That is arithmetic scheduling, not a structural
        // difference, so the bound has to admit it or the gate only passes on one build flavor.
        const bool v_is_q8_0 = type_V == GGML_TYPE_Q8_0;
        const float relmse_tol = q_is_q8_1 ? 8e-2f
                               : (n_q > 2  ? 6e-3f
                               : (v_is_q8_0 ? 2e-3f : 1e-3f));
        bool col_ok = !(std::isnan(cosine) || std::isnan(rel_mse)) && cosine >= 0.999f && rel_mse <= relmse_tol;
        if (!col_ok) {
            pass = false;
            printf("  col %d FAILED cosine=%.6f rel-MSE=%.8f\n", c, cosine, rel_mse);
            for (int d = 0; d < 6; d++) printf("    [%d] SYCL: %f, GOLDEN: %f\n", d, sc[d], ref_out[d]);
        }
    }
    printf("  worst cosine: %.6f, worst rel-MSE: %.8f, %s\n", worst_cos, worst_relmse, pass ? "PASSED" : "FAILED");

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// BISECT TEST A: device inverse WHT (GGML_OP_TURBO_WHT direction=1) value parity vs CPU.
// The real model applies this inverse to the FA output (llama-graph.cpp build_attn_mha) to undo
// the V-side WHT rotation. The existing round-trip test only checks fwd(inv(x))==x; it can't catch
// an inverse that is self-consistent but produces the WRONG un-rotated values. This compares the
// device inverse of an arbitrary vector against the CPU reference turbo_cpu_fwht_inverse.
static bool run_wht_inverse_value_test(ggml_backend_t backend, int gs) {
    const int N = 4; // independent columns/groups
    printf("Testing GGML_OP_TURBO_WHT inverse value parity, gs=%d...\n", gs);

    struct ggml_init_params params = { 8 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gs, N);
    struct ggml_tensor * inv   = ggml_turbo_wht(ctx, input, /*direction=*/1, /*group_size=*/gs, /*scale=*/nullptr);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host(gs * N);
    for (int i = 0; i < gs * N; i++) host[i] = sinf(i * 0.037f + 0.11f);
    ggml_backend_tensor_set(input, host.data(), 0, host.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, inv);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> dev(gs * N);
    ggml_backend_tensor_get(inv, dev.data(), 0, dev.size() * sizeof(float));

    std::vector<float> ref = host;
    for (int c = 0; c < N; c++) turbo_cpu_fwht_inverse(ref.data() + (size_t) c * gs, gs);

    float maxdiff = 0.0f;
    for (int i = 0; i < gs * N; i++) maxdiff = std::max(maxdiff, fabsf(dev[i] - ref[i]));
    bool pass = !std::isnan(maxdiff) && maxdiff < 1e-4f;
    printf("  max|dev-cpu| = %.8f  %s\n", maxdiff, pass ? "PASSED" : "FAILED");
    if (!pass) for (int i = 0; i < 6; i++) printf("    [%d] dev=%f cpu=%f\n", i, dev[i], ref[i]);

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// BISECT TEST C: device SET_ROWS turbo quantize with MULTIPLE 128-groups per row and MULTIPLE rows.
// The passing run_test only exercises a single 128-group / single row. The real KV cache stores
// ne00 = n_head_kv*head_dim (folded heads => several 128-groups per row, e.g. 8 for Qwen3) and
// writes n_tokens rows via indices. This reproduces that layout and compares device set_rows bytes
// against the CPU quantize_row_turbo*_ref per row (each row quantized independently, group=128).
static bool run_set_rows_multi_test(ggml_backend_t backend, ggml_type type, const char * name,
                                    void (*quant_ref)(const float *, void *, int64_t),
                                    void (*dequant_ref)(const void *, float *, int64_t),
                                    int D, int n_rows) {
    printf("Testing %s set_rows multi-group/multi-row (ne00=%d, rows=%d)...\n", name, D, n_rows);

    struct ggml_init_params params = { 32 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);
    struct ggml_tensor * output  = ggml_new_tensor_2d(ctx, type, D, n_rows);
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
    struct ggml_tensor * view    = ggml_set_rows(ctx, output, input, indices);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> host(D * n_rows);
    for (int i = 0; i < D * n_rows; i++) host[i] = sinf(i * 0.05f + 0.7f) * 0.5f;
    ggml_backend_tensor_set(input, host.data(), 0, host.size() * sizeof(float));

    std::vector<int32_t> idx(n_rows);
    for (int i = 0; i < n_rows; i++) idx[i] = i;
    ggml_backend_tensor_set(indices, idx.data(), 0, n_rows * sizeof(int32_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, view);
    ggml_backend_graph_compute(backend, gf);

    std::vector<char> dev(ggml_nbytes(output));
    ggml_backend_tensor_get(output, dev.data(), 0, dev.size());

    const size_t row_bytes = ggml_nbytes(output) / n_rows;
    std::vector<char> ref(ggml_nbytes(output));
    for (int r = 0; r < n_rows; r++) quant_ref(host.data() + (size_t) r * D, ref.data() + (size_t) r * row_bytes, D);

    bool pass = true;
    float worst = 1.0f;
    for (int r = 0; r < n_rows; r++) {
        std::vector<float> ds(D), rs(D);
        dequant_ref(dev.data() + (size_t) r * row_bytes, ds.data(), D);
        dequant_ref(ref.data() + (size_t) r * row_bytes, rs.data(), D);
        float cv = 0, ni = 0, no = 0;
        for (int i = 0; i < D; i++) { cv += ds[i] * rs[i]; ni += rs[i] * rs[i]; no += ds[i] * ds[i]; }
        float cos = cv / (sqrtf(ni) * sqrtf(no) + 1e-20f);
        worst = std::min(worst, cos);
        if (std::isnan(cos) || cos < 0.999f) {
            pass = false;
            printf("  row %d cosine=%.6f FAILED\n", r, cos);
            for (int i = 0; i < 4; i++) printf("    [%d] dev=%f ref=%f\n", i, ds[i], rs[i]);
        }
    }
    printf("  worst cosine=%.6f  %s\n", worst, pass ? "PASSED" : "FAILED");

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return pass;
}

// BISECT TEST D: faithful DECODE reproduction - GQA (n_head > n_head_kv) + a PADDING MASK where the
// KV cache is allocated to n_kv rows but only the first `seq_len` are valid; rows [seq_len, n_kv) are
// padding whose K/V is uninitialized garbage and MUST be excluded via a -inf mask. This is exactly
// the real-model generation path ("1111" degenerate output) that NO existing test exercises: the FA
// golden has no mask, no GQA and no padding; q8_0 has mask+GQA but nthreads_KQ != 1 (turbo forces
// nthreads_KQ=1). If the vec kernel mis-applies the mask to padding rows under nthreads_KQ=1, the
// garbage padding K/V poisons the softmax and the output diverges from the CPU golden.
static bool run_fattn_turbo_decode_mask_gqa(ggml_backend_t backend, ggml_type type_K, ggml_type type_V, const char * name,
                                            void (*quant_ref_K)(const float *, void *, int64_t),
                                            void (*quant_ref_V)(const float *, void *, int64_t)) {
    const int D          = 128;
    const int n_kv       = 256;   // allocated rows (multiple of FATTN_KQ_STRIDE)
    const int seq_len    = 100;   // valid rows; [seq_len, n_kv) are padding -> masked out
    const int n_head_kv  = 2;
    const int gqa_ratio  = 4;
    const int n_head     = n_head_kv * gqa_ratio; // 8 Q heads
    const int n_tokens   = 1;     // decode
    const float scale    = 1.0f / sqrtf((float) D);
    printf("Testing FA turbo DECODE+GQA+padding-mask: %s (n_head=%d/%d, n_kv=%d, valid=%d)...\n",
           name, n_head, n_head_kv, n_kv, seq_len);

    struct ggml_init_params params = { 64 * 1024 * 1024, NULL, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_tokens, n_head, 1);
    struct ggml_tensor * k    = ggml_new_tensor_4d(ctx, type_K, D, n_kv, n_head_kv, 1);
    struct ggml_tensor * v    = ggml_new_tensor_4d(ctx, type_V, D, n_kv, n_head_kv, 1);
    struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tokens, 1, 1);

    struct ggml_tensor * qr  = ggml_turbo_wht(ctx, q, /*direction=*/0, /*group_size=*/0, /*scale=*/nullptr);
    struct ggml_tensor * out = ggml_flash_attn_ext(ctx, qr, k, v, mask, scale, 0.0f, 0.0f);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Q: one distinct vector per head.
    std::vector<float> host_q(D * n_head);
    for (int h = 0; h < n_head; h++)
        for (int i = 0; i < D; i++) host_q[h*D + i] = sinf(i * 0.1f + 0.3f + h * 0.7f);
    ggml_backend_tensor_set(q, host_q.data(), 0, host_q.size() * sizeof(float));

    // K/V per kv-head: valid rows [0,seq_len) get smooth data; padding rows [seq_len,n_kv) get LARGE
    // garbage (simulates uninitialized cache) so a mask failure produces a large, obvious divergence.
    std::vector<float> host_k(D * n_kv * n_head_kv), host_v(D * n_kv * n_head_kv);
    for (int hk = 0; hk < n_head_kv; hk++) {
        for (int j = 0; j < n_kv; j++) {
            for (int i = 0; i < D; i++) {
                const size_t idx = ((size_t) hk * n_kv + j) * D + i;
                if (j < seq_len) {
                    host_k[idx] = sinf(i * 0.05f + j * 0.02f + hk * 0.5f + 0.7f) * 0.5f;
                    host_v[idx] = cosf(i * 0.04f + j * 0.03f + hk * 0.3f + 0.2f) * 0.5f;
                } else {
                    host_k[idx] = 37.0f * sinf(i * 0.9f + j * 1.3f + 3.0f); // garbage
                    host_v[idx] = 51.0f * cosf(i * 0.7f + j * 1.1f + 1.0f); // garbage
                }
            }
        }
    }

    std::vector<char> qk(ggml_nbytes(k)), qv(ggml_nbytes(v));
    // quantize per kv-head row-block (each row is D elements, one 128-group)
    const size_t rb_k = ggml_row_size(type_K, D);
    const size_t rb_v = ggml_row_size(type_V, D);
    for (int r = 0; r < n_kv * n_head_kv; r++) {
        quant_ref_K(host_k.data() + (size_t) r * D, qk.data() + (size_t) r * rb_k, D);
        quant_ref_V(host_v.data() + (size_t) r * D, qv.data() + (size_t) r * rb_v, D);
    }
    ggml_backend_tensor_set(k, qk.data(), 0, qk.size());
    ggml_backend_tensor_set(v, qv.data(), 0, qv.size());

    // Mask: 0 for valid rows, -inf for padding rows (same for the single decode token).
    std::vector<uint16_t> host_mask(n_kv);
    const uint16_t f16_zero = 0x0000;      // +0.0 in half
    const uint16_t f16_ninf = 0xFC00;      // -inf in half
    for (int j = 0; j < n_kv; j++) host_mask[j] = (j < seq_len) ? f16_zero : f16_ninf;
    ggml_backend_tensor_set(mask, host_mask.data(), 0, host_mask.size() * sizeof(uint16_t));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("  graph_compute FAILED with status %d\n", (int) status);
        ggml_free(ctx); ggml_backend_buffer_free(buffer); return false;
    }

    std::vector<float> sycl_out(ggml_nelements(out));
    ggml_backend_tensor_get(out, sycl_out.data(), 0, ggml_nbytes(out));

    // Device-rotated Q read back (n_head columns, each its own 128-group).
    std::vector<float> qr_ref(D * n_head);
    ggml_backend_tensor_get(qr, qr_ref.data(), 0, qr_ref.size() * sizeof(float));

    // Dequant K/V rows (rotated domain, same as kernel; K and V by independent types).
    const ggml_type_traits * tr_k = ggml_get_type_traits(type_K);
    const ggml_type_traits * tr_v = ggml_get_type_traits(type_V);
    std::vector<std::vector<float>> k_rows(n_kv * n_head_kv, std::vector<float>(D));
    std::vector<std::vector<float>> v_rows(n_kv * n_head_kv, std::vector<float>(D));
    for (int r = 0; r < n_kv * n_head_kv; r++) {
        tr_k->to_float(qk.data() + (size_t) r * rb_k, k_rows[r].data(), D);
        tr_v->to_float(qv.data() + (size_t) r * rb_v, v_rows[r].data(), D);
    }

    // FA output layout {D, n_head, n_tokens=1}: head h at sycl_out[h*D + d].
    bool pass = true;
    float worst_cos = 1.0f, worst_relmse = 0.0f;
    std::vector<float> scores(seq_len), ref_out(D);
    for (int h = 0; h < n_head; h++) {
        const int hk = h / gqa_ratio;
        const float * qh = qr_ref.data() + (size_t) h * D;
        for (int j = 0; j < seq_len; j++) {           // ONLY valid rows (mask excludes the rest)
            const float * krow = k_rows[(size_t) hk * n_kv + j].data();
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += qh[d] * krow[d];
            scores[j] = dot * scale;
        }
        float mx = scores[0];
        for (int j = 1; j < seq_len; j++) mx = std::max(mx, scores[j]);
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        float inv_sum = 1.0f / sum;
        std::fill(ref_out.begin(), ref_out.end(), 0.0f);
        for (int j = 0; j < seq_len; j++) {
            const float * vrow = v_rows[(size_t) hk * n_kv + j].data();
            float p = scores[j] * inv_sum;
            for (int d = 0; d < D; d++) ref_out[d] += p * vrow[d];
        }
        const float * sc = sycl_out.data() + (size_t) h * D;
        float mse = 0, cosv = 0, nr = 0, no = 0;
        for (int d = 0; d < D; d++) {
            float s = sc[d], r = ref_out[d];
            mse += (s - r) * (s - r); cosv += s * r; nr += r * r; no += s * s;
        }
        float cosine = cosv / (sqrtf(nr) * sqrtf(no) + 1e-20f);
        float rel_mse = mse / (nr / D + 1e-20f);
        worst_cos = std::min(worst_cos, cosine);
        worst_relmse = std::max(worst_relmse, rel_mse);
        bool ok = !(std::isnan(cosine) || std::isnan(rel_mse)) && cosine >= 0.999f && rel_mse <= 1e-3f;
        if (!ok) {
            pass = false;
            printf("  head %d (kv %d) FAILED cosine=%.6f rel-MSE=%.8f\n", h, hk, cosine, rel_mse);
            for (int d = 0; d < 6; d++) printf("    [%d] SYCL: %f, GOLDEN: %f\n", d, sc[d], ref_out[d]);
        }
    }
    printf("  worst cosine: %.6f, worst rel-MSE: %.8f, %s\n", worst_cos, worst_relmse, pass ? "PASSED" : "FAILED");

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
    success &= run_wht_roundtrip_test(backend, 128);
    success &= run_wht_roundtrip_test(backend, 64);

    // BISECT: the two turbo-only real-model pieces the FA golden never exercises.
    printf("\n=== Bisect: inverse WHT value + multi-group set_rows ===\n");
    success &= run_wht_inverse_value_test(backend, 128);
    success &= run_wht_inverse_value_test(backend, 64);
    // ne00=1024 == 8 groups (Qwen3 n_head_kv*head_dim); 4 token rows.
    success &= run_set_rows_multi_test(backend, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, dequantize_row_turbo3_0, 1024, 4);
    success &= run_set_rows_multi_test(backend, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, dequantize_row_turbo2_0, 1024, 4);
    success &= run_set_rows_multi_test(backend, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, dequantize_row_turbo4_0, 1024, 4);

    // Value-level FA parity (the real oracle). Runs BEFORE the smoke test because the backend
    // std::exit(1)s on any sycl::exception (ggml-sycl.cpp:4390), so the first failing FA call
    // kills the process. EXPECTED on pre-convergence code: the turbo TILE path is selected and
    // faults (OUT_OF_RESOURCES on Arc) / double-rotates; must PASS once turbo routes to VEC.
    printf("\n=== FA turbo golden parity (value-level) ===\n");
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, quantize_row_turbo3_0_ref, 1);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, quantize_row_turbo2_0_ref, 1);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, quantize_row_turbo4_0_ref, 1);
    // n_q>1 exercises the vec cols_per_block>=2 path that only turbo uses. The prefill shim declines
    // below 16 columns, so n_q=8 is VEC and n_q=16 is the dequant-to-f16 + f16 TILE path; symmetric
    // turbo needs both, same as the asymmetric pairs below.
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, quantize_row_turbo3_0_ref, 8);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, quantize_row_turbo2_0_ref, 8);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, quantize_row_turbo3_0_ref, 16);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, quantize_row_turbo2_0_ref, 16);
    success &= run_fattn_turbo_golden_test_nq(backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, quantize_row_turbo4_0_ref, 16);

    // DECODE reproduction: GQA + padding mask (the real "1111" generation path).
    printf("\n=== FA turbo DECODE + GQA + padding-mask (real-model repro) ===\n");
    success &= run_fattn_turbo_decode_mask_gqa(backend, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0", quantize_row_turbo3_0_ref, quantize_row_turbo3_0_ref);
    success &= run_fattn_turbo_decode_mask_gqa(backend, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0, "TURBO2_0", quantize_row_turbo2_0_ref, quantize_row_turbo2_0_ref);
    success &= run_fattn_turbo_decode_mask_gqa(backend, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, "TURBO4_0", quantize_row_turbo4_0_ref, quantize_row_turbo4_0_ref);

    // TURBO BEGIN - ASYMMETRIC mixed-KV golden parity: precise K (q8_0 or f16) + turbo V.
    // "V is free, K is everything": K stays high-precision, V is turbo-compressed. Pure kernel-vs-CPU
    // parity oracle (both sides read the same graph-rotated Q and dequant K/V by independent types),
    // so it validates the vec dispatch table for {q8_0,f16} x {turbo2,turbo3,turbo4} at head_dim=128.
    // Quant error cancels between kernel and CPU golden, so even 2-bit turbo2-V holds cosine>=0.999.
    struct asym_cfg {
        ggml_type    type_K;
        ggml_type    type_V;
        const char * name;
        void (*qref_K)(const float *, void *, int64_t);
        void (*qref_V)(const float *, void *, int64_t);
    };
    const asym_cfg asym_cfgs[] = {
        { GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0, "q8_0xTURBO2_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo2_0_ref },
        { GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
        { GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0, "q8_0xTURBO4_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo4_0_ref },
        { GGML_TYPE_F16,  GGML_TYPE_TURBO3_0, "f16xTURBO3_0",  quantize_row_f16_ref_wrap,  quantize_row_turbo3_0_ref },
        { GGML_TYPE_F16,  GGML_TYPE_TURBO2_0, "f16xTURBO2_0",  quantize_row_f16_ref_wrap,  quantize_row_turbo2_0_ref },
        { GGML_TYPE_F16,  GGML_TYPE_TURBO4_0, "f16xTURBO4_0",  quantize_row_f16_ref_wrap,  quantize_row_turbo4_0_ref },
    };
    printf("\n=== FA turbo golden parity, ASYMMETRIC precise-K x turbo-V (decode n_q=1) ===\n");
    for (const auto & c : asym_cfgs)
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1);
    printf("\n=== FA turbo DECODE+GQA+padding-mask, ASYMMETRIC precise-K x turbo-V ===\n");
    for (const auto & c : asym_cfgs)
        success &= run_fattn_turbo_decode_mask_gqa(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V);

    // The shim switches kernels at 16 Q columns: below that it declines and turbo VEC runs (batched
    // decode must not drag the whole cache through a dequant), at 16 and above it dequantizes to
    // f16 scratch and the f16 TILE runs. Cover both sides - two different kernels that must reach
    // the same answer.
    printf("\n=== FA turbo golden parity, ASYMMETRIC precise-K x turbo-V (n_q=8, below threshold -> VEC) ===\n");
    for (const auto & c : asym_cfgs)
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 8);
    printf("\n=== FA turbo golden parity, ASYMMETRIC precise-K x turbo-V (n_q=16, at threshold -> f16 TILE) ===\n");
    for (const auto & c : asym_cfgs)
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 16);

    // Reverse asymmetry (turbo K + q8_0 V) is curated in the vec table but is deliberately REFUSED
    // by the prefill shim (it requires a turbo V), so both column counts below stay on VEC. These
    // cases had no coverage at all before; they pin the path the shim declines to take.
    const asym_cfg rev_cfgs[] = {
        { GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0, "TURBO2_0xq8_0", quantize_row_turbo2_0_ref, quantize_row_q8_0_ref_wrap },
        { GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0, "TURBO3_0xq8_0", quantize_row_turbo3_0_ref, quantize_row_q8_0_ref_wrap },
        { GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0, "TURBO4_0xq8_0", quantize_row_turbo4_0_ref, quantize_row_q8_0_ref_wrap },
    };
    printf("\n=== FA turbo golden parity, ASYMMETRIC turbo-K x q8_0-V (VEC only, n_q=1 and n_q=8) ===\n");
    for (const auto & c : rev_cfgs) {
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1);
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 8);
    }
    // Mixed-width turbo: turbo4 K (the quality knee on K) plus a cheaper turbo V. Both sides are
    // turbo here, so unlike the precise-K pairs above the graph rotates Q forward AND inverse-rotates
    // the output. The vec kernel derives nthreads_KQ/vec_dot_KQ from type_K alone and
    // nthreads_V/V_rows_per_thread/dequantize_V from type_V alone, and turbo2/3/4 all resolve to the
    // same thread counts - so this is the proven turbo4-K half bolted to the proven turbo3/turbo2-V
    // half. These cases exist to prove that bolt holds.
    const asym_cfg mixed_cfgs[] = {
        { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, "TURBO4_0xTURBO3_0", quantize_row_turbo4_0_ref, quantize_row_turbo3_0_ref },
        { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, "TURBO4_0xTURBO2_0", quantize_row_turbo4_0_ref, quantize_row_turbo2_0_ref },
    };
    printf("\n=== FA turbo golden parity, MIXED-WIDTH turbo4-K x cheaper-turbo-V ===\n");
    for (const auto & c : mixed_cfgs) {
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1);
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 8);
        success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 16);
    }
    printf("\n=== FA turbo DECODE+GQA+padding-mask, MIXED-WIDTH turbo4-K x cheaper-turbo-V ===\n");
    for (const auto & c : mixed_cfgs)
        success &= run_fattn_turbo_decode_mask_gqa(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V);

    // head_dim 256. Two QK_TURBO blocks per row. Previously uncovered: with no turbo vec instance
    // the router returned NONE, ggml_sycl_flash_attn_ext_supported reported false, and ggml ran the
    // whole attention op on the CPU with no diagnostic (ADR-0008). Enabled once the combine stopped
    // staging every warp's partial vector in shared memory at once.
    printf("\n=== FA turbo golden parity, head_dim 256 ===\n");
    {
        const asym_cfg d256_cfgs[] = {
            { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0",     quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
            { GGML_TYPE_F16,      GGML_TYPE_TURBO3_0, "f16xTURBO3_0",      quantize_row_f16_ref_wrap,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0",          quantize_row_turbo3_0_ref,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, "TURBO4_0xTURBO3_0", quantize_row_turbo4_0_ref,  quantize_row_turbo3_0_ref },
            // Non-turbo at D=256 too: the host/device nthreads disagreement fixed alongside this
            // was never turbo-specific, so f16 and q8_0 KV on the vec path were equally affected.
            // These cases exist to keep that fixed rather than to exercise turbo.
            { GGML_TYPE_F16,      GGML_TYPE_F16,      "f16xf16",           quantize_row_f16_ref_wrap,  quantize_row_f16_ref_wrap },
            { GGML_TYPE_Q8_0,     GGML_TYPE_Q8_0,     "q8_0xq8_0",         quantize_row_q8_0_ref_wrap, quantize_row_q8_0_ref_wrap },
            { GGML_TYPE_Q4_0,     GGML_TYPE_Q4_0,     "q4_0xq4_0",         quantize_row_q4_0_ref_wrap, quantize_row_q4_0_ref_wrap },
        };
        for (const auto & c : d256_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1,  256, 256);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 16, 256, 256);
        }
    }

    // head_dim 512, non-turbo (FATTN_VEC_CASES_TURBO_D stops at 256). These measured cosine 0.039
    // and 0.047 - garbage - until nthreads was capped at 256. The cause was the work-group size:
    // nthreads was max(128, D) = 512 there, and the maximum a device grants a kernel shrinks with
    // its register pressure, which at D=512 is already near the per-lane budget from Q_reg plus
    // VKQ. Shared memory was never the problem (16 KB at D=512, far under the ceiling) and neither
    // was KQ indexing; both were measured and eliminated first.
    // Not turbo-specific: FATTN_VEC_CASES_ALL_D has always emitted these instances and the router
    // reaches them, so every head_dim-512 model on the SYCL vec path was affected.
    printf("\n=== FA golden parity, head_dim 512 (non-turbo) ===\n");
    {
        const asym_cfg d512_cfgs[] = {
            { GGML_TYPE_F16,  GGML_TYPE_F16,  "f16xf16",   quantize_row_f16_ref_wrap,  quantize_row_f16_ref_wrap },
            { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "q8_0xq8_0", quantize_row_q8_0_ref_wrap, quantize_row_q8_0_ref_wrap },
        };
        for (const auto & c : d512_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1, 256, 512);
        }
    }

    // logit_softcap. Half of every compiled vec instance carries use_logit_softcap=true as a template
    // parameter, and before this block not one of them had ever been executed by a test - the kernel
    // implements it in 21 places and the golden exercised none. Gemma-2 and Grok set it in production.
    // Both D=128 and D=256 are covered because fattn-vec.hpp gates softcap on exactly that pair.
    printf("\n=== FA turbo golden parity, logit_softcap (previously zero coverage) ===\n");
    {
        const asym_cfg softcap_cfgs[] = {
            { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0",      quantize_row_turbo3_0_ref,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_F16,      GGML_TYPE_F16,      "f16xf16",       quantize_row_f16_ref_wrap,  quantize_row_f16_ref_wrap },
        };
        for (const auto & c : softcap_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1,  256, 128, 30.0f);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 16, 256, 128, 30.0f);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1,  256, 256, 30.0f);
        }
    }

    // Attention sinks. The vec kernel handles them at fattn-vec.hpp:473-501 and no test had ever set
    // one; gpt-oss models use them. Two sink magnitudes on purpose: one below the score range so it
    // barely shifts the denominator, and one above it so the sink wins the running max and forces
    // the rescale branch (ggml-cpu/ops.cpp:8633-8637), which is the half that would otherwise never
    // execute.
    printf("\n=== FA turbo golden parity, attention sinks (previously zero coverage) ===\n");
    {
        const asym_cfg sink_cfgs[] = {
            { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0",      quantize_row_turbo3_0_ref,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_F16,      GGML_TYPE_F16,      "f16xf16",       quantize_row_f16_ref_wrap,  quantize_row_f16_ref_wrap },
        };
        for (const auto & c : sink_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1, 256, 128, 0.0f, -2.0f);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1, 256, 128, 0.0f,  8.0f);
        }
    }

    // Multi-sequence, ne[3] = 2. The kernel derives sequence = group(0)/ne02 and offsets Q/K/V by
    // nb03/nb13/nb23 from it (fattn-vec.hpp:133-137); nothing had ever exercised those. It is the
    // DEFAULT shape, not an exotic one: kv_unified is false (common/common.h:571), so n_stream =
    // n_seq_max and any -np N gives K->ne[3] = N. Each sequence gets different Q and K/V data, so a
    // kernel that dropped the nb03/nb13 offsets would mix them and fail rather than silently agree.
    printf("\n=== FA turbo golden parity, multi-sequence ne[3]=2 (previously zero coverage) ===\n");
    {
        const asym_cfg seq_cfgs[] = {
            { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0",      quantize_row_turbo3_0_ref,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_F16,      GGML_TYPE_F16,      "f16xf16",       quantize_row_f16_ref_wrap,  quantize_row_f16_ref_wrap },
        };
        for (const auto & c : seq_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1, 256, 128, 0.0f, NAN, 2);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 8, 256, 128, 0.0f, NAN, 2);
        }
    }

    // Deep context. Every case above runs at n_kv=256, i.e. ntiles_KQ=2, so launch_fattn almost
    // never picks parallel_blocks > 1 and flash_attn_combine_results stays largely unexercised.
    // n_kv=8192 gives ntiles_KQ=64 for the vec kernel (nbatch_fa = D = 128), which is where split-KV
    // decode actually lives in a long-context run.
    printf("\n=== FA turbo golden parity, DEEP CONTEXT n_kv=8192 (split-KV / combine path) ===\n");
    {
        const asym_cfg deep_cfgs[] = {
            { GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, "TURBO3_0",     quantize_row_turbo3_0_ref, quantize_row_turbo3_0_ref },
            { GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, "q8_0xTURBO3_0", quantize_row_q8_0_ref_wrap, quantize_row_turbo3_0_ref },
            { GGML_TYPE_F16,      GGML_TYPE_TURBO3_0, "f16xTURBO3_0",  quantize_row_f16_ref_wrap,  quantize_row_turbo3_0_ref },
            { GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0, "TURBO4_0xTURBO3_0", quantize_row_turbo4_0_ref, quantize_row_turbo3_0_ref },
        };
        for (const auto & c : deep_cfgs) {
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 1,  8192);
            success &= run_fattn_turbo_golden_test_nq(backend, c.type_K, c.type_V, c.name, c.qref_K, c.qref_V, 16, 8192);
        }
    }
    // TURBO END

    success &= run_fattn_turbo_smoke_test(backend);

    ggml_backend_free(backend);
    return success ? 0 : 1;
}
