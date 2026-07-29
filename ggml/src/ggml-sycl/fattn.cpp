//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn.hpp"


#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

// Turbo head dims. The old restriction to {64,128} blamed register pressure from nthreads_KQ=1,
// which stopped being true when the cooperative-register rework landed (SYCL uses 128/cpy_nb).
// The real ceiling was shared local memory in the final combine: it staged nwarps*V_cols_per_iter
// partial vectors of length D at once, and since nthreads = max(128, D) the warp count grows with D
// as well, reaching 16*8*256 = 128 KB at D=256 - twice the Xe2 per-work-group limit, so the kernel
// failed to JIT and took the whole program down with it. fattn-vec.hpp now stages those partials a
// few warps at a time and carries the running sum in registers, capping the staging area
// independently of D. head_dim 256 matters because qwen35 and similar hybrids use it, and without
// a turbo instance there the whole attention op silently falls back to the CPU (ADR-0008).
#define FATTN_VEC_CASES_TURBO_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)         \
    FATTN_VEC_CASE(128, type_K, type_V)         \
    FATTN_VEC_CASE(256, type_K, type_V)         \

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    // Auto-asymmetric mixed-KV: for high-GQA models (>=6:1) the KV cache upgrades K to q8_0 to
    // protect quality while V stays turbo (llama-kv-cache.cpp). K=q8_0 uses the normal split
    // nthreads_KQ (no full-D-Q register pressure), so D=128 is safe here too.
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    // TURBO BEGIN - asymmetric f16-K + turbo-V: the recommended prod default is q8_0-K, but f16-K
    // is the maximum-precision asymmetric mode ("V is free, K is everything"). The graph leaves Q
    // un-rotated when K is f16 (forward Q-WHT gates on K->type in llama-graph.cpp) and still applies
    // the inverse-WHT on the turbo-V output (gates on V->type), so f16-K + turbo-V is valid. Same
    // D in {64,128} cap as the other turbo-V rows; K=f16 has no full-D-Q register pressure. Only
    // (f16, turbo3) is extern-declared (fattn-vec-instance-f16-tq3.cpp); (f16, turbo2)/(f16, turbo4)
    // are implicitly instantiated here (turbo2/turbo4 are absent from the type_V extern grid).
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16, GGML_TYPE_TURBO4_0)
    // TURBO END
    // K=turbo / V=q8_0 mixed-KV: arises from layer-adaptive Boundary-V mode 7 (TURBO_LAYER_ADAPTIVE=7)
    // or an explicit "-ctk turboN -ctv q8_0". K stays turbo so Q is still WHT-rotated; V=q8_0 is
    // unrotated and the graph skips the inverse-WHT on those layers (gated on V->type turbo). turbo3/
    // turbo4 K are solid here; turbo2 K (2-bit) is marginal, so the turbo2-V auto path defaults to the
    // symmetric mode 8 (q8_0 boundaries) instead of mode 7 (llama-kv-cache.cpp).
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)

    // TURBO BEGIN - mixed-width turbo KV: turbo4 K plus a cheaper turbo V.
    // Measured on B580 (wikitext-2, Qwen3-4B Q4_K_M, c=512, 32 chunks, f16 baseline PPL 9.054):
    // turbo on V costs ~0.4%, turbo on K is what costs, and turbo4 is the quality knee on K
    // (turbo4 x turbo4 +5.1%, turbo3 x turbo3 +31.7%). So turbo4-K with a cheaper V is a frontier
    // point symmetric turbo cannot reach: ~4.3-4.5x KV saving instead of 3.76x, at turbo4-K quality.
    // Neither pair is extern-declared - TURBO4_0 is absent from the type_K axis of
    // EXTERN_DECL_FATTN_VEC_CASES (fattn-vec.hpp) - so both instantiate implicitly here, exactly
    // like the (TURBO4_0, TURBO4_0) and (TURBO4_0, Q8_0) rows above. No instance file is needed,
    // which is where this differs from ADR-0005.
    // DO NOT add EXTERN_DECL_FATTN_VEC_CASES(D, GGML_TYPE_TURBO4_0): that would suppress the
    // implicit instantiation for every TURBO4_0-K row and produce LNK2019 (ADR-0004).
    // The reverse widths (turbo3-K x turbo4-V) cost the same bytes but put the coarse quantizer on
    // the side that dominates the error, so they are strictly dominated and stay out. See ADR-0007.
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    // TURBO END
#endif // GGML_SYCL_FA_ALL_QUANTS

    // Reachable from ordinary CLI flags, not just an internal invariant, so say what to do instead
    // of only what went wrong. Examples that land here: -ctk turbo3 -ctv turbo2, or any
    // -ctk q4_0/q4_1/q5_0/q5_1 paired with a turbo V. Note -ctv turbo2 additionally auto-enables
    // adaptive mode 8 (llama-kv-cache.cpp), which leaves -ctk untouched on non-boundary layers, so
    // an unsupported K survives into the layers that actually run this kernel.
    GGML_ABORT("Not match KV type in vec: Q->ne[0]=%d K->type=%s V->type=%s\n"
               "  This KV type pair has no curated SYCL flash-attention instance.\n"
               "  Supported with a turbo V: K in {f16, q8_0, same-width turbo, turbo4}.\n"
               "  Supported with a turbo K: V in {q8_0, same-width turbo}, or V in\n"
               "  {turbo3, turbo2} when K is turbo4.\n"
               "  Recommended: -ctk q8_0 -ctv turbo3 (within 0.5%% of f16 perplexity).",
               (int) Q->ne[0], ggml_type_name(K->type), ggml_type_name(V->type));
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_TILE     = 200,
};

static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    {
        // Turbo KV may legitimately be asymmetric: the auto-asymmetric path upgrades K to q8_0
        // while V stays turbo (GQA >= 6). The graph gates the forward Q-WHT on K and the output
        // inverse-WHT on V independently, and the vec kernel dequantizes K and V by independent
        // types, so such pairs are valid. Only reject asymmetric pairs where NEITHER side is turbo.
        const bool kv_has_turbo =
            K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 ||
            V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;
        if (K->type != V->type && !kv_has_turbo) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    const bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // Turbo KV is served EXCLUSIVELY by the vector kernel (matches the CUDA reference). The tile
    // kernel cannot dequantize a turbo V cache (it is templated on type_K only and reads V as raw
    // half2), so turbo must never route there. The vec kernel dequantizes both turbo K and V and
    // consumes the graph-rotated Q. If the vector kernel cannot run (e.g. head_dim % 64 != 0),
    // report NONE rather than falling back to the broken tile path.
    const bool KV_is_turbo =
        K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 ||
        V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0;
    if (KV_is_turbo) {
        // Must track FATTN_VEC_CASES_TURBO_D exactly. Returning NONE here does not select some other
        // SYCL kernel: it makes ggml_sycl_flash_attn_ext_supported report false, and ggml then runs
        // the whole attention op on the CPU, silently and roughly 10x slower (ADR-0008).
        return (can_use_vector_kernel && K->ne[0] <= 256) ? BEST_FATTN_KERNEL_VEC : BEST_FATTN_KERNEL_NONE;
    }

    // Todo: Use the XMX kernel if possible:

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// --- TurboQuant prefill acceleration --------------------------------------------------------
// The turbo VEC kernel is decode-oriented; for prefill (many Q columns) it is ~3-7x slower than
// the batch-optimized f16 TILE kernel (turbo3 pp8192 82 t/s vs f16 580 on Arc B580). Turbo K/V
// dequantize to f16 in the rotated domain (Q is graph-rotated, the FA output is graph-inverse-WHT
// rotated), so for prefill we dequantize the turbo cache to a transient f16 scratch and run the
// proven f16 TILE. Decode (Q->ne[1] <= 2) stays on turbo VEC, preserving the full KV memory saving.

template <typename block_t, int QK, float (*dequant_fn)(const block_t *, int, float)>
static void k_turbo_dequant_to_f16(const char * __restrict__ src, sycl::half * __restrict__ dst,
                                   const int64_t ne0, const int64_t ne1, const int64_t ne2,
                                   const int64_t nb1, const int64_t nb2,
                                   const sycl::nd_item<3> & it) {
    const int64_t i0 = (int64_t) it.get_global_id(2); // head-dim element
    const int64_t i1 = (int64_t) it.get_global_id(1); // KV position
    const int64_t i2 = (int64_t) it.get_global_id(0); // KV head
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
        return;
    }
    const block_t * blk = (const block_t *) (src + i1*nb1 + i2*nb2) + (i0 / QK);
    const float v = dequant_fn(blk, (int) (i0 % QK), (float) blk->norm);
    dst[(i2*ne1 + i1)*ne0 + i0] = (sycl::half) v; // contiguous [ne0, ne1, ne2]
}

template <typename block_t, int QK, float (*dequant_fn)(const block_t *, int, float)>
static void turbo_dequant_to_f16_sycl(const char * src, sycl::half * dst,
                                      int64_t ne0, int64_t ne1, int64_t ne2,
                                      int64_t nb1, int64_t nb2, queue_ptr stream) {
    constexpr int WG = 64;
    const sycl::range<3> lws(1, 1, WG);
    const sycl::range<3> gws(ne2, ne1, ((ne0 + WG - 1) / WG) * WG);
    stream->parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> it) {
        k_turbo_dequant_to_f16<block_t, QK, dequant_fn>(src, dst, ne0, ne1, ne2, nb1, nb2, it);
    });
}

// q8_0 shares the same blocked/strided layout but keeps its scale in `d` and stores plain int8
// quants, so it cannot be expressed as a dequant_fn(block, j, norm) instance.
static void k_q8_0_dequant_to_f16(const char * __restrict__ src, sycl::half * __restrict__ dst,
                                  const int64_t ne0, const int64_t ne1, const int64_t ne2,
                                  const int64_t nb1, const int64_t nb2,
                                  const sycl::nd_item<3> & it) {
    const int64_t i0 = (int64_t) it.get_global_id(2); // head-dim element
    const int64_t i1 = (int64_t) it.get_global_id(1); // KV position
    const int64_t i2 = (int64_t) it.get_global_id(0); // KV head
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2) {
        return;
    }
    const block_q8_0 * blk = (const block_q8_0 *) (src + i1*nb1 + i2*nb2) + (i0 / QK8_0);
    const float v = (float) blk->d * (float) blk->qs[i0 % QK8_0];
    dst[(i2*ne1 + i1)*ne0 + i0] = (sycl::half) v; // contiguous [ne0, ne1, ne2]
}

static void q8_0_dequant_to_f16_sycl(const char * src, sycl::half * dst,
                                     int64_t ne0, int64_t ne1, int64_t ne2,
                                     int64_t nb1, int64_t nb2, queue_ptr stream) {
    constexpr int WG = 64;
    const sycl::range<3> lws(1, 1, WG);
    const sycl::range<3> gws(ne2, ne1, ((ne0 + WG - 1) / WG) * WG);
    stream->parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> it) {
        k_q8_0_dequant_to_f16(src, dst, ne0, ne1, ne2, nb1, nb2, it);
    });
}

// Quantized KV types the prefill shim can materialize as contiguous f16 for the f16 TILE.
// F16 is handled by passing the tensor through untouched, so it is not listed here.
static bool kv_dequantizable_to_f16(ggml_type t) {
    return t == GGML_TYPE_Q8_0 ||
           t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
}

static void kv_dequant_tensor_to_f16(const ggml_tensor * t, sycl::half * dst, queue_ptr stream) {
    const char *  src = (const char *) t->data;
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2];
    const int64_t nb1 = t->nb[1], nb2 = t->nb[2];
    switch (t->type) {
        case GGML_TYPE_TURBO2_0:
            turbo_dequant_to_f16_sycl<block_turbo2_0, QK_TURBO2, dequantize_turbo2_0>(src, dst, ne0, ne1, ne2, nb1, nb2, stream); break;
        case GGML_TYPE_TURBO3_0:
            turbo_dequant_to_f16_sycl<block_turbo3_0, QK_TURBO3, dequantize_turbo3_0>(src, dst, ne0, ne1, ne2, nb1, nb2, stream); break;
        case GGML_TYPE_TURBO4_0:
            turbo_dequant_to_f16_sycl<block_turbo4_0, QK_TURBO4, dequantize_turbo4_0>(src, dst, ne0, ne1, ne2, nb1, nb2, stream); break;
        case GGML_TYPE_Q8_0:
            q8_0_dequant_to_f16_sycl(src, dst, ne0, ne1, ne2, nb1, nb2, stream); break;
        default:
            GGML_ABORT("kv_dequant_tensor_to_f16: unsupported KV type");
    }
}

// Dequantize a turbo-bearing K/V pair to transient f16 and run the f16 TILE for prefill.
// Handles symmetric turbo as well as the asymmetric pairs ({f16,q8_0} x turbo and the reverse):
// each non-f16 side gets a contiguous f16 shadow, an f16 side is passed through untouched.
// Rotation stays consistent because the graph gates the forward Q-WHT on the *original* K type
// and the output inverse-WHT on the *original* V type; both were fixed at graph-build time and
// the turbo dequant reproduces the rotated domain the rotated Q expects.
// Returns true if it handled dst; false to fall through to the normal kernel selection.
static bool ggml_sycl_flash_attn_ext_turbo_prefill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    auto f16able = [](ggml_type t) { return t == GGML_TYPE_F16 || kv_dequantizable_to_f16(t); };

    // Require a turbo V and never mix turbo widths. That leaves exactly two shapes: symmetric turbo,
    // and the production-relevant asymmetry (precise K + compressed V, "V is free, K is everything").
    // The rest are refused on purpose, each for its own reason:
    //   - (turbo K, f16 V) has no curated FATTN_VEC_CASES_TURBO_D row, so decode aborts anyway;
    //   - (turbo K, q8_0 V) keeps its existing VEC prefill because routing it through the f16 TILE
    //     drifts past the golden rel-MSE bound for the coarsest K (turbo2 x q8_0 measured 8.3e-3
    //     against a 6e-3 tolerance on B580, while turbo3/turbo4 land at 1.9e-3/1.2e-3);
    //   - mixed-width turbo other than the curated turbo4-K pairs has no vec instance either, and is
    //     reachable both directly and through TURBO_LAYER_ADAPTIVE modes 5/6. Letting those in here
    //     would make prefill succeed and push the abort to the first decode token, i.e. after the
    //     whole prompt had already been processed.
    // The allowlist below must stay in lockstep with FATTN_VEC_CASES_TURBO_D above and with
    // `uses_tile` in tests/test-sycl-turbo.cpp.
    auto mixed_width_dispatchable = [](ggml_type k, ggml_type v) {
        return k == GGML_TYPE_TURBO4_0 && (v == GGML_TYPE_TURBO3_0 || v == GGML_TYPE_TURBO2_0);
    };
    if (!is_turbo(V->type)) return false;
    if (!f16able(K->type)) return false;
    // A turbo K at D>=256 drifts through the f16 TILE: measured rel-MSE 2.2e-2 for turbo3 x turbo3
    // and 6.2e-3 for turbo4 x turbo3 against a 6e-3 bound, while the SAME pairs through the turbo
    // VEC kernel measure exactly 0.0 and precise-K x turbo-V through TILE measures 2-3e-4. So the
    // data and the rotation are right and it is the f16 accumulation over 256 rotated terms that
    // loses ground. Keep those on VEC, mirroring the (turbo K, q8_0 V) exclusion below.
    if (is_turbo(K->type) && K->ne[0] >= 256) return false;
    if (is_turbo(K->type) && K->type != V->type && !mixed_width_dispatchable(K->type, V->type)) {
        return false;
    }
    // Column threshold. This shim dequantizes the ENTIRE K/V cache into f16 scratch, so it only
    // pays for itself once there are enough Q columns to amortise that full-cache pass; break-even
    // is around 12-16 columns. Batched decode under a unified KV cache produces a small
    // Q->ne[1] > 2 with K->ne[3] == 1 (llama-server defaults to n_parallel 4 + kv_unified, and
    // speculative decoding contributes draft+1 columns), which must stay on VEC. Real prefill
    // chunks are n_batch-sized and clear 16 by a wide margin. The bound is a property of this shim,
    // not of the K type, so symmetric turbo gets it too.
    if (Q->ne[1] < 16) return false;                          // decode -> keep turbo VEC
    const int64_t D = K->ne[0];
    if (!(D == 64 || D == 128 || D == 256 || D == 512)) return false; // f16 TILE head sizes
    if (V->ne[0] != D) return false;                          // e.g. transposed V -> bail to VEC
    // The dequant kernels index [ne0,ne1,ne2] only; a batched cache would be silently truncated.
    if (K->ne[3] != 1 || V->ne[3] != 1) return false;

    queue_ptr stream = ctx.stream();
    ggml_sycl_pool_alloc<sycl::half> k_f16(ctx.pool());
    ggml_sycl_pool_alloc<sycl::half> v_f16(ctx.pool());

    // Contiguous f16 shadows; the f16 TILE reads K and V through independent strides
    // (stride_K2 = nb11/2, stride_V2 = nb21/2), so a shadow may be mixed with a real f16 view.
    ggml_tensor Kf = *K;
    if (K->type != GGML_TYPE_F16) {
        k_f16.alloc(ggml_nelements(K));
        kv_dequant_tensor_to_f16(K, k_f16.get(), stream);
        Kf.type = GGML_TYPE_F16;
        Kf.data = k_f16.get();
        Kf.nb[0] = sizeof(sycl::half);
        Kf.nb[1] = Kf.nb[0] * Kf.ne[0];
        Kf.nb[2] = Kf.nb[1] * Kf.ne[1];
        Kf.nb[3] = Kf.nb[2] * Kf.ne[2];
    }

    ggml_tensor Vf = *V;
    if (V->type != GGML_TYPE_F16) {
        v_f16.alloc(ggml_nelements(V));
        kv_dequant_tensor_to_f16(V, v_f16.get(), stream);
        Vf.type = GGML_TYPE_F16;
        Vf.data = v_f16.get();
        Vf.nb[0] = sizeof(sycl::half);
        Vf.nb[1] = Vf.nb[0] * Vf.ne[0];
        Vf.nb[2] = Vf.nb[1] * Vf.ne[1];
        Vf.nb[3] = Vf.nb[2] * Vf.ne[2];
    }

    ggml_tensor dstc = *dst;
    dstc.src[1] = &Kf;
    dstc.src[2] = &Vf;
    ggml_sycl_flash_attn_ext_tile(ctx, &dstc);
    return true;
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);
    if (ggml_sycl_flash_attn_ext_turbo_prefill(ctx, dst)) {
        return;
    }
    switch (ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        case BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
    }
}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
