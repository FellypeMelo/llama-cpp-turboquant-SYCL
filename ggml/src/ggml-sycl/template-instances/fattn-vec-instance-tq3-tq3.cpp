#include "../fattn-vec.hpp"

// Turbo uses nthreads_KQ=1 (full pre-rotated Q per lane). On Intel GPUs that fits in the 128-GRF
// large-register file only for D<=128; D=256/512 spill catastrophically and fail the JIT build,
// so those instances are intentionally omitted (see FATTN_VEC_CASES_TURBO_D in fattn.cpp). Restore
// D=256 with the P2 cooperative-register rework.
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);

// D=256 needs an explicit definition here because this pair IS extern-declared at 256 in
// fattn-vec.hpp; the TURBO4_0-K rows are absent from that grid and instantiate implicitly.
// Adding EXTERN_DECL_FATTN_VEC_CASES for TURBO4_0 would break those with LNK2019 (ADR-0004).
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
