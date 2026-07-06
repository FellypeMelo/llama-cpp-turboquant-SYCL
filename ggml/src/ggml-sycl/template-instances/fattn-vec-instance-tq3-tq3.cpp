#include "../fattn-vec.hpp"

// Turbo uses nthreads_KQ=1 (full pre-rotated Q per lane). On Intel GPUs that fits in the 128-GRF
// large-register file only for D<=128; D=256/512 spill catastrophically and fail the JIT build,
// so those instances are intentionally omitted (see FATTN_VEC_CASES_TURBO_D in fattn.cpp). Restore
// D=256 with the P2 cooperative-register rework.
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0);
