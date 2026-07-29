#include "../fattn-vec.hpp"

// Boundary-V mixed-KV: layer-adaptive mode 7 (auto-enabled for turbo2 V, or TURBO_LAYER_ADAPTIVE=7)
// upgrades V to q8_0 on the boundary layers while K stays turbo (see llama-kv-cache.cpp). For turbo3 K
// this gives (TURBO3_0, Q8_0). Both TURBO3_0 (type_K axis) and Q8_0 (type_V axis) are in the extern
// grid in fattn-vec.hpp, so this combo is extern-declared there and needs an explicit instantiation.
// Capped to D in {64,128} like the other turbo instances (K=turbo => nthreads_KQ=1 full-D-Q registers).
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);

// D=256 needs an explicit definition here because this pair IS extern-declared at 256 in
// fattn-vec.hpp; the TURBO4_0-K rows are absent from that grid and instantiate implicitly.
// Adding EXTERN_DECL_FATTN_VEC_CASES for TURBO4_0 would break those with LNK2019 (ADR-0004).
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0);
