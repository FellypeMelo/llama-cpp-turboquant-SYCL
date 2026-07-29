#include "../fattn-vec.hpp"

// Auto-asymmetric mixed-KV: high-GQA models upgrade K to q8_0 while V stays turbo3 (see
// llama-kv-cache.cpp). (Q8_0, TURBO3_0) is extern-declared in fattn-vec.hpp (TURBO3_0 appears
// on the type_V axis for every type_K), so it needs an explicit instantiation here. Capped to
// D in {64,128} like the symmetric turbo instances; K=q8_0 uses the normal split nthreads_KQ
// (no full-D-Q register pressure), so no D=256/512 spill concern beyond the turbo V cap.
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);

// D=256 needs an explicit definition here because this pair IS extern-declared at 256 in
// fattn-vec.hpp; the TURBO4_0-K rows are absent from that grid and instantiate implicitly.
// Adding EXTERN_DECL_FATTN_VEC_CASES for TURBO4_0 would break those with LNK2019 (ADR-0004).
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0);
