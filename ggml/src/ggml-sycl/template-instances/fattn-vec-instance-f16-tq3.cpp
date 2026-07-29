#include "../fattn-vec.hpp"

// Asymmetric max-precision mixed-KV: f16 K (un-rotated, no quant loss) while V stays turbo3 (see
// llama-graph.cpp: the forward Q-WHT gates on K->type, the inverse-WHT on V->type). (F16, TURBO3_0)
// is extern-declared in fattn-vec.hpp (TURBO3_0 appears on the type_V axis for every type_K), so it
// needs an explicit instantiation here. Capped to D in {64,128} like the symmetric turbo instances;
// K=f16 uses the normal split nthreads_KQ (no full-D-Q register pressure), so no D=256/512 spill
// concern beyond the turbo V cap. (F16, TURBO2_0)/(F16, TURBO4_0) are implicitly instantiated in
// fattn.cpp (turbo2/turbo4 are absent from the type_V extern grid), so they need no file here.
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);

// D=256 needs an explicit definition here because this pair IS extern-declared at 256 in
// fattn-vec.hpp; the TURBO4_0-K rows are absent from that grid and instantiate implicitly.
// Adding EXTERN_DECL_FATTN_VEC_CASES for TURBO4_0 would break those with LNK2019 (ADR-0004).
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_TURBO3_0);
