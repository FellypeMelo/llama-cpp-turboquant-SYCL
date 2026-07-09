# Changelog — TurboQuant SYCL fork

## [Unreleased] — sync/upstream-2026-07

### Merged upstream
- Merged `upstream/master` (`fb30ba9a6`, 2026-07-09) into the turbo fork. merge-base was
  `7fc1c4ef` (2026-04-21); ~1075 upstream commits, 230 fork commits. 31 files conflicted; all
  resolved by re-integrating the turbo logic onto upstream's new structure (not blind theirs/ours).

### Type numbering (ADR-0003, Option A)
- Aligned `GGML_TYPE_Q2_0 = 42` with upstream; moved the turbo types to 43–47
  (`TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47`), `GGML_TYPE_COUNT = 48`.
  Mirrored in `gguf-py/gguf/constants.py`. `llama_ftype` unchanged (take-both, no collision).

### Fixed (silent auto-merge breakages caught by the build)
- `ggml.c`: bumped `static_assert(GGML_OP_COUNT == …)` 97 → 98 (turbo op `GGML_OP_TURBO_WHT`
  coexisting with upstream's new ops).
- `ggml-sycl.cpp`: `get_sycl_env(...)` → `ggml_sycl_get_env(...)` (upstream renamed the helper).
- `ggml-sycl/common.hpp`: keep `GGML_SYCL_FA_ALL_QUANTS` **disabled** (upstream now enables it by
  default). Enabling it odr-uses turbo FA-vec template instances that were never generated
  (turbo×standard, and turbo @ D≥256) → LNK2019. Disabling restores the fork's validated FA path.

### Re-integrated turbo hooks onto upstream refactors
- `llama-kv-cache.cpp`: auto-asymmetric K-upgrade, `+3` rotation-tensor mem_size, and the
  OFF-by-default attention-rotation policy adapted onto upstream's `n_layer_all`/`n_layer_kv()` and
  the new `if(other)` shared-cell + DeepSeek-indexer structure.
- `llama-context.cpp`: turbo K/V head-dim padding now uses `hparams.n_layer()` (was a field).
- SYCL dispatch (`mmvq`/`cpy`/`convert`/`set_rows`/`ggml-sycl.cpp`): turbo cases merged as a union
  with upstream's new types (Q1_0/MXFP4/NVFP4/Q2_0).
- `llama-model-loader.cpp`: `TQ3_1S`/`TQ4_1S` ftype names ported to upstream's prefix-based style.

### Validation (Intel Arc B580, SYCL)
- Build: green (icx/icpx, Ninja, Release). `ggml-sycl.dll`, `test-sycl-turbo.exe`, `llama-cli.exe`.
- Golden gate `test-sycl-turbo`: **all PASSED, exit 0** (turbo quant cosine ~1.0, TQ4_1S 0.999843,
  WHT round-trip, set_rows, FA turbo parity, FA DECODE+GQA, FA with TURBO3_0 KV cache).

### Known follow-ups (documented, not blocking SYCL)
- PPL end-to-end gate + bench not run this session (reference pure-attention model + wikitext absent).
- CUDA/Metal/Vulkan conflicts resolved but not compiled (SYCL-only build). Vulkan turbo3 FA on the
  scalar/coopmat1 paths went non-functional post-merge (upstream moved FA dequant to a new file);
  CM2 turbo3 preserved. Best-effort non-Arc.
- Re-enabling `GGML_SYCL_FA_ALL_QUANTS` requires generating the full turbo FA-vec instance matrix.
