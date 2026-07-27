# Changelog — TurboQuant SYCL fork

All notable **fork-specific** changes are documented here — the TurboQuant
KV-cache work and the maintenance around it. This file does not track
upstream `llama.cpp` history; see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
for that.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
A `tqp-sycl-v0.1.0` tag exists in this repository, but it marks the point
where the release-packaging CI workflow was added, not a numbered content
release of the turbo feature set — this fork has not yet adopted semantic
versioning for its own change history, so every entry below stays under
`[Unreleased]` rather than being assigned a version number that would
otherwise have to be invented.

## [Unreleased]

### Added

- Asymmetric K/V precision: `f16`-K + turbo-V flash-attention dispatch on
  SYCL, so a precise (unquantized) key can be paired with a compressed turbo
  value (`-ctk f16 -ctv turbo3`, etc.) in addition to the existing
  `q8_0`-K + turbo-V combination. Three curated dispatch rows plus one new
  instance file; `GGML_SYCL_FA_ALL_QUANTS` stays disabled (see ADR-0005 in
  `docs/en/decisions.md`).
- Golden numerical-parity coverage extended to asymmetric mixed-KV
  (`{q8_0,f16} x {turbo2,turbo3,turbo4}`) in `tests/test-sycl-turbo.cpp`.
- End-to-end turbo KV-cache coherence gate: `tests/test-e2e-turbo-kv.sh`,
  wired into CTest as `test-e2e-turbo-kv` (labels `e2e;gpu;turbo`). Drives
  real `llama-cli` generation with turbo KV cache on and asserts the output
  is coherent (on-topic keyword, no degenerate repetition, no abort). SKIPs
  (exit 77) without a GPU/model, so it is a no-op on hosted CI and a real
  gate only on the maintainer's self-hosted Arc box. Later extended to also
  cover the asymmetric precise-K + turbo-V combinations.

### Changed

- Merged `upstream/master` (`fb30ba9a6`, 2026-07-09) into the turbo fork.
  Merge-base was `7fc1c4ef` (2026-04-21); roughly 1,075 upstream commits and
  230 fork commits had diverged. 31 files conflicted; all were resolved by
  re-integrating the turbo logic onto upstream's new structure rather than
  taking either side blindly (see ADR-0001/ADR-0002 in `docs/en/decisions.md`
  and `docs/en/upstream-sync.md`).
- Type numbering realigned with upstream (ADR-0003, Option A): kept
  `GGML_TYPE_Q2_0 = 42` as upstream defines it and moved the turbo types to
  43–47 (`TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47`),
  `GGML_TYPE_COUNT = 48`. Mirrored in `gguf-py/gguf/constants.py`.
  `llama_ftype` is unaffected (both sides' ftypes coexist without collision).

### Fixed

- `ggml.c`: `static_assert(GGML_OP_COUNT == …)` bumped 97 → 98 so the turbo
  op `GGML_OP_TURBO_WHT` coexists with upstream's new ops.
- `ggml-sycl.cpp`: updated the renamed upstream helper,
  `get_sycl_env(...)` → `ggml_sycl_get_env(...)`.
- `ggml-sycl/common.hpp`: kept `GGML_SYCL_FA_ALL_QUANTS` disabled (upstream
  now enables it by default). Enabling it forces instantiation of turbo
  flash-attention template combinations that don't have generated instances
  (turbo × standard, and turbo at head_dim ≥ 256), which fails to link
  (`LNK2019`). Disabling it restores the fork's validated flash-attention
  path (ADR-0004 in `docs/en/decisions.md`).
- `llama-kv-cache.cpp`: re-integrated the auto-asymmetric K-upgrade, the
  `+3` rotation-tensor `mem_size` adjustment, and the off-by-default
  attention-rotation policy onto upstream's refactored `n_layer_all()` /
  `n_layer_kv()` and its new shared-cell / DeepSeek-indexer structure.
- `llama-context.cpp`: turbo K/V head-dim padding now reads
  `hparams.n_layer()` instead of a field upstream removed.
- SYCL dispatch (`mmvq`, `cpy`, `convert`, `set_rows`, `ggml-sycl.cpp`):
  turbo cases merged as a union with upstream's new types
  (Q1_0/MXFP4/NVFP4/Q2_0) instead of being overwritten by the merge.
- `llama-model-loader.cpp`: `TQ3_1S`/`TQ4_1S` file-type names ported to
  upstream's prefix-based naming style.

### Known limitations

- The PPL/speed quality gate (`scripts/turbo-quality-gate.sh`) has not been
  re-run in the sessions documented so far — it needs a locally available
  pure-attention reference model plus a `wikitext-2-raw` dataset. Treat it
  as a documented target, not a currently-passing measurement (see
  `docs/en/testing.md`).
- CUDA/Metal/Vulkan turbo code paths were resolved during the upstream
  merge but are not compiled or re-validated as part of the SYCL-only gates
  described in `docs/en/testing.md`; they are best-effort.
- Vulkan turbo3 flash-attention on the scalar/coopmat1 paths became
  non-functional after the 2026-07 upstream merge (upstream moved the
  flash-attention dequant code to a new file); the coopmat2 path is
  preserved.
- Re-enabling `GGML_SYCL_FA_ALL_QUANTS` requires generating the full
  turbo flash-attention-vec instance matrix (turbo × every KV type, every
  supported head dim) — out of scope until there is a concrete need for it.
- A perplexity rescue run on Qwen2.5-7B Q4_K_M (to validate the asymmetric
  K/V combinations on a model with a different GQA ratio) is pending; the
  model was not present on the validation machine's disk.
