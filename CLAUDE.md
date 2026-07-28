# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Read [AGENTS.md](AGENTS.md) in full before any work. This file summarizes the fork-specific
> parts; AGENTS.md is authoritative and its PR/commit rules are non-negotiable.

## What this repository is

A **private fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** whose reason to exist is
**TurboQuant KV-cache quantization on the SYCL backend for Intel GPUs**. Everything turbo-related is the
product; the rest of llama.cpp is upstream scaffolding.

- **SYCL is the only backend that matters.** CUDA/Metal/Vulkan turbo files exist as reference/ports but are
  not validated here. Do not spend effort on non-SYCL backends unless explicitly asked.
- **Validation hardware:** Intel Arc B580 (Battlemage / Xe2, `bmg_g21`), 12 GB. **Reference model:**
  Qwen3-4B Q4_K_M (head_dim 128, GQA 4:1, 36 layers, pure attention).
- Branch of record: `feature/turboquant-kv-cache`. `origin` = the user's fork
  (github.com/FellypeMelo/llama-cpp-turboquant-SYCL); `upstream` = ggml-org (**never push to upstream**).

## Non-negotiable rules (from AGENTS.md)

- **Never** run `git push`, `gh pr create`, `gh pr comment`, or `gh issue create` on the user's behalf, and
  **never** author a PR description, commit message defense, or reviewer reply. This is stated as
  non-overridable. Private forks are exempt from the maintainer *closure* policy, but the mechanical act of
  pushing / opening a PR is the human's to perform - prepare the branch and hand over the commands.
- **No commit or push without explicit per-action human approval.** When the user explicitly asks you to
  commit, use a trailer `Assisted-by: <assistant name>` - **never** `Co-authored-by:`.
- **ASCII only** in code and commit messages: use `-` `->` `x` `...`, not em-dash / unicode arrows / multiply-sign / ellipsis glyphs.
- Keep comments concise (explain non-obvious invariants, not what the code already says). Prefer reusing
  existing infrastructure over new subsystems; if a change is large or introduces a new pattern, pause and
  confirm with the user first.
- **Test-first on real silicon.** No turbo change is "done" until it passes the on-device golden numeric test
  plus the e2e coherence gate on the B580. "Looks coherent" is not evidence.

## Build (Windows, oneAPI, Intel Arc)

oneAPI lives at `C:\Program Files (x86)\Intel\oneAPI` (compiler `2026.0`), VS 2022. The one-shot script is
`build_sycl.bat`:

```bat
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8 --target llama-cli llama-server llama-bench llama-quantize test-sycl-turbo
```

Two build flavors are used:
- **build-sync** - JIT (kernels compiled on first run), `ggml-sycl.dll` ~60 MB. Fast to build, slow first launch.
- **build-perf** - AOT for `bmg_g21` (`-fsycl-targets=intel_gpu_bmg_g21`), DLL ~262 MB. Slower build, fast launch. Use for real perf/PPL numbers. (Note: `--offload-arch` alone is *not* AOT.)

**Toolchain gotchas** (running `icx`/`setvars.bat` under an agent shell):
1. Add `C:\Program Files (x86)\Microsoft Visual Studio\Installer` to PATH (for `vswhere.exe`).
2. `set "NoDefaultCurrentDirectoryInExePath="` so cmd resolves component `vars.bat` from cwd.
3. `icx` defaults to `-fno-exceptions` - add `/EHsc` for standalone compiles.
4. Run `.bat` via the **PowerShell tool** (`cmd /c ...`); the Bash tool's `cmd //c` loses the system PATH and `vswhere` fails.

**Before relinking:** kill any running `llama-cli` / `llama-server` / `llama-bench` / `test-sycl-*` - they hold `ggml-sycl.dll` and block the link. **Never run two GPU benches/servers concurrently** on the single B580 (they corrupt each other's measurements and clash on the device).

## Test / validate (the three turbo gates)

There are three complementary gates; a turbo change must pass the first two before it is considered correct.

1. **Golden numeric parity** (`tests/test-sycl-turbo.cpp` -> `build/bin/test-sycl-turbo.exe`, ctest name
   `test-sycl-turbo`). CPU-golden vs SYCL kernel: quant cosine ~1.0, flash-attention parity cosine >= 0.99999
   across turbo2/3/4, symmetric + asymmetric (precise-K x turbo-V), D in {64,128}. This is the primary gate.
   Run it standalone via `run_test.bat` (sources setvars, runs the exe) or:
   ```
   ctest --test-dir build -R test-sycl-turbo -V
   ```
2. **E2E coherence** (`tests/test-e2e-turbo-kv.sh`, ctest name `test-e2e-turbo-kv`, labels `e2e;gpu;turbo`).
   Drives real `llama-cli` generation on the GPU and asserts coherent text (on-topic keyword, no `?`-corruption,
   no degenerate 5-gram repetition, no abort/NaN) for the gated configs. Requires the reference model:
   ```
   TURBO_E2E_MODEL=/path/Qwen3-4B-Q4_K_M.gguf bash tests/test-e2e-turbo-kv.sh
   ```
   Absent model -> exit 77 (SKIP). Gated configs: symmetric turbo3/turbo4/turbo2, and asymmetric
   `{q8_0,f16} x {turbo2,turbo3,turbo4}`.
3. **Quality + speed** (`scripts/turbo-quality-gate.sh`) - perplexity within 5% of q8_0 and context-scaling
   ratio > 0.95. Needs wikitext-2 + a larger model; run before shipping perf-affecting changes.

**Run a single golden case:** the golden binary runs its whole suite; to bisect, filter with ctest
(`ctest --test-dir build -R test-sycl-turbo`) or run the exe directly and read its per-case PASS/FAIL lines.

## Architecture: how TurboQuant works here

**The idea.** For long context the KV cache, not the weights, is the memory wall (fp16 KV @64k ~= 9.2 GB,
near-OOM on 12 GB). TurboQuant compresses KV to ~2-4 bits/value by **rotating then quantizing**: a random-sign
**Walsh-Hadamard Transform (WHT)** per head-dim vector smooths outliers into a near-Gaussian that a tiny
Lloyd-Max codebook quantizes cleanly. Rotation is orthogonal, so `softmax(Q.K^T)` is invariant to it.

**Three formats** (block size 128 = head_dim; byte sizes are the `static_assert` truth, some header comments are stale):
`turbo2` 34 B / 2.13 bpv / 7.5x, `turbo3` 50 B / 3.13 bpv / 5.1x (best balance), `turbo4` 68 B / 4.25 bpv / 3.8x.

**Where rotation lives (threaded through the compute graph, never re-quantized in the hot loop):**
- **K/V** rotated + quantized once at cache-write (`ggml-sycl/set_rows.cpp`).
- **Q** pre-rotated by a dedicated graph node `GGML_OP_TURBO_WHT` (gated on `k->type` being turbo).
- **Output** inverse-rotated after attention (gated on `v->type` being turbo).
- Consequence for asymmetric KV: with `K=f16` Q is left un-rotated while `V=turbo` still applies the inverse
  WHT - no graph edits needed to mix precise-K with turbo-V.

**Two attention kernels, opposite sweet spots:**
- **VEC** (`fattn-vec.hpp`, decode-oriented, 1-2 cols/block) - dequantizes each K/V element inside the inner
  product. **All turbo decode goes here.**
- **TILE** (`fattn-tile.hpp`, batch/prefill-oriented f16 GEMM-style).

**Prefill "Option A"** (in `fattn.cpp`, `ggml_sycl_flash_attn_ext_turbo_prefill`): turbo prefill on VEC was
3.2x slower than f16 (widening to 7x at 8k). Rather than write a fragile native turbo-TILE loader (the
34/50/68-byte non-power-of-two strides produced garbage), a small strided kernel dequantizes the permuted,
non-contiguous turbo K/V views into **contiguous f16 scratch**, then the proven **f16 TILE** kernel runs on it.
Result: prefill at f16 parity at every length; decode stays on VEC so the memory saving is preserved.

**Dispatch/router** (`fattn.cpp`, `ggml_sycl_get_best_fattn_kernel`): if any side is turbo -> VEC path for
decode; asymmetric precise-K/turbo-V decode -> curated `FATTN_VEC_CASES_TURBO_D` rows. Unhandled combos hit
`GGML_ABORT "Not match KV type in vec"`.

Option A (`ggml_sycl_flash_attn_ext_turbo_prefill`) runs *before* that router and requires a **turbo V** with
K in {f16, q8_0, same-width turbo}; each non-f16 side gets a contiguous f16 shadow, an f16 side passes
through. Column threshold `Q->ne1 >= 16`, uniform - the shim dequantizes the whole cache, so batched decode
under a unified KV cache (server default `n_parallel 4` + `kv_unified`) must stay on VEC. Excluded on purpose:
reverse pairs (turbo K + f16/q8_0 V) and **mixed-width turbo** (letting the latter in makes prefill succeed
and moves the `GGML_ABORT` to the first decode token). See ADR-0006.

**Split-KV / flash-decoding is already active** for decode: `launch_fattn` (`fattn-common.hpp`) computes
`parallel_blocks` at runtime and runs `flash_attn_combine_results` when > 1, so deep-context decode is split
across work-groups. The turbo-vs-f16 gap at depth is therefore per-element **dequant throughput**, not
occupancy. (There is a `// todo` / commented `parallel_blocks = ntiles_KQ` knob near line 1150 tuned for f16.)

**Adaptive modes** (`src/llama-kv-cache.cpp`): layer-adaptive quant selection. Mixed turbo-K/turbo-V of
*different* bit-widths used to abort in SYCL FA, so a symmetric "mode 8" keeps every layer same-typed.
Since ADR-0007 the pairs `turbo4-K x {turbo3,turbo2}-V` are curated and work; every other mixed width
still aborts. Note **mode 8 overrides `--cache-type-v` to turbo2** on non-boundary layers regardless of
what was requested (it only auto-enables when `-ctv` is already turbo2, so the default is unaffected). Context-shift is **gracefully disabled for turbo** (`get_can_shift()` returns false) because turbo
K-shift is unimplemented and there is no SYCL turbo<->f32 cast - it falls back instead of crashing.

## Performance picture (honest, Arc B580)

- **Prefill:** at f16 parity, symmetric *and* asymmetric. Asymmetric `q8_0-K x turbo3-V` was the gap - it
  missed Option A entirely until 2026-07-28 and ran at ~12% of f16; now 2540/1766/798 t/s at pp512/2048/8192
  vs f16's 2582/1769/798 (B580, AOT, Qwen3-4B Q4_K_M), i.e. 4.0x/5.9x/8.3x faster than before.
- **Decode @ shallow context:** near-fp16 (turbo3 tg128 70.4 vs f16 75.6).
- **Decode @ depth:** collapses (turbo3 ~3.4x slower than f16 @32k) - but `q8_0` collapses *identically*
  (turbo3 actually beats it), so it is not a turbo defect. Turbo delivers q8_0-class decode speed at up to
  2.7x less memory than q8_0 - that is the win. **Do not repeat the old "inherent to quantized KV" framing:**
  f16 decode routes to the GQA-batched TILE (one K/V dequant shared across the GQA group) while every
  quantized type is forced onto VEC with `ncols2` hardcoded to 1 (`fattn-vec.hpp:632`), so VEC re-dequantizes
  each row once per query head (~4x on GQA 4:1). Structural and addressable, but gated on GRF pressure - see
  ADR-0006 "Adiado".
- **KV memory @64k:** turbo2 5.6x, turbo3 5.1x, turbo4 3.8x less than fp16 (q8_0 only 1.9x).

## Key decisions and why (full ADRs in `docs/pt-BR/decisions.md`)

- **ADR-0001 - upstream sync via `merge`, not `rebase`.** The fork is ~230 custom commits over a merge-base
  2.5 months behind; rebasing would replay each turbo commit against 1075 upstream commits and reconflict the
  same hot files dozens of times. A single merge commit resolves each file once and preserves turbo history.
- **ADR-0003 - `ggml_type` enum slot 42.** Upstream took `Q2_0 = 42`, which the fork used for turbo. Chose
  **Option A**: align with upstream (`Q2_0=42`, `TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47`).
  Why: turbo2/3/4 are *runtime KV-cache* types (never written to GGUF) so renumbering them is free; no TQ-weight
  GGUFs exist. Zero permanent divergence -> future syncs never reconflict this slot. Numbering must stay
  identical across `ggml/include/ggml.h`, `gguf-py/gguf/constants.py`, and `ggml.c` (designated initializers by name).
- **ADR-0004 - `GGML_SYCL_FA_ALL_QUANTS` stays OFF.** Upstream now defines it by default, which odr-uses turbo
  FA-vec instances (`D>=256`) that don't exist (turbo GRF-spills at D>=256 on Intel) -> LNK2019. The fork keeps
  the curated `#else` FA path (F16/Q4_0/Q8_0 + curated turbo combos). One documented 1-line divergence.
- **ADR-0005 - asymmetric f16-K + turbo-V.** "V is free, K is everything." Added 3 curated
  `FATTN_VEC_CASES_TURBO_D(F16, TURBO{2,3,4}_0)` rows (cap D in {64,128}) + one instance file for the single
  extern-declared combo `(F16, TURBO3_0)`; turbo2/turbo4 instantiate implicitly. No graph edits needed (see
  rotation gating above). FA_ALL_QUANTS stayed off; build links clean.
- **Locked engineering stance** (project memory): retire native turbo-TILE decode (all turbo decode via VEC,
  mirroring CUDA structure); force symmetric turbo where mixed-width would abort; **defer InnerQ** (default-off,
  inert on SYCL - pin OFF for parity captures); target native **sub-group 16** with `reqd_sub_group_size(16)` as
  the warp analogue (SG32 opt-in, capability-gated, never hard-pinned).
- **Method rule:** validate turbo numerics against a **CPU golden in the rotated domain** before and after any
  FA or WHT change. The one pre-existing FA turbo test only asserted finiteness, which hid structural bugs
  (double-rotation, half2-V misread) - a finiteness smoke test is not a parity test.

## Source map (turbo-relevant)

| Area | Files |
|------|-------|
| KV-cache integration, adaptive modes, shift guard | `src/llama-kv-cache.cpp` |
| SYCL FA dispatch + turbo prefill (Option A) | `ggml/src/ggml-sycl/fattn.cpp` |
| VEC decode kernel + launch/split-KV | `ggml/src/ggml-sycl/fattn-vec.hpp`, `fattn-common.hpp` |
| f16 TILE prefill kernel | `ggml/src/ggml-sycl/fattn-tile.hpp` |
| Turbo primitives (centroids, dequant, WHT) | `ggml/src/ggml-sycl/turbo-quants.hpp`, `turbo-wht.cpp` |
| Cache-write rotate+quantize | `ggml/src/ggml-sycl/set_rows.cpp` |
| Block layouts / codebooks / type names | `ggml/src/ggml-common.h`, `ggml/src/ggml.c`, `ggml/include/ggml.h` |
| CLI type parsing (`turbo2`/`turbo3`/`turbo4`) | `common/arg.cpp` |
| Tests / gates | `tests/test-sycl-turbo.cpp`, `tests/test-e2e-turbo-kv.sh`, `scripts/turbo-quality-gate.sh` |
| CI/CD (Windows+Linux SYCL, release packaging) | `.github/workflows/tqp-sycl.yml` |
| Engineering deep-dive / ADRs / benchmarks | `docs/en/architecture.md`, `docs/{en,pt-BR}/decisions.md`, `docs/{en,pt-BR}/benchmarks.md` |
| Session handoff (read at session start) | `TURBO_HANDOFF.md` |

## Running turbo

```
llama-server -m model.gguf -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v turbo3 -c 32768
```

**Quality, measured 2026-07-28** (wikitext-2, Qwen3-4B Q4_K_M, `-c 512 --chunks 32`, first ever run of the
perplexity gate): turbo on **V** costs ~0.4%; turbo on **K** costs ~30%. PPL vs f16 9.054 - `q8_0 x turbo3`
9.091 (+0.4%), `f16 x turbo3` 9.104 (+0.6%), `turbo4 x turbo4` 9.511 (+5.1%), `turbo3 x q8_0` 11.781
(+30.1%), `turbo3 x turbo3` 11.920 (+31.7%), `turbo2 x turbo2` 12.821 (+41.6%). So **never recommend
symmetric turbo2/turbo3**; `-ctk q8_0 -ctv turboN` is the config. Note the golden tests are structurally
blind to this (they quantize both sides identically, so quant error cancels) - only perplexity sees it.

Accepted `--cache-type-k` / `-v`: `turbo2`, `turbo3`, `turbo4` (not `turbo2_0`). **Requires `--flash-attn on`**
(bare `--flash-attn` without a value now errors upstream). Only the turbo build parses these type strings. On
Intel GPUs set `SYCL_CACHE_PERSISTENT=1` once so the JIT caches kernels to disk (first launch compiles all
kernels and otherwise looks hung / can trip a wrapper's `/health` timeout). Recommended prod default is
`-ctk q8_0 -ctv turbo3` (protect K at 8-bit, compress V).
