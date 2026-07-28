# UPSTREAM SYNC — 2026-07

> Hot-zone map + deterministic turbo re-integration playbook.
> Companion to the ADRs in `docs/en/decisions.md`. Read both before resuming the sync.

## References

| Item | Value |
|---|---|
| fork↔upstream merge-base | `7fc1c4ef` (2026-04-21) "metal: workaround macOS GPU watchdog" |
| upstream/master target | `fb30ba9a6` (2026-07-09 10:15 -0700) "hexagon: tiling..." |
| canonical fork HEAD | `feature/turboquant-kv-cache` @ `5874b9fe1` (2026-07-09) |
| sync branch | `sync/upstream-2026-07` @ `5874b9fe1` (CLEAN — trial-merge aborted) |
| commits fork is ahead | 230 |
| commits upstream is ahead | 1075 |
| strategy | **merge** (ADR-0001) |
| overall state | **RESOLVED + VALIDATED (SYCL)** — 31 conflicts resolved, merge done, SYCL build green, golden gate green. See "Execution result" below. |

## How to resume (next session, WITH the oneAPI toolchain)

```bash
git switch sync/upstream-2026-07
git merge upstream/master        # recreates the 31 conflicts below
# resolve file-by-file per the playbook (the "Map" section)
# build SYCL (see "Build/validation") -> run tests/test-sycl-turbo + turbo-quality-gate.sh
# only then: git commit (the merge)
```
Bail out at any point with: `git merge --abort`.

## Key structural finding

- **The enum collision is only at `ggml_type = 42`** (upstream's Q2_0 vs. turbo). See ADR-0003 —
  **owner decision was pending** at the time this was written. `llama_ftype` does NOT collide
  (take-both).
- **`ggml.c`'s type_traits uses designated initializers by name** → auto-merged correctly; it
  compiles as soon as `ggml.h`'s enum is resolved to give each name a distinct slot.
- **`llama-graph.cpp` (+912 upstream churn) auto-merged textually clean**, all 42 turbo references
  intact including the inverse-WHT on the FA output (`ggml_turbo_wht`, ~line 2421). Risk is
  **silent/semantic** — only a build confirms it.
- **SYCL is the backend that matters** (Intel Arc B580; see `TURBO_HANDOFF.md`). CUDA/Metal/Vulkan
  are best-effort, not validated on Arc → lower priority in the sync.

## Hot-zone map (31 conflicting files + silent auto-merges)

Risk legend: **LOW** trivial/mechanical · **MEDIUM** surgical re-integration · **HIGH**
rewrite/mandatory validation.
Priority: **P0** SYCL+types (turbo core on Arc) · **P1** llama KV/graph · **P2** CUDA/Metal/Vulkan
(non-Arc) · **P3** infra/tests/non-turbo.

### P0 — Type foundation (all backends)

| File | Conflicts | Risk | Turbo logic at risk / recipe |
|---|---|---|---|
| `ggml/include/ggml.h` | 1 | LOW* | `ggml_type` enum: collision at slot 42. *Mechanically low risk but **gated by ADR-0003**. Apply the chosen numbering. |
| `gguf-py/gguf/constants.py` | 3 | LOW | Mirror `ggml.h`'s numbering in `GGMLQuantizationType` + add Q2_0 and TQ3_1S/TQ4_1S to the block-size map (`Q2_0:(64,2+16)`, `TQ3_1S:(32,2+2+12)`, `TQ4_1S:(32,2+2+16)`). `LlamaFileType.MOSTLY_*` = take-both. |
| `include/llama.h` | 1 | LOW | `llama_ftype`: take-both. Q1_0=40, Q2_0=41 (upstream), TQ3_1S=43, TQ4_1S=44 (ours). No renumbering (42 stays free). |
| `ggml/src/ggml.c` | 0 (auto) | LOW | Auto-merged. Just verify that `[GGML_TYPE_Q2_0]` AND `[GGML_TYPE_TURBOx]/[TQx]` coexist in `type_traits` and in the `quantize` switch (~lines 684/767-799/7830-7858). Compiles once the enum is resolved. |
| `ggml/src/ggml-common.h` | 0 (auto) | LOW | Auto-merged. Turbo blocks (`block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s`) intact alongside upstream's `block_q2_0`. Check the static_asserts. |

### P0 — SYCL backend (turbo core, Intel Arc)

| File | Conflicts | Upstream churn | Risk | Turbo logic at risk / recipe |
|---|---|---|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 7 | +2017 | MEDIUM | (a) includes: take-both (`turbo-wht.hpp` + conv2d*). (b) device-enum refactor (`info.device_count`→`info.devices[i]`, vmm/usm/hw_info): **take upstream** — our side is an old version without turbo. (c) 3 turbo type guards (dmmv exclusion ~L4414; `mul_mat_id` support for TQ3/TQ4 ~L5736; op-support type list ~L5818): **re-add the turbo types to upstream's NEW lists**, merging the two lists (turbo + Q1_0/MXFP4/NVFP4). |
| `ggml/src/ggml-sycl/mmvq.cpp` | 1 | +1538 | HIGH | turbo mmvq (TQ weight dp4a). Upstream rewrote almost everything around it. Re-hook the turbo path onto the new dispatch; **a build is mandatory**. |
| `ggml/src/ggml-sycl/cpy.cpp` | 2 | +706 | HIGH | turbo cpy (KV set-rows-like). Re-integrate onto the new cpy structure. |
| `ggml/src/ggml-sycl/convert.cpp` | 1 | +83 | MEDIUM | Turbo dequant registration. Add turbo cases to the new table. |
| `ggml/src/ggml-sycl/set_rows.cpp` | 1 | +20 | MEDIUM | Turbo set_rows (WHT). Small conflict. |
| `ggml/src/ggml-sycl/dmmv.cpp` | 0 (auto) | +916 | MEDIUM(silent) | Auto-merged despite +916. **Verify by building** that the turbo dmmv hooks survived semantically. |
| `turbo-quants.hpp`, `turbo-wht.cpp/.hpp`, `fattn*.hpp/.cpp`, `fattn-vec-instance-*tq3*` | 0 | new/low | LOW(silent) | New or barely-touched files. No textual conflict. Confirm the signatures they call (e.g. `fattn-common.hpp` upstream +12) haven't changed. |

### P1 — llama KV cache + graph

| File | Conflicts | Upstream churn | Risk | Turbo logic at risk / recipe |
|---|---|---|---|---|
| `src/llama-kv-cache.cpp` | 4 | +324 | HIGH | (1) ~L60 `ggml_mul_mat_aux` + cross-TU InnerQ state (per-channel equalization, CUDA): upstream has nothing here → **keep ours**. (2) ~L142 the **auto-asymmetric turbo-K** block (GQA≥6 → upgrade K to q8_0) plus upstream's `n_layer`→`n_layer_all`/`n_layer_kv` refactor: **preserve the turbo block, adapt it to the new vars**. (3) ~L187 `mem_size` **+3 turbo tensors** (rotation + rotation_inv + innerq_scale_inv): preserve the +3, adapt to `n_layer`. (4) ~L534 the turbo attention-rotation policy (#21038 OFF by default in the fork): preserve the decision. |
| `src/llama-graph.cpp` | 0 (auto) | +912 | MEDIUM(silent) | Auto-merged clean, 42 turbo references OK (inverse-WHT on the FA output, InnerQ scale). **Validate by building** that upstream's `build_attn`/shapes refactor didn't break the turbo hooks. |
| `src/llama-context.cpp` | 2 | — | LOW-MEDIUM | Check the turbo/KV type handling. |
| `src/llama-model-loader.cpp` | 1 | — | LOW | Turbo type registration/load. |

### P2 — CUDA (non-Arc, best-effort; needs to compile only if a CUDA build is attempted)

| File | Conflicts | Risk | Recipe |
|---|---|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | 4 | MEDIUM | `switch` on `K->type` with TURBO2/3/4 cases (D%64 guard) → upstream extracted a `ggml_cuda_fattn_kv_type_supported()` helper. **Add the turbo types to the new helper** (or keep the turbo guard ahead of it). |
| `ggml/src/ggml-cuda/fattn-common.cuh` | 1 | MEDIUM | Our HIP fix (bypass the mem-pool for f16 temp, avoiding OOM with quantized KV) vs. upstream's `ggml_cuda_flash_attn_ext_get_f16_extra_data` refactor. Re-integrate the HIP fix onto the new structure. |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 1 | MEDIUM | MMA config table: upstream added a full set (112..576). Take upstream + check whether our custom 640/512 config is still needed. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 4 | MEDIUM | `is_tq_weight` guard (TQ4_1S/TQ3_1S → fused dp4a path, outside mmvq/mmq). Re-add it to the refactored dispatch. |
| `set-rows.cu`, `fattn-vec.cuh` | 0 (auto) | LOW(silent) | Auto-merged (ours +929/+372, upstream +88/+29). Build-verify. |

### P2 — Metal (non-Arc)

| File | Conflicts | Risk | Note |
|---|---|---|---|
| `ggml-metal-device.cpp` | 1 | MEDIUM | Turbo pipeline registration. |
| `ggml-metal-device.h` | 1 | LOW | Signature. |
| `ggml-metal-ops.cpp` | 1 | MEDIUM | Turbo/WHT op dispatch. |
| `ggml-metal.metal` | 3 | MEDIUM | Turbo kernels. (`turbo-matrices.h`, 8207 lines, + `turbo-wht.h` are new, no conflict.) |

### P2 — Vulkan (non-Arc)

| File | Conflicts | Upstream churn | Risk | Note |
|---|---|---|---|---|
| `ggml-vulkan/ggml-vulkan.cpp` | 7 | +3121 | HIGH | Massive upstream rewrite. Re-hook the turbo types. Low priority (non-Arc). |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | 2 | — | MEDIUM | Turbo shader registration in the generator. |
| `vulkan-shaders/dequant_funcs_cm2.glsl` | 1 | — | MEDIUM | Turbo cm2 dequant. |
| `vulkan-shaders/flash_attn_base.glsl` | 1 | — | MEDIUM | Turbo FA base. |
| new shaders (`dequant_turbo3_0.comp`, `dequant_tq4_1s.comp`, `mul_mat_vec_tq4_1s.comp`, `turbo_wht.comp`, ...) | 0 | — | LOW | New, no conflict. |

### P3 — Non-turbo custom + infra + tests

| File | Conflicts | Risk | Note |
|---|---|---|---|
| `common/arg.cpp` | 1 | MEDIUM | **Non-turbo:** our ngram-mod speculative decoding (`speculative.type`, `ngram_size_n`) vs. upstream's new API (`speculative.types.push_back`, `ngram_mod` struct). Adopt upstream's new API with our values. |
| `common/speculative.cpp` | 1 | MEDIUM | Same speculative-decoding feature, same story. |
| `tools/server/server-task.cpp` | 1 | LOW | Param parsing (likely the KV type string). |
| `tests/test-backend-ops.cpp` | 1 | LOW | Turbo type registration in the tests. |
| `tests/test-quantize-fns.cpp` | 1 | LOW | Same, for quantize fns. |
| `.github/workflows/build.yml` | 0* | LOW | *Listed without markers (a shared upstream change to build.yml). Our turbo CI is `tqp-sycl.yml` (SEPARATE, doesn't conflict → survives the sync). Take upstream on build.yml. |
| `.devops/nix/package.nix` | 1 | LOW | Take upstream. |
| `.gitignore` | 1 | LOW | Take-both. |

## Is turbo functionality preserved?

**Textual evidence: YES (provisional).** All turbo core symbols survive the trial-merge:
- Types (`block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s`, enums) — intact, only need renumbering at
  slot 42 (ADR-0003).
- KV turbo logic (auto-asymmetric, +3 rotation tensors, InnerQ) — conflicting but **preservable**
  (upstream didn't overwrite the semantics, just renamed `n_layer`).
- Graph inverse-WHT (`ggml_turbo_wht`) — auto-merged intact.
- SYCL turbo primitives (turbo-quants, turbo-wht, turbo FA, set_rows) — new files intact; the hooks
  in the rewritten files are surgical (type lists).

**Runtime evidence: PENDING** at the time this was written. No build/golden-test had run yet
(ADR-0002). **Do not declare turbo preserved until `tests/test-sycl-turbo` is green and
`turbo-quality-gate.sh` is within 5% PPL.**

## Remaining risks

1. **[HIGH] Unvalidated SYCL re-integration** (mmvq +1538, cpy +706, dmmv +916 auto-merged): the
   biggest source of a silent bug. Only build+golden catches it.
2. **[HIGH] `n_layer`→`n_layer_all`/`n_layer_kv` refactor** in `llama-kv-cache.cpp`: if the turbo
   block/mem-size adaptation gets the variable wrong, turbo KV allocates incorrectly → crash or
   garbage.
3. **[MEDIUM] Numbering decision (ADR-0003)** can break existing TQ GGUFs if Option A is chosen and
   such files exist.
4. **[MEDIUM] `llama-graph.cpp` semantics**: 912 lines of upstream churn around the turbo hooks —
   a clean textual auto-merge doesn't guarantee identical shapes/API.
5. **[LOW] CUDA/Metal/Vulkan**: need to compile but aren't validated on Arc; they can stay broken
   without affecting the SYCL path.

## Build / validation (next session)

Environment (from `TURBO_HANDOFF.md`): oneAPI 2026.0 (`C:\Program Files (x86)\Intel\oneAPI`), VS 2022.
```
# configure
cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
# build targets
cmake --build build --config Release -j 8 --target llama-cli llama-server llama-bench llama-quantize test-sycl-turbo
```
Known fixes for the agent shell: PATH for vswhere (`...VisualStudio\Installer`),
`set "NoDefaultCurrentDirectoryInExePath="`, `/EHsc`, run `.bat` files via **PowerShell**'s
`cmd /c`. Kill any process holding the ggml-sycl DLL before relinking.

Gate: `tests/test-sycl-turbo` (golden cosine ~0.99999) + `bash scripts/turbo-quality-gate.sh` (turbo3
PPL < 1.05× the q8_0 baseline, speed ratio > 0.95) + `.github/workflows/tqp-sycl.yml` CI green.

---

## Execution result (2026-07-09, session with the toolchain)

**Merge done and resolved.** `git merge upstream/master` (fb30ba9a6) onto `sync/upstream-2026-07`.
The 31 conflicts were resolved by re-integrating the turbo logic onto upstream's new structure
(NOT blindly taking theirs/ours). ADR-0003 = Option A applied (Q2_0=42, turbo 43–47, COUNT=48).

### SYCL build — GREEN
- Toolchain: oneAPI 2026.0 (icx/icpx), Ninja, Win10 SDK 10.0.26100. Correct recipe in
  `docs/en/testing.md`.
- `cmake --build build-sync --target test-sycl-turbo llama-cli` → **exit 0**, `ggml-sycl.dll`
  (59 MB), `test-sycl-turbo.exe`, `llama-cli.exe` linked. 0 errors.
- **3 silent auto-merge breakages that only the build caught (fixed):**
  1. `ggml.c` `static_assert(GGML_OP_COUNT == 97)` → **98** (turbo op `GGML_OP_TURBO_WHT` plus new
     upstream ops).
  2. `ggml-sycl.cpp` called `get_sycl_env(...)` → upstream renamed it to `ggml_sycl_get_env(...)`.
  3. **`GGML_SYCL_FA_ALL_QUANTS`**: upstream started `#define`-ing it in `common.hpp` (the fork had
     it OFF). This activated the FA-vec branch that odr-uses `TURBO3_0 × {F16,Q4_0,Q4_1,Q5_0,Q5_1}`
     and `TURBO3_0 @ D256/512` — combinations with an `extern template` declaration but WITHOUT an
     explicit instance (only `tq3-tq3`, `q8_0-tq3`, `tq3-q8_0` @ D=64/128 exist; turbo GRF-spills at
     D≥256 on Intel). → LNK2019.
     **Fix:** keep `GGML_SYCL_FA_ALL_QUANTS` OFF (the fork's validated config; restores the curated
     FA `#else` path). Marked `// TURBO BEGIN/END` in `common.hpp`. Cost: SYCL FA covers
     F16/Q4_0/Q8_0 + turbo (not standard Q4_1/Q5_0/Q5_1 — which the fork never supported). Re-enabling
     ALL_QUANTS requires generating the full turbo instance matrix first.

### Turbo gate — GREEN (real evidence, Arc B580)
`build-sync/bin/test-sycl-turbo.exe` under the oneAPI runtime → **exit 0**, all PASSED:
- TURBO2/3/4_0 quant: MSE 0.0, **cosine 1.000000**.
- TQ3_1S weight mul: cosine 1.000000 · TQ4_1S: cosine 0.999843.
- `GGML_OP_TURBO_WHT` fwd + fwd→inv round-trip (gs=32/64/128): PASSED, max|diff| ~3e-7.
- Inverse-WHT value parity dev-vs-cpu (gs=64/128): max|dev-cpu| = 0.0.
- set_rows multi-group TURBO2/3/4 (ne00=1024): cosine 1.000000.
- FA turbo golden parity TURBO2/3/4 (D=128, n_q=1/8): cosine 0.999986–1.000000.
- FA turbo DECODE+GQA+padding-mask TURBO2/3/4: cosine 1.000000.
- Flash-attention with a TURBO3_0 KV cache: `graph_compute` status 0, PASSED.

### Honest pending items
- **`turbo-quality-gate.sh` (end-to-end PPL)**: NOT run — neither the validated model (pure-attention
  Qwen3-4B) nor wikitext are present on this machine; the only local `.gguf` is a hybrid model
  (Gated Delta Net, turbo not validated on it by design). The golden `test-sycl-turbo` covers turbo
  numerical parity directly on the GPU (quant+WHT+FA+KV). Run the PPL gate once the pure-attention
  model is available.
- **Bench with no regression >5%**: not measured this session (same reason — needs the reference
  model).
- **CUDA/Metal/Vulkan**: resolved (0 markers) by sub-agents, **not compiled** (SYCL-only build).
  Vulkan: turbo3 FA on the scalar/coopmat1 paths became non-functional post-merge (upstream moved
  the FA dequant to a new `flash_attn_dequant.glsl`, with no TURBO3_0 case — outside the scope of
  the 4 files the agent touched); CM2 turbo3 preserved. Best-effort non-Arc; follow-up documented.

## Turbo inline touch-point checklist (goal: a future `git merge upstream` with near-zero conflict)

Upstream files with inline turbo hooks (re-apply/verify on every sync). Marked
`// TURBO BEGIN/END` where practical. `file:symbol` → what to preserve:

| File | Symbol / location | Turbo hook to preserve |
|---|---|---|
| `ggml/include/ggml.h` | `enum ggml_type` | turbo slots 43–47 (ADR-0003, Option A); realign if upstream occupies 43+ |
| `ggml/src/ggml.c` | `type_traits[]`, `GGML_OP_NAME/SYMBOL[]`, `static_assert(GGML_OP_COUNT==N)` | turbo entries + the TURBO_WHT op; **bump the assert** |
| `ggml/src/ggml-common.h` | `block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s` | structs + static_asserts |
| `gguf-py/gguf/constants.py` | `GGMLQuantizationType`, `LlamaFileType`, block-size map | mirror the numbering |
| `include/llama.h` | `llama_ftype` | `MOSTLY_TQ3_1S=43/TQ4_1S=44` (take-both) |
| `ggml/src/ggml-sycl/common.hpp` | `GGML_SYCL_FA_ALL_QUANTS` | keep **OFF** (otherwise LNK2019 in turbo FA-vec) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | `ggml_sycl_tq_convert_q8` (TQ_NATIVE), dmmv exclusion, MUL_MAT TQ guard, SET_ROWS list, `GGML_OP_TURBO_WHT` dispatch+support | re-add to the new lists/dispatch |
| `ggml/src/ggml-sycl/{mmvq,cpy,convert,set_rows,dmmv}.cpp` | dispatch by type | turbo cases (take-both with upstream's new types) |
| `ggml/src/ggml-sycl/fattn.cpp` | `ggml_sycl_flash_attn_ext_vec` (`#else` branch) | `FATTN_VEC_CASES_TURBO_D` combinations (turbo×turbo, q8_0×turbo, turbo×q8_0 @ D64/128) |
| `ggml/src/ggml-sycl/fattn-vec.hpp` | `EXTERN_DECL_FATTN_VEC_CASES(*, TURBO3_0)` | turbo3 extern declarations |
| `src/llama-kv-cache.cpp` | `ggml_mul_mat_aux`+InnerQ state, auto-asymmetric K, `+3` mem_size, rotation policy (`if(other)`/DeepSeek), adaptive-mode (`hparams.n_layer()`) | preserve; adapt to upstream renames |
| `src/llama-context.cpp` | turbo K/V head-dim padding (2×) | uses `hparams.n_layer()` (used to be a field) |
| `src/llama-graph.cpp` | `ggml_turbo_wht` fwd/inv, `get_turbo_innerq_scale_inv` | inverse-WHT on the FA output (auto-merge; validate by building) |
| `src/llama-model-loader.cpp` | `llama_ftype_name` | `TQ3_1S/TQ4_1S` cases in the new prefix style |
