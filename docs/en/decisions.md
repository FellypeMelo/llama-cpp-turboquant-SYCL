# DECISIONS (ADR) — TurboQuant SYCL fork

> Architecture decision log. Short ADR format.

---

## ADR-0001 — Upstream sync strategy (2026-07): MERGE, not rebase

**Date:** 2026-07-09
**Status:** Accepted (partial — sync not yet complete at the time of writing, see `docs/en/upstream-sync.md`)
**Branch:** `sync/upstream-2026-07` (created from `feature/turboquant-kv-cache` @ `5874b9fe1`)

### Context
- merge-base fork↔upstream = `7fc1c4ef` (2026-04-21).
- Target: `upstream/master` = `fb30ba9a6` (2026-07-09).
- Fork is **230 commits** of custom work ahead of the merge-base. Upstream is **1075 commits** ahead. Gap ~2.5 months.
- Upstream heavily rewrote the areas where TurboQuant lives (especially the **SYCL** backend: `ggml-sycl.cpp` +2017 lines, `mmvq.cpp` +1538, `dmmv.cpp` +916, `cpy.cpp` +706; plus `llama-graph.cpp` +912, `llama-kv-cache.cpp` +324, Vulkan `ggml-vulkan.cpp` +3121).
- Owner's rule: preserve turbo functionality above everything else; resolving a conflict means **re-integrating** the turbo logic on top of the new base, not blindly taking "theirs".

### Decision
**Use `git merge upstream/master` (a single merge commit), NOT `git rebase` of the 230 commits.**

Reasons:
1. Rebasing 230 commits replays each turbo commit against 1075 upstream commits → repeated conflicts in the SAME file dozens of times (e.g. every commit that touched `ggml-sycl.cpp` reconflicts). A merge resolves each file **once**, as a whole.
2. A merge preserves the turbo history intact (it does not rewrite the 230 commits) — important for auditability and for the existing CI/release setup.
3. Turbo re-integration needs visibility into the FINAL state of each file (ours vs. theirs vs. base-3), which the merge's diff3 view gives directly.

Trade-off accepted: the history ends up with one "large" merge commit; mitigated by this ADR plus `docs/en/upstream-sync.md` documenting exactly what came in.

---

## ADR-0002 — This session stops at the trial-merge + playbook; branch stays CLEAN (merge aborted)

**Date:** 2026-07-09
**Status:** Accepted

### Context
- Session scoped as **preparation and risk analysis**, explicitly NOT the full blind rebase.
- Trial-merge executed: **31 files in conflict** (bounded, characterized — see the map in `docs/en/upstream-sync.md`). Many hot files auto-merged textually clean (`ggml.c`, `llama-graph.cpp`, `ggml-common.h`).
- **SYCL toolchain unavailable this session:** oneAPI 2026.0 is installed at `C:\Program Files (x86)\Intel\oneAPI` but `icx`/`icpx` are not on PATH (setvars not sourced), and in any case **you cannot build a mid-merge tree** with 31 open conflicts.
- Owner's rule: **do not declare success without validation**; losing turbo is total failure.

### Decision
After the trial-merge characterized the risk, run **`git merge --abort`** — leave the working tree CLEAN (branch `sync/upstream-2026-07` back at `5874b9fe1`), and deliver a **deterministic playbook** for the resolution in `docs/en/upstream-sync.md`.

Reasons (a deliberate senior-engineering call):
1. **No resolution of mine would be validatable in this session** (no build). Baking in semantic turbo decisions (re-integrating SYCL dispatch, the `n_layer` refactor, type guards) unvalidated, in a partial tree that would sit for days = risk of a silently inherited bug — exactly what the owner forbids.
2. **A mid-merge tree blocks the main worktree**: git refuses to switch branches with unmerged paths. Leaving the owner's repo stuck in a 1075-commit merge for days is hostile to normal use.
3. **Nothing is lost:** the merge is reproducible in one command (`git merge upstream/master`), and the playbook gives a file-by-file recipe. The requested "partial state" is represented by the **branch created + the exhaustive playbook**, not by a fragile, conflicted tree.

Consequence: the actual resolution (+ build + golden test) is left for the **next session with the toolchain available**. The build/test gate = **PENDING** (not declared green).

---

## ADR-0003 — `ggml_type` enum collision at slot 42 (Q2_0 vs. TurboQuant): RESOLVED, Option A

**Date:** 2026-07-09 (accepted 2026-07-09, sync execution session)
**Status:** **Accepted — Option A** (owner confirmed: no turbo GGUFs exist that need to remain loadable)

### Owner's decision
Option A chosen. Align with upstream: `Q2_0=42`, and move the turbo types to 43–47:
`TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47, GGML_TYPE_COUNT=48`.
Reason: no GGUF files quantized as TQ3_1S/TQ4_1S are in use (the validated path is Q4_K_M weights + runtime turbo KV). Zero permanent numbering divergence → future syncs won't reconflict on Q2_0. The cost (renumbering TQ3_1S/TQ4_1S IDs) is harmless since no files are affected.

Applied in: `ggml/include/ggml.h`, `gguf-py/gguf/constants.py` (`GGMLQuantizationType` + block-size map), consistent with `ggml.c`/`ggml-common.h` (designated initializers by name). `include/llama.h` / `LlamaFileType` = take-both (no renumbering; 42 stays free in the ftype space).

---

### Context (original)

**Date:** 2026-07-09
**Original status:** Proposed (needed owner confirmation before finalizing the sync)

### Context
The sync's only *hard* numbering collision:
- **Upstream** added `GGML_TYPE_Q2_0 = 42` (a new weight quant), `GGML_TYPE_COUNT = 43`.
- **Fork** already used `42..46` for turbo: `TURBO2_0=42, TURBO3_0=43, TURBO4_0=44, TQ3_1S=45, TQ4_1S=46`, `COUNT=47`.
- (`llama_ftype` / `GGMLQuantizationType.MOSTLY_*` do NOT collide: the fork skipped 41/42 and used 43/44 → clean take-both. The collision is only in the `ggml_type` enum and its `GGMLQuantizationType` mirror.)
- `ggml.c` uses designated initializers `[GGML_TYPE_XXX] = {...}` (by name, not position) → compiles correctly with any numbering, as long as the names exist and `COUNT` is large enough.

### Decision point (the hinge)
**Does any `.gguf` file already quantized as TQ3_1S/TQ4_1S need to remain loadable?**
(turbo2/3/4 are **runtime KV-cache** types, they do NOT go into the GGUF → renumbering is harmless for them. Only TQ3_1S/TQ4_1S are weight types written to the GGUF.)

- **If NO** (likely — the validated path uses Q4_K_M weights + turbo KV; TQ weight quant is experimental) → **Option A (recommended): align with upstream.**
  `Q2_0=42, TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47, COUNT=48`.
  Advantage: zero permanent numbering divergence → future syncs won't reconflict on Q2_0.
  Cost: TQ3_1S/TQ4_1S IDs change (45→46, 46→47) — breaks existing TQ GGUFs (assumed not to exist).

- **If YES** → **Option B: preserve turbo IDs.**
  `TURBO2_0=42..TQ4_1S=46` (unchanged), `Q2_0=47`, `COUNT=48`.
  Advantage: existing turbo GGUFs stay valid.
  Cost: permanent divergence of Q2_0 vs. upstream → every future sync reconflicts on that slot; the fork's Q2_0 GGUFs become incompatible with upstream (Q2_0 is new/unused in the fork, so the cost is low).

Whichever is chosen, **the numbering must stay identical** across: `ggml/include/ggml.h`, `gguf-py/gguf/constants.py` (`GGMLQuantizationType` + block-size map), and consistent with `ggml.c`. `include/llama.h` and `LlamaFileType` = take-both (no renumbering).

**Recommendation:** Option A, unless the owner confirms there are TQ GGUFs in use.

---

## ADR-0004 — `GGML_SYCL_FA_ALL_QUANTS` stays OFF in the fork (turbo FA link)

**Date:** 2026-07-09 (sync execution session)
**Status:** Accepted

### Context
Upstream started `#define`-ing `GGML_SYCL_FA_ALL_QUANTS` by default in `ggml-sycl/common.hpp` (enables
all quant types in SYCL flash-attention). The fork never defined this macro; its validated config
uses the FA-vec `#else` branch (F16/Q4_0/Q8_0 + curated turbo combinations).

Turning the macro ON activates the branch that does an **odr-use** of
`ggml_sycl_flash_attn_ext_vec_case<D, TURBO3_0, V>` for every `V ∈ {F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0}` and
for `D ∈ {256,512}`. Those combinations have `extern template` declarations in `fattn-vec.hpp` (which
suppress implicit instantiation) but only 3 explicit instances exist (`tq3-tq3`, `q8_0-tq3`,
`tq3-q8_0`, at D=64/128 — turbo GRF-spills at D≥256 on Intel, see `FATTN_VEC_CASES_TURBO_D`). Result
with the macro ON: **LNK2019 unresolved externals** when linking `ggml-sycl.dll`. (turbo2/turbo4 link
fine via implicit instantiation — they have no extern declaration.)

### Decision
Keep `GGML_SYCL_FA_ALL_QUANTS` **disabled** (commented out, marked `// TURBO BEGIN/END` in
`common.hpp`). This restores the fork's validated FA `#else` path.

- **Cost:** SYCL FA covers KV F16/Q4_0/Q8_0 (standard) + turbo2/3/4 — not standard Q4_1/Q5_0/Q5_1.
  The fork never supported those in FA (not a regression versus the fork's own baseline).
- **Divergence from upstream:** 1 line, marked and documented (reconflicts trivially on future syncs).
- **Reverting (re-enabling ALL_QUANTS)** requires generating the full matrix of turbo FA-vec
  instances (turbo × every type, D=64..512) — large, and most of those combinations never occur at
  runtime (turbo K only pairs with turbo V or q8_0). Out of scope until there is a real need.

**Evidence:** with the macro OFF, the SYCL build links (exit 0) and `test-sycl-turbo` passes green on
the Arc B580.

---

## ADR-0005 — Asymmetric f16-K + turbo-V: 3 curated rows + 1 instance file (FA_ALL_QUANTS stays OFF)

**Date:** 2026-07-09
**Status:** Accepted
**Branch:** `feature/asymmetric-turbo-kv` (from `feature/turboquant-kv-cache` @ `5602f58d0`)

### Context
Asymmetry = precise K (q8_0 or f16) + turbo V ("V is free, K is everything"). The recommended
production default (`-ctk q8_0 -ctv turboN`) already dispatched in the fork (the `Q8_0 x TURBOx`
rows in `fattn.cpp`). The genuinely NEW config is `f16`-K + turbo-V (maximum-precision K, no
quantization loss). The router (`ggml_sycl_get_best_fattn_kernel`) already routed this pair to
`BEST_FATTN_KERNEL_VEC` (the `KV_is_turbo` gate only checks whether EITHER side is turbo), but the
dispatch in `ggml_sycl_flash_attn_ext_vec` had no `FATTN_VEC_CASES_TURBO_D(F16, TURBOx_0)` rows →
it fell into the `GGML_ABORT` "Not match KV type in vec" (`fattn.cpp:166`), a reachable runtime
abort, not just a defensive assert.

### Decision
Add the 3 curated rows `FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16, GGML_TYPE_TURBO{2,3,4}_0)` to the
`#else` block (marked `// TURBO BEGIN/END`), cap D at {64,128} like the other turbo-V rows, plus ONE
instance file `template-instances/fattn-vec-instance-f16-tq3.cpp` for the one extern-declared
combination: `(F16, TURBO3_0)` at D=64/128 (mirrors `q8_0-tq3.cpp`).

- `(F16, TURBO2_0)` and `(F16, TURBO4_0)` are instantiated IMPLICITLY in `fattn.cpp` (turbo2/turbo4
  don't appear on the type_V axis of `EXTERN_DECL_FATTN_VEC_CASES`) → no instance file needed for
  those.
- Rotation semantics were already correct in the graph: the Q-WHT forward gates on `k->type` being
  turbo, and the inverse-WHT gates on `v->type` being turbo (`llama-graph.cpp`), so K=f16 leaves Q
  NOT rotated while V=turbo still applies the inverse-WHT on the output. No graph edits needed.
- `GGML_SYCL_FA_ALL_QUANTS` stays **OFF** (ADR-0004). The new rows do not require re-enabling it:
  they only need the 1 curated instance file; no LNK2019 occurred (the build links, exit 0). DLL
  relinked: build-sync (JIT) 60.3 MB; build-perf (AOT bmg-g21) 262.5 MB — both link cleanly.

### Evidence (Arc B580, build-sync JIT)
- Golden `test-sycl-turbo`: 34 PASSED / 0 FAIL. Asymmetric `{q8_0,f16} x {turbo2,turbo3,turbo4}`
  golden decode + DECODE/GQA/padding-mask with cosine 1.000000 (f16 rel-MSE 0.0, q8_0 rel-MSE
  ~8e-4). RED captured before the fix: `fattn.cpp:166 Not match KV type: K=f16 V=turbo3`.
- e2e `test-e2e-turbo-kv.sh`: the 6 asymmetric configs generate COHERENT text (on-topic keyword, no
  degenerate 5-gram repetition, no abort/NaN, no '?' corruption).

### Evidence (Arc B580, build-perf AOT — gentle re-verify 2026-07-09, ninja -j4 + Idle)
- Reconfirms under AOT (bmg-g21): Golden 34/34 (12 asymmetric cosine 1.000000, f16 decode
  rel-MSE ~2-3e-5, q8_0 ~5e-4; no regression across the 21 symmetric cases). e2e ctest 1/1, 9/9
  configs coherent (Qwen3-4B Q4_K_M).
- Adversarial read-only review (4 lenses): CONFIRMED_GREEN, zero refutation / zero critical issue.
- PENDING: PPL rescue on Qwen2.5-7B Q4_K_M (absolute precision at long context) — model absent from
  disk.
