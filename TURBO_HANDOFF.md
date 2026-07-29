# TurboQuant SYCL — Session Handoff

_Last updated: 2026-07-29. Read this first when resuming in a new session._

_This file records **state and traps**, not a plan. It deliberately pins no commit SHA — run
`git log --oneline -15` on `feature/turboquant-kv-cache` for that. The previous version of this file
pinned one, went 23 days and ~230 commits stale, and ended up recommending the worst measured
configuration in the project._

## North star
TurboQuant KV-cache quantization working flawlessly on the **SYCL backend**, max performance on Intel
Arc. **SYCL is the only backend that matters** — CUDA/Metal/Vulkan turbo files exist as reference and
are not validated here. Target HW: Intel Arc B580 (Battlemage / Xe2, `bmg_g21`), 12 GB. Reference
model: Qwen3-4B Q4_K_M (head_dim 128, GQA 4:1, 36 layers, pure attention).

---

## The one thing to get right

```
llama-server -m model.gguf -ngl 99 --flash-attn on --cache-type-k q8_0 --cache-type-v turbo3 -c 32768
```

**Protect K at 8-bit, compress V.** Measured on wikitext-2 (Qwen3-4B Q4_K_M, `-c 512 --chunks 32`,
2026-07-28), perplexity vs f16 9.054:

| config | PPL | delta |
|---|---|---|
| `q8_0 x turbo3` | 9.091 | **+0.4%** |
| `f16 x turbo3` | 9.104 | +0.6% |
| `turbo4 x turbo4` | 9.511 | +5.1% |
| `turbo3 x q8_0` | 11.781 | +30.1% |
| `turbo3 x turbo3` | 11.920 | **+31.7%** |
| `turbo2 x turbo2` | 12.821 | +41.6% |

**Never recommend symmetric turbo2/turbo3.** Turbo on **V** costs ~0.4%; turbo on **K** costs ~30%.

This is the single most important correction in the project's history, and it arrived late: for three
weeks every document here recommended symmetric turbo3, on the strength of the e2e gate showing
coherent text. **Coherence is not quality.** A model 32% worse in perplexity still writes fluent
sentences. The golden test is structurally blind to it too — it quantizes K and V identically, so the
quantization error cancels. Only perplexity sees it.

---

## Current state

### Works, validated on the B580
- **turbo2 / turbo3 / turbo4** KV on SYCL. Golden 110/110 under both JIT and AOT; e2e 11/11 configs
  coherent.
- **Prefill at f16 parity**, symmetric *and* asymmetric. `q8_0-K x turbo3-V` does 2540/1766/798 t/s at
  pp512/2048/8192 against f16's 2582/1769/798.
- **Asymmetric KV** `{q8_0,f16} x {turbo2,turbo3,turbo4}` — all six dispatch and are gated.
- **Curated mixed-width** `turbo4-K x {turbo3,turbo2}-V` (ADR-0007). Every other mixed width aborts.
- **head_dim 64, 128, 256** for turbo; non-turbo types also reach 512.
- **KV memory @64k vs fp16:** turbo2 5.6x, turbo3 5.1x, turbo4 3.8x (q8_0 only 1.9x).

### Known limitations — read before diagnosing anything
1. ~~head_dim 64 + asymmetric KV silently runs attention on the CPU~~ — **fixed** (ADR-0011). When
   either side of the KV pair is turbo, both sides are now padded to the same multiple of 128, so
   `-ctk q8_0 -ctv turbo3` dispatches on the GPU at head_dim 64. No-op for head_dim 128 and 256.
   Covered by `tests/test-sycl-turbo-hd64.cpp`, which needs a synthetic model (`TURBO_HD64_MODEL`,
   exit 77 without it) — see that file's header for the two commands that regenerate it.
2. **Any uncovered head_dim does the same thing** (ADR-0008). `ggml_sycl_flash_attn_ext_supported` is
   just `get_best_fattn_kernel != NONE`, and ggml responds to false by scheduling the op on the CPU
   backend — not by aborting. **No gate can detect this**: golden compares values (CPU values are
   correct), e2e compares text (text is coherent), PPL compares quality (quality is unchanged).
   Before blaming turbo for slowness on a new model, check `n_embd_head_k` first.
   **It is per-cache, not per-model.** An iSWA model builds one cache per attention type and they can
   have different head_dims, so part of the layers can run on the GPU and part on the CPU. Measured
   on `gemma-4-12b` (48 layers -> caches of 40 at head_dim 256 and 8 at head_dim 512): with turbo the
   8 head_dim-512 layers fall to the CPU. Verified head_dims on the models present here:
   Qwen3-4B 128, Qwen3.5-4B/9B and Qwen3.6-28B 256, gemma-4-12b 256+512, GLM-4.7-Flash 576/512 (MLA).
   Only the first two groups run turbo on the GPU.
3. **Decode collapses at deep context** (~3.4x slower than f16 @32k). `q8_0` collapses identically, so
   it is not a turbo defect — but it is *not* "inherent to quantized KV" either. f16 decode routes to
   the GQA-batched TILE, which dequantizes each K/V row once for the whole GQA group; every quantized
   type is forced onto VEC with `ncols2` hardcoded to 1, so VEC re-reads each row once per query head
   (~4x at GQA 4:1). Structural and addressable. Deferred, gated on GRF pressure.
4. **Context shift is disabled for turbo** (`get_can_shift()` returns false) — turbo K-shift is
   unimplemented and there is no SYCL turbo<->f32 cast. It falls back instead of crashing.
5. **`--cache-reuse` does not work on M-RoPE / I-M-RoPE models** — `can_shift` returns false for
   `n_pos_per_embd() > 1`. Unrelated to turbo.
6. **The `D=64` turbo rows in `FATTN_VEC_CASES_TURBO_D` are dead code.** No turbo tensor can have 64
   columns (`blck_size` 128) and the cache always pads first. Compiled into every AOT build, never
   executed. Left in place on purpose.

### Adaptive modes
`TURBO_LAYER_ADAPTIVE` in `src/llama-kv-cache.cpp`. **Mode 8 auto-enables when `-ctv turbo2`** and
overrides `--cache-type-v` to turbo2 on non-boundary layers. Modes 5/6/7 abort in SYCL FA. Modes
5/6/7 have never been spot-checked with `-ctk turbo4`.

---

## Gates — how to run them

| gate | what it proves | command |
|---|---|---|
| **golden numeric** | CPU-golden vs SYCL kernel parity | `build-perf\bin\test-sycl-turbo.exe` (or `ctest --test-dir build -R test-sycl-turbo`) |
| **e2e coherence** | real generation is coherent | `TURBO_E2E_MODEL=/path/Qwen3-4B-Q4_K_M.gguf bash tests/test-e2e-turbo-kv.sh` |
| **quality + speed** | perplexity within 5% of q8_0 | `LLAMA=build-perf/bin bash scripts/turbo-quality-gate.sh` |
| **static lockstep** | shim allowlist == curated rows == test predicate | `python scripts/check-fattn-lockstep.py` |

All three GPU gates need the oneAPI environment sourced first, and the e2e/quality ones exit **77**
(SKIP) when the model is absent — a 77 is not a pass.

**What each gate cannot see:** golden is blind to asymmetric quality (it quantizes both sides the
same); e2e is blind to quality entirely; all three are blind to the silent CPU fallback. When
something is slow but correct, no gate will tell you — check the head_dim.

---

## Build

oneAPI at `C:\Program Files (x86)\Intel\oneAPI` (compiler `2026.0`), VS 2022. Two flavors:

- **build-sync** — JIT, `ggml-sycl.dll` ~61 MB. Fast to build, slow first launch. Use for iteration.
- **build-perf** — AOT for `bmg_g21`, DLL ~270 MB. Slow to build, fast launch. **Use for any real
  perf or PPL number.**

```
cmake -B build-perf -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx \
      -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_DEVICE_ARCH=bmg-g21
cmake --build build-perf --config Release -j 8 --target llama-cli llama-server llama-bench test-sycl-turbo
```

**Toolchain gotchas under an agent shell:** add `C:\Program Files (x86)\Microsoft Visual Studio\Installer`
to PATH (for `vswhere.exe`); `set "NoDefaultCurrentDirectoryInExePath="`; `icx` defaults to
`-fno-exceptions` so add `/EHsc` for standalone compiles; run `.bat` via the **PowerShell tool**
(`cmd /c ...`) because the Bash tool's `cmd //c` loses the system PATH.

**Before relinking, kill every `llama-cli` / `llama-server` / `llama-bench` / `test-sycl-*`** — they
hold `ggml-base.dll` and `ggml-sycl.dll` and the link fails with `LNK1104`. **Never run two GPU
processes concurrently** on the single B580.

Set `SYCL_CACHE_PERSISTENT=1` once so the JIT caches kernels to disk; otherwise first launch looks
hung and can trip a wrapper's `/health` timeout.

---

## Open work, in the order it is worth doing

1. **Upstream merge — 231 commits behind, 54 conflicting files. NOT started, and not obviously worth
   starting.** The fork carries intentional one-line divergences (`GGML_SYCL_FA_ALL_QUANTS` stays OFF,
   the curated `#else` FA path) that a merge can revert silently. What breaks is the *link* of
   instances that do not exist, so it surfaces at build time, not in any test. Merge only with time to
   fix the fallout. Use `merge`, never `rebase` (ADR-0001).
2. **VEC `ncols2` GQA sharing** — the real fix for decode-at-depth. Biggest remaining perf win.
3. **head_dim 64 asymmetric padding** (ADR-0011) — needs a head_dim-64 model on the validation
   machine. Do not attempt without one: a wrong padding produces silently wrong numbers, not an abort.
4. **`nsm = max_compute_units/16`** — B580 reports 160 CUs so `nsm` lands at 10, but BMG G21 has 20
   Xe-cores. Needs instrumentation and arch gating.
5. **Adaptive modes 5/6/7 with `-ctk turbo4`** — never spot-checked.
6. **Shared header for the FA allowlist** so the dispatch table, the prefill shim and the test
   predicate cannot drift. `scripts/check-fattn-lockstep.py` enforces this externally today.
7. **AOT kernel/DLL cost** — record it, and reconsider the dead D=64 turbo instances.
8. **XMX (`joint_matrix`) prefill — PARKED by decision.** Stage 0 PoC passed on the B580 (one fp16->f32
   DPAS tile, err 1e-10, SG16). Parked because prefill is already at f16 parity, so the payoff is
   marginal. Plan is in memory `xmx-fa-prefill.md`; worktree `G:\Programas\llama-cpp-turbo-xmx`
   (branch `feature/xmx-fa-prefill`, unbuilt).

---

## Working constraints

- **Never** `git push`, `gh pr create`, `gh pr comment` or `gh issue create` on the user's behalf, and
  never author a PR description or reviewer reply. Prepare the branch and hand over the command.
  Non-overridable (AGENTS.md).
- **No commit or push without explicit per-action human approval.** When asked to commit, use
  `Assisted-by: <assistant name>`, never `Co-authored-by:`.
- **ASCII only** in code and commit messages.
- **Test-first on real silicon.** No turbo change is done until the golden gate plus the e2e gate pass
  on the B580. "Looks coherent" is not evidence — see the perplexity story above.
- Validate turbo numerics against a **CPU golden in the rotated domain** before and after any FA or
  WHT change. A finiteness smoke test is not a parity test.

## Where things are

`CLAUDE.md` — architecture and current state, authoritative. `AGENTS.md` — rules, non-negotiable.
`docs/pt-BR/decisions.md` — ADR-0001..0011, the record of file. `docs/en/decisions.md` — ADR-0001..0005
in full, 0006..0011 condensed. `GEMINI.md` — early design playbook, **stale, not authoritative**.
