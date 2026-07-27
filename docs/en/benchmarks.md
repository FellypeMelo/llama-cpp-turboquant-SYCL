# BENCHMARKS — TurboQuant SYCL fork

_Measured numbers. Nothing estimated. Read together with `docs/en/testing.md` (the build+gate recipe) and `STATE.md`._

## Perf-pass 2026-07-09 — Intel Arc B580 (Xe2 / BMG)

- **GPU:** Intel Arc B580 Graphics (level_zero, 20 EU slices, 12452 MiB).
- **Bench model:** Qwen3-4B Q4_K_M (2.32 GiB), `-ngl 99`.
- **Command:** `llama-bench -p 512 -n 128 -r 3` (average of 3 reps ± stddev; mean ≈ median, stable runs).
- **Binary's base build:** merge commit `b76ddd343`.
- **Clean GPU:** B580 idle before the bench, compute engines ~0%, no contention during the runs.

| Config | Flags | pp512 (tok/s) | tg128 (tok/s) | Golden green? | Date | GPU |
|---|---|---|---|---|---|---|
| baseline | `GGML_SYCL=ON`, F16=OFF, no AOT (JIT), DNN=ON, Release | 1246.70 ± 2.87 | 79.03 ± 0.48 | yes | 2026-07-09 | Arc B580 |
| perf-f16-aot | `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, AOT `GGML_SYCL_DEVICE_ARCH=bmg-g21`, DNN=ON, Release | 2528.80 ± 18.62 | 76.41 ± 2.34 | yes | 2026-07-09 | Arc B580 |

**Note on which build these two rows measure:** both rows in this table come from a build with `GGML_SYCL_F16` and `GGML_SYCL_DEVICE_ARCH=bmg-g21` (AOT) explicitly set as shown in the Flags column — the *baseline* row keeps those flags off, the *perf-f16-aot* row turns them on. This is a **different, separately-configured binary** from the one produced by the plain `cmake --build` in the top-level README's Quick Start (which builds with `GGML_SYCL_F16=OFF` and no AOT device-arch compile). If a number from this file is ever quoted next to a number from `docs/en/architecture.md` or the root README, check which of these two build configurations it came from before treating them as directly comparable — they are not the same binary.

### Reading the numbers

- **Prefill (pp512): +102.8% (2.03×).** A large gain. F16 turns on the FP16 matmul path on Xe2's
  XMX units, and prefill is compute-bound → it scales strongly.
- **Decode (tg128): -3.3%.** Marginal regression, within noise (the error bars nearly touch:
  perf 74.1-78.8 vs. base 78.6-79.5). Decode is memory-bandwidth-bound: F16 doesn't help there, and
  the FP16 path adds minimal overhead. In practice: decode is flat.
- **Verdict:** the perf-f16-aot config roughly doubles prefill while keeping decode practically
  unchanged.

### Golden gate (proof of correctness on the GPU)

`test-sycl-turbo` (build-perf) = **GREEN**: exit 0, 22 PASSED / 0 FAIL on the Arc B580 via level_zero,
`GGML_SYCL_DNNL: yes`. Quant cosine ~1.0, WHT round-trip MSE 0, FA turbo parity for TURBO2/3/4_0
worst cosine 0.999986-1.000000, FA DECODE+GQA+padding-mask cosine 1.000000, flash-attention with a
TURBO3_0 KV cache `graph_compute` status 0. **The F16/AOT flags do NOT corrupt turbo on Xe2.**

### Output sanity check (perf build, the highest-risk one for corruption)

`llama-cli -n 50 --seed 42` produces **coherent** text: a correct explanation of Rayleigh scattering
(why the sky is blue), no garbage/repetition (Prompt 192.5 t/s | Generation 64.1 t/s). Anti-corruption
confirmed on the build that matters. The baseline run's output was not captured in time (slow JIT
warmup on the first launch — exactly the cost AOT removes); this is not garbage, it is UNVERIFIED.
Equivalent correctness to the baseline is already covered by the golden gate.

### Method notes

- build-sync (baseline) didn't have an `llama-bench` binary; the `llama-bench` target was built in
  build-sync WITHOUT changing any flag (existing CMakeCache: Release, `GGML_SYCL=ON`, F16=OFF,
  DEVICE_ARCH empty, DNN=ON). `ggml-sycl.dll` was relinked from the same baseline sources — no
  flag/source change.
- build-perf's `ggml-sycl.dll` is ~247 MB because it embeds ALL AOT kernels for bmg-g21. The AOT link
  was very slow (~1h+ just for linking, due to the turbo flash-attention instance matrix). Trade-off:
  JIT warmup at runtime disappears, at the cost of a slow build — normal for SYCL AOT.

## turbo KV-cache — end-to-end COHERENCE matrix 2026-07-09 (Arc B580, build-perf F16+AOT)

Answers the owner's central question: **with the turbo KV cache REALLY on, does generation come out
coherent?** This is not kernel parity (that's the golden `test-sycl-turbo`, cosine ~1.0) — it is
**real, end-to-end generation** via `llama-cli`. New proof, complementary to the golden test.

- **Model:** Qwen3-4B-Instruct-2507 Q4_K_M (pure attention, 36 layers, `n_embd_head=128`).
- **GQA:** `n_head=32`, `n_head_kv=8` → **ratio 4:1**. Below the auto-asymmetric logic's threshold
  of 6 → **auto-asymmetric does NOT engage** (expected and correct; the code notes "Mistral 4:1 →
  turbo3 K works fine"). No `auto-asymmetric` log line for this model. (It would only engage on a
  model like Qwen2.5 with a 7:1 ratio.)
- **Command:** `llama-cli -ngl 99 -fa on -c 4096 --temp 0 -n 150 -st` (greedy, deterministic), prompt
  `"Explain why the sky is blue in 3 sentences."` (plus one reasoning-check prompt).
- **VRAM-KV:** exact value from the `llama_kv_cache: size = ...` log line (includes turbo's
  head_dim zero-padding).
- **Type names accepted by `-ctk`/`-ctv`:** `turbo2`, `turbo3`, `turbo4` (NOT `turboN_0`).

| `-ctk`/`-ctv` | KV @4k (MiB) | K / V (MiB) | vs. f16 | gen tok/s | pp tok/s | coherent? | aborted? | note |
|---|---|---|---|---|---|---|---|---|
| `f16`/`f16` (default) | 576.00 | 288 / 288 | 1.00× | 77.4 | 204 | **yes** (reference) | no | baseline |
| `q8_0`/`q8_0` | 306.00 | 153 / 153 | 1.88× | 73.4 | 72 | **yes** (≈ f16) | no | near-lossless quant |
| `q4_0`/`q4_0` | 162.00 | 81 / 81 | 3.56× | 73.6 | 71 | **yes** | no | |
| `turbo4`/`turbo4` | 153.00 | 76.5 / 76.5 | 3.76× | 73.6 | 70 | **yes** | no | safest turbo |
| `turbo3`/`turbo3` | 112.50 | 56.25 / 56.25 | **5.12×** | 74.8 | 77 | **yes** | no | **recommended default (best balance)** |
| `turbo2`/`turbo2` (auto) | 102.00 | 51 / 51* | **5.65×** | 74.2 | 69 | **yes** | no | mode 8 auto (q8_0 boundaries); **most compression while staying coherent** |
| `turbo2`/`turbo2` `TURBO_LAYER_ADAPTIVE=0` | 76.50 | 38.25 / 38.25 | 7.53× | 70.6 | 76 | **NO — degenerate repetition** | no | uniform 2-bit; shows WHY the auto boundary exists |
| `turbo3`-K / `q8_0`-V (mixed) | 209.25 | 56.25 / 153 | 2.75× | 73.8 | 199 | yes (this run) | no | K=turbo/V=q8_0 marked **unreliable** in code → NOT gated |
| `turbo2`/`turbo2` `TURBO_LAYER_ADAPTIVE=5` | 80.75 | 38.25 / 42.5 | — | — | — | — | **YES (abort)** | `fattn.cpp:166 Not match KV type: K=turbo2 V=turbo4` — SYCL-unsafe mixed config |

\* `turbo2` (auto) = **mode 8 boundary-symmetric-q8_0** engages on its own (log: `Boundary symmetric
q8_0 auto-enabled for turbo2-V`): the first 2 and last 2 layers use K+V=q8_0, the 32 middle layers use
K+V=turbo2. That's why the real 102 MiB is above the nominal "51+51" of pure turbo2.
`TURBO_LAYER_ADAPTIVE=0` disables the boundary → pure uniform 2-bit turbo2 → **degenerates** (repetition
loop: "…due to Rayleigh scattering of sunlight by molecules and small particles in the atmosphere."
repeated 3-4×, grammar breaks down as "blue due because"). This is the honest finding: uniform turbo2
degrades; the auto boundary (mode 8) rescues it into coherence.

### ASYMMETRIC matrix — precise K + turbo V (2026-07-09, Arc B580, build-sync JIT)

Asymmetric config = precise K (q8_0 or f16) + turbo V ("V is free, K is everything"). The recommended
production default (`-ctk q8_0 -ctv turboN`) already dispatched; the NEW one is `f16`-K + turbo-V (see
ADR-0005 and the `fattn.cpp` rows `FATTN_VEC_CASES_TURBO_D(F16, TURBOx_0)`). head_dim=128. Golden
parity proven in `test-sycl-turbo` (cosine 1.000000; f16 rel-MSE 0.0, q8_0 rel-MSE ~8e-4) and e2e
coherence across all 6 configs.

- **Golden (cosine / rel-MSE, decode n_q=1):** all cosine **1.000000**. rel-MSE: `f16`-K = **0.0**
  (bit-exact f16 dequant); `q8_0`-K = ~**8e-4** (q8_0 device-vs-CPU rounding). DECODE/GQA/mask:
  same pattern (q8_0 rel-MSE ~8e-5, f16 0.0).
- **KV @4k (MiB):** exact sum of the per-side sizes already measured in the symmetric matrix above
  (`f16`=288, `q8_0`=153, `turbo4`=76.5, `turbo3`=56.25, pure `turbo2`=38.25 per side).
- **tok/s measured (build-sync JIT, `llama-bench` -p 512 -n 128 -r 2):** `q8_0/turbo3` pp512
  **392.6**, tg128 **69.9**; `f16/turbo3` pp512 **415.8**, tg128 **70.3**. Decode is ~70 t/s,
  essentially identical between configs (shared VEC path). These are build-sync JIT numbers (a
  floor); the **validated performance headline is the build-perf F16+AOT one** (symmetric matrix:
  +102.8% prefill) — re-measuring the asymmetric configs under AOT is PENDING.

| `-ctk`/`-ctv` | KV @4k (MiB) | K / V (MiB) | vs. f16 | coherent? | aborted? | note |
|---|---|---|---|---|---|---|
| `q8_0`/`turbo3` | 209.25 | 153 / 56.25 | 2.75× | **yes** | no | **recommended production default** (K protected at 8-bit) |
| `q8_0`/`turbo4` | 229.50 | 153 / 76.5 | 2.51× | **yes** | no | K q8_0 + safest V |
| `q8_0`/`turbo2` | 191.25 | 153 / 38.25 | 3.01× | **yes** | no | K q8_0 + max-compression V |
| `f16`/`turbo3` | 344.25 | 288 / 56.25 | 1.67× | **yes** | no | **NEW** — max-precision K; Q stays un-rotated |
| `f16`/`turbo4` | 364.50 | 288 / 76.5 | 1.58× | **yes** | no | **NEW** — K f16 + turbo4 V |
| `f16`/`turbo2` | 326.25 | 288 / 38.25 | 1.77× | **yes** | no | **NEW** — K f16 + turbo2 V |

\* Rows with `turbo2`-V assume pure turbo2 per side (38.25 MiB). If turbo2-V's auto boundary mode 8
also engages in the asymmetric case (as it does in the symmetric turbo2-auto case, ~51 MiB per side),
the real V-side size would be higher — not directly re-measured in this pass (`llama-cli` in `-st`
mode doesn't print the `KV self size` line). turbo3/turbo4-V have no boundary → the VRAM figure is
exact. Coherence of all 6 was confirmed independently of the exact VRAM value.

Before the fix, `f16`-K aborted at `fattn.cpp:166 Not match KV type in vec: K=f16 V=turbo3` (RED).
NEVER enable turbo-K + turbo-V together (asymmetry = only ONE side turbo).

### Real samples (turbo KV ON)

- **f16 (A, reference):** "The sky appears blue because molecules in the Earth's atmosphere scatter
  sunlight. Shorter wavelengths of light, like blue and violet, are scattered more than longer
  wavelengths such as red and yellow. Although violet light is scattered even more than blue, our
  eyes are more sensitive to blue…"
- **turbo3 (B, headline):** "The sky appears blue because molecules in the atmosphere scatter
  shorter wavelengths of sunlight more effectively than longer wavelengths. Blue light … is
  scattered more than other colors because it has a shorter wavelength. This scattering, known as
  Rayleigh scattering, spreads the blue light in all directions, making the sky appear blue during
  the day." → **coherent and correct**.
- **turbo2 (auto mode 8):** "The sky is blue because molecules in the atmosphere scatter shorter
  wavelengths of light (like blue) more than longer wavelengths (like red). This scattering spreads
  the blue light in all directions, making the sky appear blue during the day…" → **coherent**.
- **turbo2 uniform (`TURBO_LAYER_ADAPTIVE=0`, DEGRADED):** "…The blue color of the sky is due to
  Rayleigh scattering of sunlight by molecules and small particles in the atmosphere. The blue sky
  is due to Rayleigh scattering of sunlight by molecules and small particles in the atmosphere. The
  blue color of the sky is due to Rayleigh scattering of." → **degenerate loop, do NOT use**.
- **Reasoning prompt** (60 km in 45 min → km/h): f16, turbo3 and turbo2-auto all produced the correct
  step-by-step setup (converting 45 min → 0.75 h; speed = distance/time) — coherent across all 150
  tokens.

### Verdict and OPTIMAL config

- **Yes, turbo KV generates coherently** — for the **SYCL-safe symmetric configs**: `turbo3/turbo3`,
  `turbo4/turbo4` and `turbo2/turbo2` (with the auto mode-8 boundary). Decode ~74 t/s (≈ f16's 77
  t/s, −4%).
- **OPTIMAL config (best compression while staying coherent): `-ctk turbo2 -ctv turbo2`** (auto
  mode 8) → **5.65× less KV VRAM**, coherent. If you want a larger quality margin for ~10% less
  savings, the **recommended default is `-ctk turbo3 -ctv turbo3`** (5.12×, uniform, not dependent
  on the boundary).
- **It depends on the mode:** **uniform** turbo2 (`TURBO_LAYER_ADAPTIVE=0`) degrades (repetition) —
  avoid it. `TURBO_LAYER_ADAPTIVE=5/6/7` (mixed-turbo-type boundaries) **abort** in SYCL FA-vec
  (`fattn.cpp:166`). Mixed turbo-K/q8_0-V runs but is marked unreliable in code — not recommended.
- **e2e gate:** `tests/test-e2e-turbo-kv.sh` (via ctest `test-e2e-turbo-kv`) locks in these 3
  recommended configs and asserts coherence (on-topic keyword + no degenerate repetition + no
  abort). Green on the B580.
