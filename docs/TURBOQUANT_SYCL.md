# TurboQuant+ on SYCL — Engineering Deep-Dive

> Bringing 2/3/4-bit rotated KV-cache quantization to Intel GPUs (Arc / Xe2) in `llama.cpp`, at up to **7.5× less KV-cache memory** than fp16 with **prefill at fp16 parity**.

This document is the engineering story behind this fork: what TurboQuant is, how it was ported to the SYCL backend, the bugs that had to be solved to make it correct, and the honest performance picture on real hardware (Intel Arc B580 / Battlemage).

**Validation hardware:** Intel Arc B580 (Battlemage / Xe2, `bmg_g21`), 12 GB. **Model:** Qwen3-4B Q4_K_M (head_dim 128, GQA 4:1, 36 layers).

---

## 1. The problem: the KV cache is the memory wall

For long-context inference the KV cache — not the weights — dominates VRAM. An fp16 KV cache for Qwen3-4B at 64k context is **9.2 GB**, which alone nearly fills a 12 GB card and leaves no room to grow context. The usual mitigation, `q8_0` KV, only halves it (1.9×).

**TurboQuant** compresses the KV cache far harder — down to ~2.1 bits/value — while keeping generation quality intact, by *rotating* the key/value vectors before quantizing them.

## 2. The core idea: rotate, then quantize

Naïve low-bit quantization of attention K/V fails because their per-channel distributions have heavy outliers — a few large components blow up the quantization scale and destroy the small ones.

TurboQuant applies a **Walsh–Hadamard Transform (WHT)** rotation to each head-dim vector first. A random-sign Hadamard rotation is (a) orthogonal (norm-preserving, so attention scores are unchanged in expectation) and (b) an *outlier smoother* — it mixes every component into every output, turning a spiky distribution into a near-Gaussian one that quantizes cleanly with a tiny fixed codebook.

The rotation is a fast in-register butterfly, O(D log D), no matrix multiply:

```
rotate(x):  x *= 1/‖x‖            # normalize
            x  = SIGNS1 ⊙ x       # random signs (fixed per position)
            x  = hadamard_butterfly(x)   # radix-2, log2(D) stages
            x *= 1/sqrt(D)        # orthonormal scale  (D=128 → 0.08838834)
            x  = SIGNS2 ⊙ x       # second random-sign layer
```

Quantized values are then just an index into a **Lloyd–Max optimal codebook** fitted to `N(0, 1/D)` (the post-rotation distribution), scaled by the stored per-vector norm.

### Where the rotation lives in the graph
Rotating inside the attention kernel for every token every step would be wasteful and easy to get wrong. Instead the rotation is threaded through the compute graph:

- **K / V:** rotated **once** at cache-write time (the SYCL `set_rows` turbo kernel rotates then quantizes; the corrected norm is stored per block).
- **Q:** pre-rotated by a dedicated graph node (`GGML_OP_TURBO_WHT`) before the attention score.
- **Output:** un-rotated by the inverse WHT after attention.

Because rotation is orthogonal, `softmax(Q·Kᵀ)` is invariant to it, so the only cost is the cheap butterfly — never a re-quantization in the hot loop.

## 3. The three formats

All three use block size 128 (= head_dim, one block per rotation group, so no redundant norms). Byte sizes below are the **`static_assert`-enforced** truth (some inline comments in the header are stale from an earlier 32-value design):

| Type | Layout (per 128 values) | Bytes | bits/val | vs fp16 |
|------|-------------------------|------:|---------:|--------:|
| `turbo2` | norm(fp16) + 2-bit indices | **34** | 2.13 | **7.5×** |
| `turbo3` | norm(fp16) + 2-bit indices + 1-bit signs (3-bit) | **50** | 3.13 | **5.1×** |
| `turbo4` | norm(fp16) + rnorm(fp16) + 4-bit PolarQuant indices | **68** | 4.25 | **3.8×** |

- `turbo2` — 4 symmetric centroids `{-0.133462, -0.039994, 0.039994, 0.133462}` (Lloyd–Max for `N(0,1/128)`). Maximum compression; quality is marginal, best on layers that tolerate it.
- `turbo3` — 8-level (2-bit magnitude + 1-bit sign). **Best balance** — near-fp16 quality at ~5× savings.
- `turbo4` — 16-level nibble-packed PolarQuant. Safest quality.

## 4. Two attention paths — and why prefill needed a different one

`llama.cpp`'s SYCL flash-attention has two kernels with opposite optimization points:

- **VEC** (`fattn-vec.hpp`) — decode-oriented, 1–2 columns per block. Dequantizes each K/V element inside the inner product.
- **TILE** (`fattn-tile.hpp`) — batch/prefill-oriented, tiled f16 GEMM-style.

Turbo KV was initially routed **exclusively to VEC**. That was correct for decode but catastrophic for **prefill**: turbo prefill ran at ~394 t/s vs 1271 for f16/q8_0 — a **3.2× gap** that widened to **7×** at 8k context, because VEC processes prefill one/two columns at a time.

### The fix — dequant-to-f16 + reuse the proven TILE kernel ("Option A")
Rather than write a fragile native turbo-TILE loader (the K/V blocks are 34/50/68-byte, non-power-of-two strides — the existing scaffolding produced garbage), the prefill path was solved by **decoupling dequant from attention**:

1. A small strided SYCL kernel dequantizes the (permuted, non-contiguous) turbo K/V views into **contiguous f16 scratch** in the compute pool.
2. The battle-tested **f16 TILE** kernel runs on that scratch.

Correct because turbo already stores rotated values (Q is graph-rotated, output inverse-rotated), so the f16 shadow is numerically the real attention input. Result: **prefill jumps to fp16 parity at every length** with zero stride surgery in the hot kernel.

| config | pp512 before → after | pp8192 before → after |
|--------|---------------------|----------------------|
| turbo3 | 394 → **1244** (f16 1267) | 82 → **577** (f16 580) |
| turbo4 | → **1249** | |

Decode stays on VEC, so the memory saving is fully preserved.

## 5. The bugs that stood between "compiles" and "correct"

Low-bit rotated KV is unforgiving — an off-by-one in a packed-bit stride silently corrupts the cache. The hard-won fixes:

- **`set_rows` single-row killer** — the turbo cache-write kernel only wrote row 0 of each batch, leaving the rest of the KV cache as garbage. Generation looked plausible then diverged. Fixed to write all rows.
- **Double-rotation** — Q was rotated both by the new graph node *and* by leftover inline kernel code → Q rotated twice. Removed the inline rotation.
- **half2-V misread** — the V tile was loaded as raw `half2` without dequantizing turbo V. Fixed the loader typing.
- **f16 TILE type guard** — a truthiness check on `type_K` mis-dispatched f16/q8_0 through the turbo path, producing garbage. Fixed the guard.
- **turbo2 mixed-KV abort** (`Not match KV type in vec`, `0xC0000409`) — a layer-adaptive mode created `K=turbo2 / V=q8_0` boundary layers with no matching VEC instance. Fixed by adding the mixed K=turbo/V=q8_0 cases **and** a new symmetric adaptive "mode 8" so every layer stays same-typed.
- **Context-shift crash** (`0xC0000005`) — the quantized K-shift path dereferenced a null rotation tensor and there is no SYCL turbo↔f32 cast. Turbo K-shift isn't implemented yet, so `get_can_shift()` now returns `false` for turbo → context-shift is **gracefully disabled** instead of crashing.

Every fix was locked in behind an on-device **golden numerical test** (`tests/test-sycl-turbo.cpp`): cosine ≥ 0.99999 vs the f16 reference across turbo2/3/4, mixed KV, and D∈{64,128}.

## 6. Performance — the honest picture

**KV-cache memory (MiB, Qwen3-4B, from `common_memory_breakdown`):**

| ctx | f16 | q8_0 | turbo4 | turbo3 | turbo2 |
|----:|----:|-----:|-------:|-------:|-------:|
| 16k | 2304 | 1224 | 612 | 450 | 408 |
| 64k | 9216 | 4896 | 2448 | 1800 | 1632 |

Savings vs fp16 @64k: **turbo2 5.6×** (adaptive mode 8; pure turbo2 = 7.5×), **turbo3 5.1×**, **turbo4 3.8×** (vs q8_0's 1.9×). fp16 @64k nearly OOMs the 12 GB B580; turbo runs 64k with 6+ GB free (~256k context reachable).

**Prefill / decode (llama-bench, -fa 1):** prefill matches fp16 at every length (§4). Decode at shallow context is near-fp16 (turbo3 tg128 70.4 vs f16 75.6).

**Decode at depth — the honest limitation:**

| depth | f16 | turbo3 | q8_0 |
|------:|----:|-------:|-----:|
| d0 | 75.9 | 70.8 | 72.3 |
| 8k | 57.6 | 28.9 | 26.1 |
| 32k | 33.9 | 10.1 | 8.9 |

Turbo decode slows at deep context — **but this is not a TurboQuant defect.** `q8_0` collapses *identically* (turbo3 actually beats it at depth). The bottleneck is the per-element dequant inside the VEC inner loop, inherent to **all** quantized-KV decode; f16 stays fast only because it skips dequant. So turbo delivers `q8_0`-class decode speed **at up to 2.7× less memory than `q8_0`** — that is the win. Closing the f16 gap at depth would need a fundamentally different decode kernel (the CUDA VEC path doesn't either).

## 7. What was explored and consciously deferred

- **XMX (joint_matrix) prefill kernel** — a Stage-0 proof-of-concept confirmed Intel's matrix engine works on the B580 (one fp16→f32 DPAS tile, error 1e-10, JIT, sub-group 16). **Parked**: since prefill is *already* at fp16 parity, the projected win is marginal (~1.6% at pp512). Full design + supported DPAS shapes are documented for a future revisit.
- **Cooperative-register VEC rework** — turbo `vec_dot` was rewritten to slice Q across 8 lanes (matching the f16/q8_0 convention, 8× register headroom toward D=256). Kept as a clean refactor; it confirmed the depth bottleneck is dequant *throughput*, not occupancy.
- **Native head_dim=64** and **turbo K-shift** — scoped, deferred.

## 8. Engineering practices used here

- **Test-first on real silicon** — a CPU golden reference + on-device numerical parity test gate every change; no "looks coherent" hand-waving.
- **Reuse over rewrite** — the prefill fix reuses the proven f16 TILE kernel instead of hand-rolling a fragile low-bit tile loader.
- **Fail safe, not loud** — unsupported paths (turbo context-shift, non-128 head dims) fall back gracefully rather than abort.
- **Honest benchmarking** — depth collapse is reported, root-caused, and shown to be common to all quantized KV, not hidden.
- **Reproducible delivery** — a dedicated CI/CD pipeline (`.github/workflows/tqp-sycl.yml`) auto-builds Windows + Linux SYCL on every push and publishes self-contained release packages.

## 9. Using it

```bash
# Turbo KV cache — requires flash-attention
llama-server -m model.gguf -ngl 99 --flash-attn on \
             --cache-type-k turbo3 --cache-type-v turbo3 -c 32768
```

Accepted `--cache-type-k` / `-v` strings: **`turbo2`**, **`turbo3`**, **`turbo4`**. Requires `--flash-attn on`. On Intel GPUs set `SYCL_CACHE_PERSISTENT=1` once so the SYCL JIT caches kernels to disk (first launch compiles all kernels — otherwise startup looks slow).

Pre-built binaries: see the [Releases page](https://github.com/FellypeMelo/llama-cpp-turboquant-SYCL/releases).

## 10. Source map

| Area | Files |
|------|-------|
| KV-cache integration, adaptive modes, shift guard | `src/llama-kv-cache.cpp` |
| SYCL flash-attention (dispatch, turbo prefill) | `ggml/src/ggml-sycl/fattn.cpp` |
| VEC decode kernel / cooperative `vec_dot` | `ggml/src/ggml-sycl/fattn-vec.hpp`, `fattn-common.hpp` |
| f16 TILE prefill kernel | `ggml/src/ggml-sycl/fattn-tile.hpp` |
| Turbo primitives (centroids, WHT, dequant) | `ggml/src/ggml-sycl/turbo-quants.hpp`, `turbo-wht.cpp` |
| Cache-write rotate+quantize | `ggml/src/ggml-sycl/set_rows.cpp` |
| Block formats / codebooks | `ggml/src/ggml-common.h`, `ggml/src/ggml.c` |
| Golden / stress tests | `tests/test-sycl-turbo.cpp` |
| CI/CD | `.github/workflows/tqp-sycl.yml` |

---

*TurboQuant is a KV-cache quantization technique; this fork implements and validates it on the `llama.cpp` SYCL backend for Intel GPUs. Built on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).*
