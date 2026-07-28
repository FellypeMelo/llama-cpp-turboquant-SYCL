# Roadmap — TurboQuant SYCL fork

This is a forward-looking summary of what is deliberately deferred, scoped-but-not-implemented, or
still pending in this fork, distilled from the engineering notes in `docs/en/architecture.md`,
`docs/pt-BR/decisions.md` (ADR-0005), and `docs/en/benchmarks.md`. Nothing here is a promise or a
schedule — it is an honest list of open engineering work, in no particular priority order unless
stated.

## Correctness / quality gates not yet closed

- **PPL / speed quality gate** (`scripts/turbo-quality-gate.sh`) — exists and is wired into the test
  suite, but has not been run end-to-end. It needs a locally available pure-attention reference
  model (the validated target is Qwen3-4B) plus a `wikitext-2-raw` dataset; neither has been present
  on the validation machine in the sessions documented so far. The golden cosine-similarity gate
  (see `docs/en/testing.md`) passes and proves numerical kernel parity, but that is not the same
  claim as a measured perplexity delta — until this gate runs, the quality story rests on the golden
  gate plus the qualitative e2e-coherence samples in `docs/en/benchmarks.md`.
- **Perplexity rescue on a second GQA ratio** — the asymmetric K/V work (ADR-0005) was validated on
  Qwen3-4B (GQA 4:1); a confirmation run on a model with a different GQA ratio (a Qwen2.5-class 7:1
  model was the intended target) is pending. The auto-asymmetric K-upgrade logic has a GQA≥6
  threshold that only a model in that range can actually exercise.
- **Re-measuring the asymmetric K/V configs under the AOT perf build** — the six asymmetric
  `{q8_0,f16} x {turbo2,turbo3,turbo4}` combinations are currently only benchmarked on the
  build-sync (JIT) binary; the validated performance headline (+102.8% prefill) comes from a
  separate build-perf (F16+AOT) binary that has not yet had the asymmetric configs re-run against
  it.

## Scoped but not implemented

- **XMX (`joint_matrix`) prefill kernel** — a Stage-0 proof of concept confirmed Intel's matrix
  engine works for this on the B580 (one fp16→f32 DPAS tile, error 1e-10, JIT, sub-group 16). It is
  parked rather than pursued further: prefill is already at fp16 parity via the dequant-to-f16 path
  (see `docs/en/architecture.md` §4), so the projected additional gain is marginal (~1.6% at pp512),
  and the payoff would only plausibly grow at much longer prefill lengths (pp8192+). Revisit if a
  future workload makes very-long-prefill throughput the bottleneck.
- **Native head_dim=64 support** — scoped, not implemented. Turbo is currently validated only at
  head_dim=128.
- **Turbo K-shift** — context-shift on a turbo-quantized K cache is not implemented; it currently
  disables gracefully (`get_can_shift()` returns `false` for turbo types) instead of crashing or
  being supported. Implementing it would need a quantized-K shift kernel with no f32 intermediate
  cast, which does not exist today.
- **Cooperative-register VEC rework** — the turbo `vec_dot` was rewritten experimentally to slice Q
  across 8 lanes (matching the f16/q8_0 convention, with headroom toward D=256). Kept as a clean
  refactor rather than merged as a performance win: it confirmed that the decode-at-depth slowdown
  is a dequant-*throughput* bottleneck, not an occupancy problem, so this rework alone would not
  close the gap.

## Backend coverage

- **CUDA / Metal / Vulkan turbo support** — present in the codebase from earlier phases of this
  fork's history (see `ggml/src/ggml-metal/turbo-matrices.h` and the CUDA/Vulkan dispatch paths in
  `docs/en/upstream-sync.md`), but SYCL on Intel Arc is the only backend covered by the gates in
  `docs/en/testing.md`. The other backends are best-effort and are not re-validated on every
  upstream sync.
- **Known Vulkan regression** — after the 2026-07 upstream merge, turbo3 flash-attention on
  Vulkan's scalar and coopmat1 paths became non-functional (upstream moved flash-attention's dequant
  code to a new shader file with no `TURBO3_0` case). The coopmat2 path is preserved. This is a
  best-effort, non-Arc backend and has not been prioritized for a fix.

## Build configuration follow-ups

- **Re-enabling `GGML_SYCL_FA_ALL_QUANTS`** (see ADR-0004 in `docs/pt-BR/decisions.md` /
  `docs/en/decisions.md`) requires generating the full turbo flash-attention-vec instance matrix
  (turbo × every KV type, across every supported head dimension) before the macro can be turned back
  on without a link failure. Out of scope until there is a concrete need for the additional
  standard-type combinations it would unlock (Q4_1/Q5_0/Q5_1 paired with turbo).

## Ongoing maintenance

- **Upstream re-sync** — this fork tracks `ggml-org/llama.cpp` via periodic large merges rather than
  a one-off snapshot (see `docs/en/upstream-sync.md`). The touch-point checklist at the end of that
  document exists specifically so the next sync's conflict surface stays small; keeping it current
  as the turbo hooks evolve is ongoing work, not a one-time task.
