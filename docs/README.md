# Documentation index

This `docs/` tree mixes two things: the unmodified upstream `llama.cpp` documentation, and this
fork's own internal documentation about the TurboQuant SYCL work. Only the fork's own documentation
is bilingual.

## Fork-owned documentation (bilingual)

| Topic | English | Português (Brasil) |
|---|---|---|
| Architecture / engineering deep-dive | [`en/architecture.md`](en/architecture.md) | [`pt-BR/architecture.md`](pt-BR/architecture.md) |
| ADR log (architecture decision records) | [`en/decisions.md`](en/decisions.md) | [`pt-BR/decisions.md`](pt-BR/decisions.md) |
| Benchmarks (measured results) | [`en/benchmarks.md`](en/benchmarks.md) | [`pt-BR/benchmarks.md`](pt-BR/benchmarks.md) |
| Upstream sync runbook | [`en/upstream-sync.md`](en/upstream-sync.md) | [`pt-BR/upstream-sync.md`](pt-BR/upstream-sync.md) |
| Testing & build recipe | [`en/testing.md`](en/testing.md) | [`pt-BR/testing.md`](pt-BR/testing.md) |
| Roadmap | [`en/roadmap.md`](en/roadmap.md) | [`pt-BR/roadmap.md`](pt-BR/roadmap.md) |

Each pair covers the same content, same structure, same headings, same order — pick whichever
language you read more comfortably. Code blocks, commands, identifiers, paths and error strings are
identical in both.

Two fork-owned documents are intentionally **not** mirrored here and stay where they are, at the
repository root:

- **`STATE.md`** — a Portuguese-language internal session log (what was done, in what session, on
  what date). It is a running journal, not reference documentation, so it isn't translated or moved.
- **`TURBO_HANDOFF.md`** — an English-language internal handoff/session-resumption note for whoever
  picks the work back up next. Same reasoning: it's a snapshot of one session's working state, not a
  stable reference doc.

`QUALITY.md` at the repository root is now a short redirect stub pointing at
[`en/testing.md`](en/testing.md) — the content that used to live there moved into the bilingual tree
above.

## Upstream documentation (unmodified, English only)

Everything else under `docs/` — `build.md`, `backend/**`, `android.md`, `android/**`, `development/**`,
`multimodal.md`, `multimodal/**`, `ops.md`, `ops/**`, `docker.md`, `install.md`, `multi-gpu.md`,
`preset.md`, `speculative.md`, `function-calling.md`, `llguidance.md`, `autoparser.md`,
`build-riscv64-spacemit.md`, `build-s390x.md`, `rocm-mi300x-test-results.md` and their assets — is
unmodified upstream `llama.cpp` documentation. It stays exactly where upstream put it, in English
only, and is not part of this fork's own documentation set. Do not move, translate, or reformat it;
treat it as frozen third-party content.
