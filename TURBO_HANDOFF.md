# TurboQuant SYCL — Session Handoff

_Last updated: 2026-07-06. Read this first when resuming in a new session._

## North star
Full TurboQuant KV-cache quantization working **flawlessly on the SYCL backend**, max performance on Intel Arc hardware. **SYCL is the only backend that matters.** Target HW: Intel Arc B580 (Battlemage / Xe2, arch `bmg_g21`), 12 GB. Reference model: Qwen3-4B Q4_K_M (head_dim=128, GQA 4:1, 36 layers, pure attention).

## Immediate goal for the next session
**Run turbo KV cache on the Open-ChatBot project** (`G:\Programas\Open-ChatBot`) — a chatbot frontend/backend that launches `llama-server`.

### Plan
1. **Stop** any running Open-ChatBot llama-server (avoid two GPU procs clashing on the B580).
2. **Smoke-test turbo3 first** on that project's exact model via `llama-cli` with a short prompt — the model is a **hybrid arch (untested with turbo)**, so verify no abort + coherent output before wiring it into the app.
   - Model: `G:\Programas\Open-ChatBot\models\Qwen3-4B-Hivemind-Inst-Hrtic-Ablit-Uncensored-Q4_K_M-imat.gguf`
   - Run: `llama-cli -m <model> -ngl 99 -fa on -ctk turbo3 -ctv turbo3 -c 8192 -p "Hello, tell me a short story." -n 64 -st`
   - Watch for `Not match KV type in vec` / gibberish. Coherent = pass.
3. If pass → set Open-ChatBot config (below), add `SYCL_CACHE_PERSISTENT=1`, restart.
4. If abort/garbage → diagnose the hybrid-arch turbo path in `src/llama-kv-cache.cpp` (adaptive-mode/boundary logic assumes standard attention layers), OR fall back to `q4_0`.
5. Optional: bump `-c` to 32768+ so turbo's RAM savings actually matter.

### Open-ChatBot config change
In the backend's config `additional_args`, swap:
```
--cache-type-k q4_0 --cache-type-v q4_0 --flash-attn auto
```
→
```
--cache-type-k turbo3 --cache-type-v turbo3 --flash-attn on
```
- **Accepted type strings:** `turbo2`, `turbo3`, `turbo4` (NOT `turbo2_0`). turbo4 = 4-bit safest, **turbo3 = 3-bit best balance (~5.1× less KV RAM)**, turbo2 = 2-bit max savings/marginal quality.
- **3 requirements:** (1) use the turbo `llama-server.exe` (copy `build\bin\llama-server.exe` → `Open-ChatBot\llama_bin\`; turbo types only parse on this build); (2) flash-attn **ON** (turbo requires FA); (3) oneAPI runtime present (their launcher already sources setvars → fine).
- **Slow-start 503 fix:** SYCL JIT-compiles all kernels on first run → warmup exceeds the wrapper's 30-attempt `/health` window and *looks* like a failure, but the server does come up. Set `SYCL_CACHE_PERSISTENT=1` in the launcher env → kernels cache to disk, later starts are fast.

### ⚠️ Hybrid-model caveat
Open-ChatBot's model logs **`fused Gated Delta Net`** → it's a Qwen3-Next-style **hybrid** (attention layers + linear/DeltaNet layers), not the pure-attention Qwen3-4B turbo was validated on. Turbo on hybrid arch is **untested**. Test before trusting; fallback `q4_0`.

## Current state of the turbo work

### What works (validated on standard Qwen3-4B, B580)
- Turbo KV cache **turbo2 / turbo3 / turbo4** on SYCL: golden test green (cosine 0.99999), coherent generation ("Tokyo"), flash-attention enabled.
- **KV RAM savings vs f16 @64k:** turbo2 5.6×, turbo3 5.1×, turbo4 3.8× (q8_0 only 1.9×).
- **Prefill at f16 parity** (Option A: dequant turbo→f16 + f16 TILE kernel; turbo3 pp512 1244 vs f16 1267).
- **Decode at d0** ≈ f16 (turbo3 70 vs 75 t/s). Decode **collapses at deep context** — but this is inherent to *all* quantized KV (q8_0 collapses identically); not a turbo defect and not fixable without a fundamentally different decode kernel.
- Context-shift with turbo: gracefully disabled (turbo K-shift not implemented; `get_can_shift()` returns false for turbo → no crash).

### Git / build / CI
- Branch `feature/turboquant-kv-cache`, **HEAD `9300aca34`**, pushed to `origin` (github.com/FellypeMelo/llama-cpp-turboquant-SYCL — the user's fork; `upstream` = ggml-org, do NOT push there).
- 7 turbo code commits + 1 CI commit pushed.
- **Final Release build done + verified:** `build\bin\{llama-cli,llama-server,llama-bench,llama-quantize,test-sycl-turbo}.exe` (version `9098 (76fe21c3f)`, runs OK).
- **CI/CD deployed:** `.github/workflows/tqp-sycl.yml`. Auto-builds on push to the branch + on manual dispatch (downloadable artifacts under the run); push a `tqp-sycl-v*` tag → publishes a GitHub Release (self-contained Windows SYCL zip with bundled oneAPI DLLs). First run `28824261096` — **PASSED GREEN** (windows-sycl 29m7s, linux-sycl 17m40s). Downloadable build artifacts (`turboquant-plus-sycl-windows-x64` zip + linux tar) attached to that run. Workflow validated end-to-end.

### XMX prefill kernel — PARKED
Scoped a joint_matrix (XMX) FA prefill kernel. **Stage 0 PoC passed** (one fp16→f32 DPAS tile on B580, err 1e-10, JIT, SG16 — joint_matrix works). But **parked by user decision**: prefill is already at f16 parity so the payoff is marginal (~1.6% at pp512; win only maybe at pp8192). Full plan + DPAS combos + toolchain fixes are in memory `xmx-fa-prefill.md`. Worktree `G:\Programas\llama-cpp-turbo-xmx` (branch `feature/xmx-fa-prefill`, unbuilt) banked for later.

## Build / run environment (Windows, oneAPI)
- oneAPI at `C:\Program Files (x86)\Intel\oneAPI`, compiler `2026.0`. VS 2022.
- **Standalone icx / setvars fixes** (needed under the agent shell): (1) add `C:\Program Files (x86)\Microsoft Visual Studio\Installer` to PATH (vswhere); (2) `set "NoDefaultCurrentDirectoryInExePath="` (so cmd resolves component `vars.bat`); (3) `/EHsc` (icx defaults to `-fno-exceptions`). Run `.bat` via the **PowerShell tool** `cmd /c ...` — Bash `cmd //c` loses system PATH → vswhere fails.
- **Configure:** `cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release` (see `build_sycl.bat`).
- **Build targets:** `cmake --build build --config Release -j 8 --target llama-cli llama-server llama-bench llama-quantize test-sycl-turbo`.
- **Kill** llama-cli/llama-server/llama-bench/test procs before relinking (they hold the ggml-sycl DLL).
- **Cannot run two GPU benches/servers concurrently** (they corrupt each other's measurements / clash on the device).
- Scratchpad bats + outputs from this work: session scratchpad (build_poc.bat, final_build.bat, etc.).

## Key files
- Turbo KV cache logic (adaptive modes, shift guard, cpy): `src/llama-kv-cache.cpp`
- SYCL flash-attention: `ggml/src/ggml-sycl/fattn.cpp` (dispatch + turbo prefill), `fattn-vec.hpp` (decode VEC), `fattn-common.hpp` (vec_dot), `fattn-tile.hpp` (f16 TILE)
- Turbo primitives: `ggml/src/ggml-sycl/turbo-quants.hpp`, `turbo-wht.cpp`, `set_rows.cpp`
- Type defs / names: `ggml/include/ggml.h`, `ggml/src/ggml-common.h`, `ggml/src/ggml.c` (type_name `turbo2`/`turbo3`/`turbo4`, line ~748); CLI parse `common/arg.cpp:384`
- Golden/stress tests: `tests/test-sycl-turbo.cpp`
- CI: `.github/workflows/tqp-sycl.yml`

## Memory files (auto-load via MEMORY.md)
`turbo-sycl-perf` · `turbo-sycl-parity-project` · `turbo-sycl-diagnosis` · `xmx-fa-prefill` · `open-chatbot-turbo-usage`

## Working constraints
- **No commit/push without explicit approval** (AGENTS.md). User has approved past turbo commits + the CI push.
- Caveman response style active (terse; code/commits written normally).
- Ultracode on (use Workflow for substantive multi-step tasks; adversarially verify).
