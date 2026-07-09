# QUALITY — build + gate recipe (TurboQuant SYCL fork)

Single source of truth for building and validating the SYCL turbo backend on Windows + Intel Arc.

## Build environment (Windows, oneAPI 2026.0, Intel Arc B580)

icx links through MSVC and needs the MSVC LIB/INCLUDE env **and** the Windows SDK `rc.exe`/`mt.exe`.
A pure PATH-prepend of the oneAPI dirs is NOT enough (fails: `LNK1104`, then `LNK1158 cannot run rc.exe`).
The working recipe sources `setvars.bat` (with two agent-shell fixes) **plus** puts Ninja and the
Win10 SDK bin on PATH. All chained in ONE invocation (env does not persist between PowerShell calls):

```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\Installer;$env:PATH"   # vswhere for setvars
cmd /c 'set "NoDefaultCurrentDirectoryInExePath=" && ^
  "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul 2>&1 && ^
  set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%" && ^
  <cmake configure / build here>'
```

The two agent-shell fixes for `setvars.bat`:
1. Prepend the VS **Installer** dir to PATH (in PowerShell, before `cmd`) so setvars' `vswhere` resolves.
2. `set "NoDefaultCurrentDirectoryInExePath="` so `cmd` can run the component `vars.bat` files.

Extra PATH entries added inside cmd (setvars does not add them here):
- Ninja: `...\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja` (ships with VS).
- Win10 SDK bin (rc.exe/mt.exe): `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64`.

### Configure (RECOMENDADO — perf build oficial p/ Arc B580 / Xe2)
Medido em 2026-07-09: estas flags dobram o prefill (pp512 +102.8%, 2.03x) e mantêm decode flat (-3.3%,
ruído), com golden `test-sycl-turbo` VERDE (22/22) e saída coerente. Ver `docs/BENCHMARKS.md`.
```
cmake -S . -B build-perf -G Ninja -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
  -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g21
```
- `GGML_SYCL_F16=ON` — matmul FP16 no XMX Xe2 (fonte do ganho de prefill).
- `GGML_SYCL_DEVICE_ARCH=bmg-g21` — AOT via `spir64_gen`, mata o warmup JIT no runtime. **Atenção: use o
  hífen `bmg-g21`** (forma que o ocloc 2026.0 espera); o underscore `bmg_g21` é o seletor
  `-fsycl-targets=intel_gpu_bmg_g21`, mecanismo diferente, e NÃO funciona por este caminho do CMake.
  Trade-off: o link AOT é muito lento (~1h+) e o `ggml-sycl.dll` fica ~247 MB.
- `GGML_SYCL_DNN=ON` — oneDNN ligado.

> **Ressalva Xe2:** as flags F16/AOT ativam caminhos numéricos mais agressivos. São seguras **nesta GPU**
> (golden verde + saída coerente na B580), mas **re-rodar o golden gate é obrigatório** em qualquer outra
> arch/toolchain antes de confiar nelas. Se o golden falhar ou o build AOT lento incomodar, o fallback é
> F16-only sem AOT (link rápido, paga JIT no 1o run) ou o baseline abaixo.

### Configure (baseline / fallback — sem flags de perf)
```
cmake -S . -B build-sync -G Ninja -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
```

### Build (ninja flags go AFTER `--`; cmake rejects `-k`)
```
cmake --build build-sync --config Release \
  --target test-sycl-turbo llama-cli llama-perplexity llama-bench llama-quantize -- -j 8 -k 0
```
Ninja is incremental — if the 10-min shell timeout kills it, just re-run; it resumes.
Kill any llama-*/test-sycl-* procs before relinking (they hold `ggml-sycl.dll`).

## Turbo gates

1. **Golden (primary, no model needed)** — numeric parity of every turbo primitive on the GPU:
   ```
   <setvars env> && set "SYCL_CACHE_PERSISTENT=1" && build-sync\bin\test-sycl-turbo.exe
   ```
   PASS = exit 0, all cases PASSED (quant cosine ~1.0, WHT fwd/inv round-trip, FA turbo parity,
   FA DECODE+GQA+padding-mask, FA with TURBO3_0 KV cache). **Last run: exit 0, all PASSED.**

2. **PPL quality + speed gate (needs the reference model + wikitext)** — `scripts/turbo-quality-gate.sh`
   (turbo3 PPL < 1.05× q8_0 baseline; turbo3/q8_0 speed ratio > 0.95 @ 4K). Requires the validated
   pure-attention model (Qwen3-4B) + `wikitext-2-raw`. Not runnable when only a hybrid model is present.

3. **CI** — `.github/workflows/tqp-sycl.yml` (windows-sycl + linux-sycl), separate from upstream's
   `build.yml` (which upstream deleted); survives the sync.

## First-run note
SYCL JIT-compiles all kernels on first launch (slow warmup). Set `SYCL_CACHE_PERSISTENT=1` so kernels
cache to disk and later starts are fast.
