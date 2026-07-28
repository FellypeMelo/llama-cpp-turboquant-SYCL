# Testes e receita de build — fork TurboQuant SYCL

Fonte única de verdade para compilar e validar o backend turbo do SYCL no Windows + Intel Arc.

## Ambiente de build (Windows, oneAPI 2026.0, Intel Arc B580)

O icx linka através do MSVC e precisa do ambiente LIB/INCLUDE do MSVC **e** do `rc.exe`/`mt.exe` do
Windows SDK. Só colocar os diretórios do oneAPI no PATH NÃO basta (falha com `LNK1104`, depois
`LNK1158 cannot run rc.exe`). A receita que funciona faz source do `setvars.bat` (com duas correções
para o shell do agente) **mais** coloca o Ninja e o bin do Win10 SDK no PATH. Tudo encadeado em UMA
única invocação (o ambiente não persiste entre chamadas do PowerShell):

```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\Installer;$env:PATH"   # vswhere para o setvars
cmd /c 'set "NoDefaultCurrentDirectoryInExePath=" && ^
  "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" > nul 2>&1 && ^
  set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH%" && ^
  <configure/build do cmake aqui>'
```

As duas correções de shell de agente para o `setvars.bat`:
1. Adicionar o diretório **Installer** do VS ao PATH (no PowerShell, antes do `cmd`) para que o
   `vswhere` do setvars resolva.
2. `set "NoDefaultCurrentDirectoryInExePath="` para que o `cmd` consiga rodar os `vars.bat` de
   componente.

Entradas extras de PATH adicionadas dentro do cmd (o setvars não as adiciona sozinho):
- Ninja: `...\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja` (vem com o VS).
- Bin do Win10 SDK (rc.exe/mt.exe): `C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64`.

### Configure (RECOMENDADO — build de perf oficial para Arc B580 / Xe2)
Medido em 2026-07-09: estas flags praticamente dobram o prefill (pp512 +102,8%, 2,03×) e mantêm o
decode estável (-3,3%, dentro do ruído), com o gate golden `test-sycl-turbo` VERDE (22/22) e saída
coerente. Ver `docs/pt-BR/benchmarks.md`.
```
cmake -S . -B build-perf -G Ninja -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
  -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_DEVICE_ARCH=bmg-g21
```
- `GGML_SYCL_F16=ON` — liga o caminho de matmul FP16 nas unidades XMX do Xe2 (a fonte do ganho de
  prefill).
- `GGML_SYCL_DEVICE_ARCH=bmg-g21` — compilação AOT via `spir64_gen`, eliminando o warmup de JIT em
  runtime. **Atenção: use o hífen `bmg-g21`** (é o que o ocloc 2026.0 espera por esse caminho do
  CMake); a forma com underscore `bmg_g21` é o seletor `-fsycl-targets=intel_gpu_bmg_g21`, um
  mecanismo diferente, e NÃO funciona por este caminho do CMake. Trade-off: o link AOT é muito lento
  (~1h+) e o `ggml-sycl.dll` fica com ~247 MB.
- `GGML_SYCL_DNN=ON` — liga o oneDNN.

> **Ressalva Xe2:** as flags F16/AOT ativam caminhos numéricos mais agressivos. São seguras **nesta
> GPU** (golden verde + saída coerente na B580), mas **re-rodar o gate golden é obrigatório** em
> qualquer outra arquitetura/toolchain antes de confiar nelas ali. Se o gate golden falhar, ou o
> build AOT lento incomodar, o fallback é F16-only sem AOT (link rápido, paga o custo de JIT no
> primeiro run) ou a configuração baseline abaixo.

### Configure (baseline / fallback — sem flags de perf)
```
cmake -S . -B build-sync -G Ninja -DGGML_SYCL=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON
```

### Build (as flags do ninja vêm DEPOIS de `--`; o cmake rejeita `-k`)
```
cmake --build build-sync --config Release \
  --target test-sycl-turbo llama-cli llama-perplexity llama-bench llama-quantize -- -j 8 -k 0
```
O Ninja é incremental — se o timeout de 10 min do shell matar o processo, é só rodar de novo; ele
retoma de onde parou. Mate qualquer processo llama-*/test-sycl-* antes de relinkar (eles seguram o
`ggml-sycl.dll`).

## Gates do turbo

1. **Golden (primário, não precisa de modelo)** — paridade numérica de cada primitiva turbo na GPU:
   ```
   <ambiente setvars> && set "SYCL_CACHE_PERSISTENT=1" && build-sync\bin\test-sycl-turbo.exe
   ```
   PASS = exit 0, todos os casos PASSED (cosseno de quant ~1,0, round-trip fwd/inv da WHT, paridade
   de FA turbo, FA DECODE+GQA+padding-mask, FA com cache KV TURBO3_0, além da paridade de KV misto
   ASSIMÉTRICO para `{q8_0,f16} x {turbo2,turbo3,turbo4}` K/V). **Última execução: exit 0, 34/34
   PASSED.**

2. **Gate e2e de COERÊNCIA do KV turbo (OBRIGATÓRIO — precisa do modelo de referência + uma GPU)** —
   `tests/test-e2e-turbo-kv.sh`, integrado ao ctest como `test-e2e-turbo-kv` (labels
   `e2e;gpu;turbo`). Executa **geração real via `llama-cli`** com o KV turbo LIGADO e verifica se as
   configurações seguras no SYCL -- as três simétricas (`turbo3/turbo3`, `turbo4/turbo4`,
   `turbo2/turbo2`) mais as seis ASSIMÉTRICAS de K preciso + V turbo (`{q8_0,f16} x
   {turbo2,turbo3,turbo4}`) -- produzem texto **coerente** (palavra-chave no tópico + sem repetição
   degenerada + sem abort/NaN + sem corrupção '?'). Essa é a prova direta de que *"com o KV turbo
   realmente ligado, a geração é coerente"* — o golden (gate 1) só prova paridade de kernel, não
   geração ponta a ponta. Para rodar (com o ambiente oneAPI já carregado):
   ```
   <ambiente setvars> && ctest --test-dir build-perf -R test-e2e-turbo-kv --output-on-failure
   # ou standalone:  TURBO_E2E_MODEL=G:/models/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
   #                 bash tests/test-e2e-turbo-kv.sh
   ```
   PASS = exit 0, todas as configs COERENTES. **Faz SKIP (exit 77) quando o modelo/llama-cli está
   ausente** (ex.: nos runners do GitHub, que não têm modelo), então é um no-op no `tqp-sycl.yml` e
   um gate real de fato só na **máquina Arc self-hosted**. **Última execução: PASS na B580** (3
   simétricas + 6 assimétricas, todas coerentes). Matriz completa de resultados + amostras em
   `docs/pt-BR/benchmarks.md` ("matriz e2e de COERÊNCIA").

3. **Gate de qualidade PPL + velocidade (precisa do modelo de referência + wikitext)** —
   `scripts/turbo-quality-gate.sh` (PPL do turbo3 < 1,05× a baseline q8_0; razão de velocidade
   turbo3/q8_0 > 0,95 @ 4K). Requer o modelo de referência validado, puramente de atenção
   (Qwen3-4B), mais o `wikitext-2-raw`. Não roda quando só há um modelo híbrido disponível.

4. **CI** — `.github/workflows/tqp-sycl.yml` (windows-sycl + linux-sycl), separado do `build.yml` do
   upstream (que o upstream removeu); sobrevive ao sync. (O gate e2e de coerência faz SKIP aqui — sem
   GPU/modelo nos runners hospedados; rode-o na máquina Arc self-hosted.)

### Configuração recomendada de KV turbo (medida em 2026-07-09, Qwen3-4B, B580)
Melhor compressão de KV mantendo coerência: **`-ctk turbo2 -ctv turbo2`** (modo 8 de borda
automática) → **5,65×** menos VRAM de KV. Default mais seguro, com margem de qualidade maior:
**`-ctk turbo3 -ctv turbo3`** (5,12×). Ambos exigem `-fa on`. Evite `TURBO_LAYER_ADAPTIVE=0` com
turbo2 (repetição degenerada) e `TURBO_LAYER_ADAPTIVE=5/6/7` (abort no FA do SYCL). Strings de tipo
aceitas: `turbo2`/`turbo3`/`turbo4`.

ASSIMÉTRICO (K preciso + V turbo, "V é de graça, K é tudo"): `-ctk q8_0 -ctv turbo3` é o default de
produção com prioridade em qualidade (K protegido em 8 bits, 2,75×), e `-ctk f16 -ctv turboN` é o
modo de K com precisão máxima (Q fica não-rotacionado). Todas as seis combinações `{q8_0,f16} x
{turbo2,turbo3,turbo4}` fazem dispatch e são gated como coerentes (head_dim=128). NUNCA habilite
turbo-K + turbo-V juntos.

## Nota sobre a primeira execução
O SYCL compila os kernels via JIT no primeiro lançamento (warmup lento). Defina
`SYCL_CACHE_PERSISTENT=1` para que os kernels fiquem em cache no disco e as próximas partidas sejam
rápidas.
