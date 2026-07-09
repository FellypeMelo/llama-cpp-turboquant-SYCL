# BENCHMARKS — TurboQuant SYCL fork

_Números medidos. Nada estimado. Ler junto com `QUALITY.md` (receita de build+gate) e `STATE.md`._

## Perf-pass 2026-07-09 — Intel Arc B580 (Xe2 / BMG)

- **GPU:** Intel Arc B580 Graphics (level_zero, 20 EU-slices, 12452 MiB).
- **Modelo bench:** Qwen3-4B Q4_K_M (2.32 GiB), `-ngl 99`.
- **Comando:** `llama-bench -p 512 -n 128 -r 3` (média de 3 reps ± stddev; média ~ mediana, runs estáveis).
- **Build base do binário:** merge commit `b76ddd343`.
- **GPU limpa:** B580 idle antes do bench, engines de compute ~0%, sem contenção durante os runs.

| Config | Flags | pp512 (tok/s) | tg128 (tok/s) | Golden verde? | Data | GPU |
|---|---|---|---|---|---|---|
| baseline | `GGML_SYCL=ON`, F16=OFF, sem AOT (JIT), DNN=ON, Release | 1246.70 ± 2.87 | 79.03 ± 0.48 | sim | 2026-07-09 | Arc B580 |
| perf-f16-aot | `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`, AOT `GGML_SYCL_DEVICE_ARCH=bmg-g21`, DNN=ON, Release | 2528.80 ± 18.62 | 76.41 ± 2.34 | sim | 2026-07-09 | Arc B580 |

### Leitura dos números

- **Prefill (pp512): +102.8% (2.03x).** Ganho grande. F16 liga o matmul FP16 no XMX Xe2, e prefill é
  compute-bound -> escala forte.
- **Decode (tg128): -3.3%.** Regressão marginal, dentro do ruído (barras de erro quase se tocam:
  perf 74.1-78.8 vs base 78.6-79.5). Decode é memory-bandwidth-bound: F16 não ajuda e o caminho FP16
  adiciona overhead mínimo. Na prática: decode fica flat.
- **Veredito:** a config perf-f16-aot ~dobra o prefill e mantém o decode praticamente igual.

### Golden gate (prova de correção na GPU)

`test-sycl-turbo` (build-perf) = **VERDE**: exit 0, 22 PASSED / 0 FAIL na Arc B580 via level_zero,
`GGML_SYCL_DNNL: yes`. Quant cosine ~1.0, WHT round-trip MSE 0, FA turbo parity TURBO2/3/4_0 pior
cosine 0.999986-1.000000, FA DECODE+GQA+padding-mask cosine 1.000000, FA com TURBO3_0 KV cache
graph_compute status 0. **As flags F16/AOT NÃO corrompem o turbo no Xe2.**

### Sanidade de saída (build perf, o de maior risco de corrupção)

`llama-cli -n 50 --seed 42` = texto **coerente**: explicação correta de Rayleigh scattering (por que o
céu é azul), sem lixo/repetição (Prompt 192.5 t/s | Generation 64.1 t/s). Anti-corrupção confirmada no
build que importa. O baseline não teve saída capturada a tempo (warmup JIT lento no 1o launch — que é
exatamente o custo que o AOT elimina); não é lixo, é UNVERIFIED. Correção equivalente ao baseline já é
coberta pelo golden gate.

### Notas de método

- build-sync (baseline) não tinha binário `llama-bench`; foi buildado o target `llama-bench` em build-sync
  SEM mudar flag nenhuma (CMakeCache existente: Release, `GGML_SYCL=ON`, F16=OFF, DEVICE_ARCH vazio, DNN=ON).
  Relinkou `ggml-sycl.dll` a partir das mesmas fontes baseline — sem mudança de flag/source.
- ggml-sycl.dll do build-perf tem ~247 MB pois embute TODOS os kernels AOT para bmg-g21. Link do AOT foi
  muito lento (~1h+ só no link, pela matriz de instâncias de flash-attention turbo). Trade-off: warmup JIT
  em runtime some, ao custo de build lento — normal para AOT no SYCL.
