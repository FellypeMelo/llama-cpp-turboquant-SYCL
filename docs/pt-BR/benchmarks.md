# BENCHMARKS — TurboQuant SYCL fork

_Números medidos. Nada estimado. Ler junto com `docs/pt-BR/testing.md` (receita de build+gate) e `STATE.md`._

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

## turbo KV-cache — matriz e2e de COERÊNCIA 2026-07-09 (Arc B580, build-perf F16+AOT)

Responde a pergunta central do dono: **com o KV-cache turbo REALMENTE ligado, a geração sai coerente?**
Não é parity de kernel (isso é o golden `test-sycl-turbo`, cosine ~1.0) — é **geração real ponta-a-ponta**
via `llama-cli`. Prova nova, complementar ao golden.

- **Modelo:** Qwen3-4B-Instruct-2507 Q4_K_M (pura-atenção, 36 camadas, `n_embd_head=128`).
- **GQA:** `n_head=32`, `n_head_kv=8` → **ratio 4:1**. Abaixo do limiar 6 da lógica auto-assimétrica →
  **auto-asimétrica NÃO engaja** (esperado e correto; o código nota "Mistral 4:1 → turbo3 K works fine").
  Nenhuma linha `auto-asymmetric` no log p/ este modelo. (Só engajaria em modelo tipo Qwen2.5 7:1.)
- **Comando:** `llama-cli -ngl 99 -fa on -c 4096 --temp 0 -n 150 -st` (greedy, determinístico), prompt
  `"Explain why the sky is blue in 3 sentences."` (+ um prompt de raciocínio de conferência).
- **VRAM-KV:** valor exato do log `llama_kv_cache: size = ...` (inclui zero-padding turbo do head_dim).
- **Nomes de tipo aceitos por `-ctk`/`-ctv`:** `turbo2`, `turbo3`, `turbo4` (NÃO `turboN_0`).

| `-ctk`/`-ctv` | KV @4k (MiB) | K / V (MiB) | compressão vs f16 | gen tok/s | pp tok/s | coerente? | abortou? | observação |
|---|---|---|---|---|---|---|---|---|
| `f16`/`f16` (default) | 576.00 | 288 / 288 | 1.00× | 77.4 | 204 | **sim** (referência) | não | baseline |
| `q8_0`/`q8_0` | 306.00 | 153 / 153 | 1.88× | 73.4 | 72 | **sim** (≈ f16) | não | quant quase-lossless |
| `q4_0`/`q4_0` | 162.00 | 81 / 81 | 3.56× | 73.6 | 71 | **sim** | não | |
| `turbo4`/`turbo4` | 153.00 | 76.5 / 76.5 | 3.76× | 73.6 | 70 | **sim** | não | turbo mais seguro |
| `turbo3`/`turbo3` | 112.50 | 56.25 / 56.25 | **5.12×** | 74.8 | 77 | **sim** | não | **default recomendado (melhor equilíbrio)** |
| `turbo2`/`turbo2` (auto) | 102.00 | 51 / 51* | **5.65×** | 74.2 | 69 | **sim** | não | mode 8 auto (bordas q8_0); **maior compressão coerente** |
| `turbo2`/`turbo2` `TURBO_LAYER_ADAPTIVE=0` | 76.50 | 38.25 / 38.25 | 7.53× | 70.6 | 76 | **NÃO — repetição degenerada** | não | 2-bit uniforme; mostra POR QUE a borda auto existe |
| `turbo3`-K / `q8_0`-V (misto) | 209.25 | 56.25 / 153 | 2.75× | 73.8 | 199 | sim (neste run) | não | K=turbo/V=q8_0 marcado **não-confiável** no código → NÃO gated |
| `turbo2`/`turbo2` `TURBO_LAYER_ADAPTIVE=5` | 80.75 | 38.25 / 42.5 | — | — | — | — | **SIM (abort)** | `fattn.cpp:166 Not match KV type: K=turbo2 V=turbo4` — misto SYCL-inseguro |

\* `turbo2` (auto) = **mode 8 boundary-symmetric-q8_0** engaja sozinho (log: `Boundary symmetric q8_0
auto-enabled for turbo2-V`): 1as-2 + últimas-2 camadas K+V=q8_0, 32 camadas do meio K+V=turbo2. Por isso
os 102 MiB reais > o nominal "51+51" turbo2 puro. `TURBO_LAYER_ADAPTIVE=0` desliga a borda → turbo2 puro
2-bit → **degenera** (loop de repetição "…due to Rayleigh scattering of sunlight by molecules and small
particles in the atmosphere." repetido 3-4×, gramática "blue due because"). É o achado honesto: turbo2 puro
uniforme degrada; a borda auto (mode 8) o resgata p/ coerente.

### Matriz ASSIMÉTRICA — K preciso + V turbo (2026-07-09, Arc B580, build-sync JIT)

Config assimétrica = K preciso (q8_0 ou f16) + V turbo ("V is free, K is everything"). O default de
prod recomendado (`-ctk q8_0 -ctv turboN`) já dispatchava; a NOVA é `f16`-K + turbo-V (ver ADR-0005 e
`fattn.cpp` rows `FATTN_VEC_CASES_TURBO_D(F16, TURBOx_0)`). head_dim=128. Golden parity provada em
`test-sycl-turbo` (cosine 1.000000; f16 rel-MSE 0.0, q8_0 rel-MSE ~8e-4) e coerência e2e nas 6 configs.

- **Golden (cosine / rel-MSE, decode n_q=1):** todas cosine **1.000000**. rel-MSE: `f16`-K = **0.0**
  (dequant f16 bit-exato) ; `q8_0`-K = ~**8e-4** (arredondamento q8_0 device vs CPU). DECODE/GQA/mask:
  idem (q8_0 rel-MSE ~8e-5, f16 0.0).
- **KV @4k (MiB):** soma exata dos tamanhos por-lado já medidos na matriz simétrica acima
  (`f16`=288, `q8_0`=153, `turbo4`=76.5, `turbo3`=56.25, `turbo2` puro=38.25 por lado).
- **tok/s medido (build-sync JIT, `llama-bench` -p 512 -n 128 -r 2):** `q8_0/turbo3` pp512 **392.6**,
  tg128 **69.9** ; `f16/turbo3` pp512 **415.8**, tg128 **70.3**. Decode ~70 t/s idêntico entre configs
  (path vec compartilhado). São números build-sync JIT (piso); a **perf headline validada é a do
  build-perf F16+AOT** (matriz simétrica: +102.8% prefill) -> re-medir AOT das assimétricas = PENDENTE.

| `-ctk`/`-ctv` | KV @4k (MiB) | K / V (MiB) | compressão vs f16 | coerente? | abortou? | observação |
|---|---|---|---|---|---|---|
| `q8_0`/`turbo3` | 209.25 | 153 / 56.25 | 2.75× | **sim** | não | **default de prod recomendado** (K protegido 8-bit) |
| `q8_0`/`turbo4` | 229.50 | 153 / 76.5 | 2.51× | **sim** | não | K q8_0 + V mais seguro |
| `q8_0`/`turbo2` | 191.25 | 153 / 38.25 | 3.01× | **sim** | não | K q8_0 + V máx. compressão |
| `f16`/`turbo3` | 344.25 | 288 / 56.25 | 1.67× | **sim** | não | **NOVO** — K precisão máx.; Q fica NÃO-rotada |
| `f16`/`turbo4` | 364.50 | 288 / 76.5 | 1.58× | **sim** | não | **NOVO** — K f16 + V turbo4 |
| `f16`/`turbo2` | 326.25 | 288 / 38.25 | 1.77× | **sim** | não | **NOVO** — K f16 + V turbo2 |

\* Linhas `turbo2`-V assumem turbo2 puro por-lado (38.25 MiB). Se o auto boundary mode 8 do turbo2-V
engajar também na assimétrica (como na simétrica turbo2-auto, lado ~51 MiB), o V-side real sobe -- não
re-medido diretamente neste pass (o `llama-cli` em `-st` não imprime a linha `KV self size`). turbo3/
turbo4-V não têm boundary -> VRAM exata. Coerência das 6 confirmada no e2e independente do valor de VRAM.

Antes do fix, `f16`-K abortava em `fattn.cpp:166 Not match KV type in vec: K=f16 V=turbo3` (RED).
NUNCA habilitar turbo-K + turbo-V juntos (assimetria = só UM lado turbo).

### Amostras reais (turbo KV ON)

- **f16 (A, referência):** "The sky appears blue because molecules in the Earth's atmosphere scatter
  sunlight. Shorter wavelengths of light, like blue and violet, are scattered more than longer wavelengths
  such as red and yellow. Although violet light is scattered even more than blue, our eyes are more
  sensitive to blue…"
- **turbo3 (B, headline):** "The sky appears blue because molecules in the atmosphere scatter shorter
  wavelengths of sunlight more effectively than longer wavelengths. Blue light … is scattered more than
  other colors because it has a shorter wavelength. This scattering, known as Rayleigh scattering, spreads
  the blue light in all directions, making the sky appear blue during the day." → **coerente e correto**.
- **turbo2 (auto mode 8):** "The sky is blue because molecules in the atmosphere scatter shorter wavelengths
  of light (like blue) more than longer wavelengths (like red). This scattering spreads the blue light in
  all directions, making the sky appear blue during the day…" → **coerente**.
- **turbo2 uniforme (`TURBO_LAYER_ADAPTIVE=0`, DEGRADADO):** "…The blue color of the sky is due to Rayleigh
  scattering of sunlight by molecules and small particles in the atmosphere. The blue sky is due to Rayleigh
  scattering of sunlight by molecules and small particles in the atmosphere. The blue color of the sky is
  due to Rayleigh scattering of." → **loop degenerado, NÃO usar**.
- **Prompt de raciocínio** (60 km em 45 min → km/h): f16, turbo3 e turbo2-auto produziram setup
  passo-a-passo correto (converter 45 min → 0,75 h; velocidade = dist/tempo) — coerentes nos 150 tokens.

### Veredito e config ÓTIMA

- **Sim, o turbo KV gera coerente** — para as configs **simétricas SYCL-safe**: `turbo3/turbo3`,
  `turbo4/turbo4` e `turbo2/turbo2` (com a borda auto mode 8). Decode ~74 t/s (≈ f16 77 t/s, −4%).
- **A config recomendada é `-ctk q8_0 -ctv turbo3`**, nenhuma das simétricas. Esta seção chamava
  originalmente `-ctk turbo2 -ctv turbo2` de "ÓTIMA" (5,65× menos VRAM de KV) e `-ctk turbo3 -ctv
  turbo3` de "default recomendado" (5,12×). As duas afirmações saíram em 2026-07-09 apoiadas só em
  coerência; o gate de perplexidade rodou pela primeira vez em 2026-07-28 e as derrubou — turbo2
  simétrico é **+41,6%** de PPL e turbo3 simétrico **+31,7%**, contra **+0,4%** do `q8_0 x turbo3`
  (tabela de qualidade neste documento). Texto coerente era o instrumento errado: um modelo 32% pior
  em perplexidade continua escrevendo frases fluentes, então o gate e2e não tinha como pegar isso — e
  não pegou.
- Turbo simétrico continua correto e suportado; é um tradeoff memória-primeiro, não um default. Se
  precisar de simétrico, `turbo4 x turbo4` (+5,1%) é o único dentro de um orçamento de 5%.
- **Depende do modo:** turbo2 **uniforme** (`TURBO_LAYER_ADAPTIVE=0`) degrada (repetição) — evitar.
  `TURBO_LAYER_ADAPTIVE=5/6/7` (bordas com tipos turbo mistos) **abortam** no FA-vec SYCL
  (`fattn.cpp:166`). Misto turbo-K/q8_0-V roda mas é marcado não-confiável no código — não recomendado.
- **Gate e2e:** `tests/test-e2e-turbo-kv.sh` (via ctest `test-e2e-turbo-kv`) trava essas 3 configs
  recomendadas e afirma coerência (keyword on-topic + sem repetição degenerada + sem abort). Verde na B580.
