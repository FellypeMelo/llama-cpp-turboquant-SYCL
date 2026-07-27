# STATE — TurboQuant SYCL fork

_Atualizado: 2026-07-09. Estado vivo do projeto. Ler junto com `TURBO_HANDOFF.md`, `docs/pt-BR/upstream-sync.md`, `docs/pt-BR/decisions.md`._

## Onde estamos

- **Branch canônico:** `feature/turboquant-kv-cache` @ `5874b9fe1` — intacto, NÃO tocado neste sync.
- **Sync CONCLUÍDO (SYCL):** `sync/upstream-2026-07` — `git merge upstream/master` (`fb30ba9a6`) feito,
  os **31 conflitos resolvidos** re-integrando turbo, **build SYCL verde** e **golden gate `test-sycl-turbo`
  verde (exit 0)** na Arc B580. Detalhes/evidência em `docs/pt-BR/upstream-sync.md` ("Resultado da execução")
  e `docs/pt-BR/testing.md`. ADR-0003 = Opção A (enum turbo 43–47); ADR-0004 = `GGML_SYCL_FA_ALL_QUANTS` OFF.
  **Merge commitado apenas com build+gate verdes; NÃO pushado.**
  Pendências honestas (não bloqueiam SYCL): PPL/bench gate (falta modelo puro-atenção + wikitext local);
  CUDA/Metal/Vulkan resolvidos mas não compilados (build SYCL-only); Vulkan turbo3 FA scalar/coopmat1
  não-funcional pós-merge (CM2 preservado) — best-effort não-Arc.
- **Perf-pass CONCLUÍDO (2026-07-09, Arc B580):** build-perf com F16+AOT+DNN (ver `docs/pt-BR/testing.md`).
  Golden `test-sycl-turbo` VERDE (22/22, cosine ~1.0) — flags de perf NÃO corrompem o turbo no Xe2.
  Bench (Qwen3-4B Q4_K_M): **prefill +102.8% (2.03x)**, decode -3.3% (ruído). Saída coerente. Números
  medidos em `docs/pt-BR/benchmarks.md`. As flags viraram a config oficial recomendada em `docs/pt-BR/testing.md`
  (com ressalva Xe2: re-rodar golden em outra arch). Nenhum source/commit de merge tocado no perf-pass.
- **PPL turbo-quality-gate AINDA PENDENTE:** o perf-pass mede tok/s, NÃO qualidade de tipo turbo.
  Qwen3-4B serve p/ velocidade mas NÃO p/ PPL turbo. Falta o modelo puro-atenção validado + `wikitext-2-raw`
  local p/ rodar `scripts/turbo-quality-gate.sh` (turbo3 PPL < 1.05x q8_0; ratio velocidade > 0.95 @4K).
- **Turbo KV e2e COERÊNCIA — FEITO (2026-07-09, Arc B580):** prova de geração real ponta-a-ponta com turbo
  KV ligado (não parity de kernel). Novo gate `tests/test-e2e-turbo-kv.sh` (ctest `test-e2e-turbo-kv`,
  labels `e2e;gpu;turbo`, SKIP 77 sem modelo) — **VERDE** via ctest na B580 (~15 s). Matriz completa medida
  (f16→turbo2) em `docs/pt-BR/benchmarks.md`. **Veredito: turbo KV gera COERENTE** nas simétricas SYCL-safe
  (`turbo3/turbo3` 5.12×, `turbo4/turbo4` 3.76×, `turbo2/turbo2` auto-mode-8 5.65×; decode ~74 t/s ≈ f16).
  **Config ÓTIMA (máx. compressão coerente): `-ctk turbo2 -ctv turbo2`** (5.65× menos VRAM-KV); default
  robusto `turbo3/turbo3`. Achados honestos: turbo2 uniforme (`TURBO_LAYER_ADAPTIVE=0`) DEGRADA (repetição);
  `TURBO_LAYER_ADAPTIVE=5/6/7` ABORTAM (`fattn.cpp:166`, tipos K/V turbo mistos). Auto-assimétrica NÃO
  engaja p/ Qwen3-4B (GQA 4:1 < limiar 6) — correto.

- **KV assimétrico (K preciso + V turbo) — FEITO (2026-07-09, Arc B580):** branch
  `feature/asymmetric-turbo-kv`. Habilitado o dispatch `f16`-K + turbo-V no FA-vec SYCL (3 rows curadas
  `FATTN_VEC_CASES_TURBO_D(F16, TURBO{2,3,4}_0)` + `fattn-vec-instance-f16-tq3.cpp`; `(F16,TURBO2/4_0)`
  implícitas). `FA_ALL_QUANTS` continua OFF (ADR-0005). Antes do fix, `f16`-K abortava em `fattn.cpp:166`
  (RED capturado). Golden `test-sycl-turbo` = **34 PASSED/0 FAIL**: `{q8_0,f16} x {turbo2,turbo3,turbo4}`
  golden + DECODE/GQA/mask cosine 1.000000 (f16 rel-MSE 0.0, q8_0 ~8e-4). e2e `test-e2e-turbo-kv.sh`
  estendido p/ as **6 assimétricas** (+ check anti-'?'): todas **COERENTES** na B580. Escopo: só G1+G5,
  head_dim=128; NÃO tocado prefill-tile/kv-padding/MLA. **NÃO pushado.** Qwen2.5-7B PPL rescue = PENDENTE
  (modelo ausente no disco; não baixado por instrução).

## Sync upstream 2026-07 — resumo do estado

| Métrica | Valor |
|---|---|
| Arquivos em conflito | 31 (bounded, caracterizados em `docs/pt-BR/upstream-sync.md`) |
| Auto-merges silenciosos de risco | `ggml.c`, `ggml-common.h` (BAIXO), `llama-graph.cpp` (+912, MÉDIO), `dmmv.cpp` (+916, MÉDIO) |
| Colisão dura | enum `ggml_type` slot 42 (Q2_0 vs turbo) → **decisão do dono pendente** (ADR-0003) |
| Resolvidos | **31 / 31** (turbo re-integrado sobre estrutura nova upstream) |
| Pendentes | 0 conflitos. Auto-merges verificados por build (3 quebras silenciosas achadas+corrigidas) |
| Turbo preservado? | **SIM — validado em runtime** (golden `test-sycl-turbo` exit 0, cosine ~1.0, Arc B580) |
| Gate build/teste | **VERDE (SYCL)** — build exit 0 + golden gate exit 0. PPL/bench pendente (falta modelo ref local) |

## Cobertura de teste (registro exigido pela diretriz do dono)

| Camada | Artefato | Estado |
|---|---|---|
| Unit / golden | `tests/test-sycl-turbo.cpp` (cosine ~0.99999, stress) | Existe; binário `build/bin/test-sycl-turbo.exe` já buildado no HEAD canônico. **Não re-rodado nesta sessão.** |
| Unit quant | `tests/test-turbo-quant.c` | Existe. Não re-rodado. |
| e2e / COERÊNCIA (turbo KV ON) | `tests/test-e2e-turbo-kv.sh` (ctest `test-e2e-turbo-kv`; keyword on-topic + anti-repetição + anti-abort nas configs turbo3/turbo4/turbo2) | **NOVO. VERDE na B580 via ctest (2026-07-09).** Prova geração real coerente com turbo KV ligado. SKIP 77 sem modelo. |
| e2e / qualidade+velocidade | `scripts/turbo-quality-gate.sh` (PPL turbo3 < 1.05× baseline; ratio velocidade > 0.95 @4K) | Existe. Prova geração real fim-a-fim via `llama-perplexity`. **Não re-rodado.** |
| CI/CD | `.github/workflows/tqp-sycl.yml` (windows-sycl + linux-sycl, artefatos + release em tag) | Existe, verde no HEAD canônico. **Não conflita no sync** (arquivo separado) → sobrevive. |

## Próximos passos (ordem)

1. **Dono decide ADR-0003** (numeração enum slot 42): há GGUF TQ3_1S/TQ4_1S em uso? → Opção A (alinhar upstream, recomendada) ou B (preservar IDs turbo).
2. Sessão **com oneAPI no PATH**: `git switch sync/upstream-2026-07 && git merge upstream/master`, resolver os 31 pelo playbook de `docs/pt-BR/upstream-sync.md` (ordem P0→P1→P2→P3).
3. Build SYCL (receita em `docs/pt-BR/upstream-sync.md` / `TURBO_HANDOFF.md`).
4. Gate: `test-sycl-turbo` verde + `turbo-quality-gate.sh` dentro de 5% + `tqp-sycl.yml` verde. Sem regressão bench > 5%.
5. Só então commitar o merge. **Não push** sem aprovação do dono.

## Nota sobre a diretriz TDD/CI/CD do coordenador (2026-07-09)

A diretriz cita "sprint Arc-Forge / OV text encoder / AttributeError / torch-XPU / pytest / OpenVINO" — isso é de **outro projeto** (stack Python/OpenVINO), não deste fork C++/SYCL de llama.cpp. As partes transferíveis foram aplicadas aqui:
- **CI GitHub Actions:** já existe (`tqp-sycl.yml`), sobrevive ao sync — ok.
- **e2e provando geração real:** `turbo-quality-gate.sh` (perplexity fim-a-fim) — é o e2e do fork; roda em Arc local.
- **Gate antes de fechar:** adotado — sync NÃO fecha sem golden + gate + CI verdes (ADR-0002).
- **Cobertura registrada:** tabela acima.

Especificidades pytest/torch/OV **não se aplicam** a este repo. Se a intenção era rodar no projeto Python (`Xe-Diffusion.cpp` / Arc-Forge), essa é uma tarefa separada deste sync.
