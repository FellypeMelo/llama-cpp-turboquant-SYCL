# STATE — TurboQuant SYCL fork

_Atualizado: 2026-07-09. Estado vivo do projeto. Ler junto com `TURBO_HANDOFF.md`, `docs/UPSTREAM_SYNC.md`, `docs/DECISIONS.md`._

## Onde estamos

- **Branch canônico:** `feature/turboquant-kv-cache` @ `5874b9fe1` — intacto, NÃO tocado neste sync.
- **Sync CONCLUÍDO (SYCL):** `sync/upstream-2026-07` — `git merge upstream/master` (`fb30ba9a6`) feito,
  os **31 conflitos resolvidos** re-integrando turbo, **build SYCL verde** e **golden gate `test-sycl-turbo`
  verde (exit 0)** na Arc B580. Detalhes/evidência em `docs/UPSTREAM_SYNC.md` ("Resultado da execução")
  e `QUALITY.md`. ADR-0003 = Opção A (enum turbo 43–47); ADR-0004 = `GGML_SYCL_FA_ALL_QUANTS` OFF.
  **Merge commitado apenas com build+gate verdes; NÃO pushado.**
  Pendências honestas (não bloqueiam SYCL): PPL/bench gate (falta modelo puro-atenção + wikitext local);
  CUDA/Metal/Vulkan resolvidos mas não compilados (build SYCL-only); Vulkan turbo3 FA scalar/coopmat1
  não-funcional pós-merge (CM2 preservado) — best-effort não-Arc.

## Sync upstream 2026-07 — resumo do estado

| Métrica | Valor |
|---|---|
| Arquivos em conflito | 31 (bounded, caracterizados em `UPSTREAM_SYNC.md`) |
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
| e2e / qualidade+velocidade | `scripts/turbo-quality-gate.sh` (PPL turbo3 < 1.05× baseline; ratio velocidade > 0.95 @4K) | Existe. Prova geração real fim-a-fim via `llama-perplexity`. **Não re-rodado.** |
| CI/CD | `.github/workflows/tqp-sycl.yml` (windows-sycl + linux-sycl, artefatos + release em tag) | Existe, verde no HEAD canônico. **Não conflita no sync** (arquivo separado) → sobrevive. |

## Próximos passos (ordem)

1. **Dono decide ADR-0003** (numeração enum slot 42): há GGUF TQ3_1S/TQ4_1S em uso? → Opção A (alinhar upstream, recomendada) ou B (preservar IDs turbo).
2. Sessão **com oneAPI no PATH**: `git switch sync/upstream-2026-07 && git merge upstream/master`, resolver os 31 pelo playbook de `UPSTREAM_SYNC.md` (ordem P0→P1→P2→P3).
3. Build SYCL (receita em `UPSTREAM_SYNC.md` / `TURBO_HANDOFF.md`).
4. Gate: `test-sycl-turbo` verde + `turbo-quality-gate.sh` dentro de 5% + `tqp-sycl.yml` verde. Sem regressão bench > 5%.
5. Só então commitar o merge. **Não push** sem aprovação do dono.

## Nota sobre a diretriz TDD/CI/CD do coordenador (2026-07-09)

A diretriz cita "sprint Arc-Forge / OV text encoder / AttributeError / torch-XPU / pytest / OpenVINO" — isso é de **outro projeto** (stack Python/OpenVINO), não deste fork C++/SYCL de llama.cpp. As partes transferíveis foram aplicadas aqui:
- **CI GitHub Actions:** já existe (`tqp-sycl.yml`), sobrevive ao sync — ok.
- **e2e provando geração real:** `turbo-quality-gate.sh` (perplexity fim-a-fim) — é o e2e do fork; roda em Arc local.
- **Gate antes de fechar:** adotado — sync NÃO fecha sem golden + gate + CI verdes (ADR-0002).
- **Cobertura registrada:** tabela acima.

Especificidades pytest/torch/OV **não se aplicam** a este repo. Se a intenção era rodar no projeto Python (`Xe-Diffusion.cpp` / Arc-Forge), essa é uma tarefa separada deste sync.
