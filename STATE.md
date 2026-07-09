# STATE — TurboQuant SYCL fork

_Atualizado: 2026-07-09. Estado vivo do projeto. Ler junto com `TURBO_HANDOFF.md`, `docs/UPSTREAM_SYNC.md`, `docs/DECISIONS.md`._

## Onde estamos

- **Branch canônico:** `feature/turboquant-kv-cache` @ `5874b9fe1` — intacto, funcional (turbo2/3/4 KV cache validado em Arc B580, golden green, CI `tqp-sycl.yml` verde).
- **Sync em andamento:** `sync/upstream-2026-07` @ `5874b9fe1` (LIMPO). Trial-merge de `upstream/master` (`fb30ba9a6`, 2026-07-09) feito e **analisado**; merge **abortado** de propósito (ADR-0002). Base velha `7fc1c4ef` (2026-04-21) → base nova `fb30ba9a6`. Fork 230 commits à frente; upstream 1075.

## Sync upstream 2026-07 — resumo do estado

| Métrica | Valor |
|---|---|
| Arquivos em conflito | 31 (bounded, caracterizados em `UPSTREAM_SYNC.md`) |
| Auto-merges silenciosos de risco | `ggml.c`, `ggml-common.h` (BAIXO), `llama-graph.cpp` (+912, MÉDIO), `dmmv.cpp` (+916, MÉDIO) |
| Colisão dura | enum `ggml_type` slot 42 (Q2_0 vs turbo) → **decisão do dono pendente** (ADR-0003) |
| Resolvidos | 0 (nenhuma resolução assada — sem build p/ validar) |
| Pendentes | 31 + verificação semântica dos auto-merges |
| Turbo preservado? | Textual: SIM provisório. Runtime: **PENDENTE** (sem build/golden) |
| Gate build/teste | **PENDENTE** — toolchain SYCL não estava no PATH nesta sessão |

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
