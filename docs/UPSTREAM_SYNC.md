# UPSTREAM SYNC — 2026-07

> Mapa de zonas quentes + playbook determinístico de re-integração turbo.
> Companheiro dos ADR em `docs/DECISIONS.md`. Ler ambos antes de retomar o sync.

## Referências

| Item | Valor |
|---|---|
| merge-base fork↔upstream | `7fc1c4ef` (2026-04-21) "metal: workaround macOS GPU watchdog" |
| upstream/master alvo | `fb30ba9a6` (2026-07-09 10:15 -0700) "hexagon: tiling..." |
| fork HEAD canônico | `feature/turboquant-kv-cache` @ `5874b9fe1` (2026-07-09) |
| branch de sync | `sync/upstream-2026-07` @ `5874b9fe1` (LIMPO — trial-merge abortado) |
| commits fork à frente | 230 |
| commits upstream à frente | 1075 |
| estratégia | **merge** (ADR-0001) |
| estado geral | **PENDENTE** — trial-merge feito e analisado; resolução real + build não feita (ADR-0002) |

## Como retomar (próxima sessão, COM toolchain oneAPI)

```bash
git switch sync/upstream-2026-07
git merge upstream/master        # recria os 31 conflitos abaixo
# resolver arquivo-a-arquivo pelo playbook (seção "Mapa")
# build SYCL (ver "Build/validação") -> rodar tests/test-sycl-turbo + turbo-quality-gate.sh
# só então: git commit (merge)
```
Escape a qualquer momento: `git merge --abort`.

## Achado estrutural chave

- **Colisão de enum só no slot `ggml_type = 42`** (Q2_0 upstream vs turbo). Ver ADR-0003 — **decisão do dono pendente**. `llama_ftype` NÃO colide (take-both).
- **`ggml.c` type_traits usa designated initializers por nome** → auto-mergeou correto; compila assim que o enum de `ggml.h` for resolvido dando slot distinto a cada nome.
- **`llama-graph.cpp` (churn upstream +912) auto-mergeou textualmente limpo**, 42 refs turbo intactas incl. inverse-WHT no output do FA (`ggml_turbo_wht`, ~linha 2421). Risco = **silencioso/semântico**, só um build confirma.
- **SYCL é o backend que importa** (Intel Arc B580; ver `TURBO_HANDOFF.md`). CUDA/Metal/Vulkan = best-effort, não validados em Arc → prioridade menor no sync.

## Mapa das zonas quentes (31 arquivos em conflito + auto-merges silenciosos)

Legenda risco: **BAIXO** trivial/mecânico · **MÉDIO** re-integração cirúrgica · **ALTO** reescrita/validação obrigatória.
Prioridade: **P0** SYCL+tipos (turbo core Arc) · **P1** KV/graph llama · **P2** CUDA/Metal/Vulkan (não-Arc) · **P3** infra/testes/non-turbo.

### P0 — Fundação de tipos (todos backends)

| Arquivo | Conf | Risco | Lógica turbo em risco / receita |
|---|---|---|---|
| `ggml/include/ggml.h` | 1 | BAIXO* | enum `ggml_type`: colisão slot 42. *Risco baixo mecânico mas **gated por ADR-0003**. Aplicar numeração escolhida. |
| `gguf-py/gguf/constants.py` | 3 | BAIXO | Espelhar numeração do ggml.h em `GGMLQuantizationType` + adicionar Q2_0 e TQ3_1S/TQ4_1S no block-size map (`Q2_0:(64,2+16)`, `TQ3_1S:(32,2+2+12)`, `TQ4_1S:(32,2+2+16)`). `LlamaFileType.MOSTLY_*` = take-both. |
| `include/llama.h` | 1 | BAIXO | `llama_ftype`: take-both. Q1_0=40, Q2_0=41 (upstream), TQ3_1S=43, TQ4_1S=44 (nosso). Sem renumerar (42 fica livre). |
| `ggml/src/ggml.c` | 0 (auto) | BAIXO | Auto-mergeou. Verificar só que `[GGML_TYPE_Q2_0]` E `[GGML_TYPE_TURBOx]/[TQx]` coexistem no `type_traits` + no switch de `quantize` (linhas ~684/767-799/7830-7858). Compila após enum resolvido. |
| `ggml/src/ggml-common.h` | 0 (auto) | BAIXO | Auto-mergeou. Blocos turbo (`block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s`) intactos + `block_q2_0` upstream. Conferir static_asserts. |

### P0 — Backend SYCL (turbo core, Intel Arc)

| Arquivo | Conf | Churn upstream | Risco | Lógica turbo em risco / receita |
|---|---|---|---|---|
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | 7 | +2017 | MÉDIO | (a) include: take-both (`turbo-wht.hpp` + conv2d*). (b) device-enum refactor (`info.device_count`→`info.devices[i]`, vmm/usm/hw_info): **take upstream** — nosso lado é versão antiga sem turbo. (c) 3 guards de tipo turbo (dmmv exclusion ~L4414; `mul_mat_id` support TQ3/TQ4 ~L5736; op-support type list ~L5818): **re-adicionar tipos turbo às listas NOVAS do upstream**, merge das duas listas (turbo + Q1_0/MXFP4/NVFP4). |
| `ggml/src/ggml-sycl/mmvq.cpp` | 1 | +1538 | ALTO | mmvq turbo (TQ weight dp4a). Upstream reescreveu quase tudo em volta. Re-hook do path turbo na nova dispatch; **build obrigatório**. |
| `ggml/src/ggml-sycl/cpy.cpp` | 2 | +706 | ALTO | cpy turbo (KV set-rows-like). Re-integrar sobre nova estrutura de cpy. |
| `ggml/src/ggml-sycl/convert.cpp` | 1 | +83 | MÉDIO | Registro de dequant turbo. Adicionar cases turbo à tabela nova. |
| `ggml/src/ggml-sycl/set_rows.cpp` | 1 | +20 | MÉDIO | set_rows turbo (WHT). Conflito pequeno. |
| `ggml/src/ggml-sycl/dmmv.cpp` | 0 (auto) | +916 | MÉDIO(silent) | Auto-mergeou apesar de +916. **Verificar por build** que os hooks turbo dmmv sobreviveram semânticamente. |
| `turbo-quants.hpp`, `turbo-wht.cpp/.hpp`, `fattn*.hpp/.cpp`, `fattn-vec-instance-*tq3*` | 0 | novos/baixo | BAIXO(silent) | Arquivos novos ou pouco tocados. Sem conflito textual. Confirmar assinaturas que eles chamam (ex.: `fattn-common.hpp` upstream +12) não mudaram. |

### P1 — llama KV cache + graph

| Arquivo | Conf | Churn upstream | Risco | Lógica turbo em risco / receita |
|---|---|---|---|---|
| `src/llama-kv-cache.cpp` | 4 | +324 | ALTO | (1) ~L60 `ggml_mul_mat_aux` + InnerQ cross-TU state (per-channel equalization CUDA): upstream vazio → **keep ours**. (2) ~L142 bloco **auto-asymmetric turbo-K** (GQA>=6 → upgrade K p/ q8_0) + refactor upstream `n_layer`→`n_layer_all`/`n_layer_kv`: **preservar bloco turbo, adaptar às novas vars**. (3) ~L187 `mem_size` **+3 tensores turbo** (rotation + rotation_inv + innerq_scale_inv): preservar o +3, adaptar a `n_layer`. (4) ~L534 política de attention-rotation turbo (#21038 OFF por default no fork): preservar decisão. |
| `src/llama-graph.cpp` | 0 (auto) | +912 | MÉDIO(silent) | Auto-mergeou limpo, 42 refs turbo OK (inverse-WHT no FA output, innerq scale). **Validar por build** que o refactor upstream de `build_attn`/shapes não quebrou os hooks turbo. |
| `src/llama-context.cpp` | 2 | — | BAIXO-MÉDIO | Conferir handling de tipo turbo/KV. |
| `src/llama-model-loader.cpp` | 1 | — | BAIXO | Registro/load de tipo turbo. |

### P2 — CUDA (não-Arc, best-effort; precisa compilar se build CUDA for tentado)

| Arquivo | Conf | Risco | Receita |
|---|---|---|---|
| `ggml/src/ggml-cuda/fattn.cu` | 4 | MÉDIO | switch `K->type` com cases TURBO2/3/4 (guard D%64) → upstream extraiu helper `ggml_cuda_fattn_kv_type_supported()`. **Adicionar tipos turbo ao helper novo** (ou manter guard turbo antes dele). |
| `ggml/src/ggml-cuda/fattn-common.cuh` | 1 | MÉDIO | Nosso fix HIP (bypass mem-pool p/ f16 temp, evita OOM com KV quantizado) vs refactor upstream `ggml_cuda_flash_attn_ext_get_f16_extra_data`. Re-integrar o fix HIP na estrutura nova. |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | 1 | MÉDIO | Tabela MMA config: upstream adicionou set completo (112..576). Take upstream + conferir se nossa config custom 640/512 é necessária. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 4 | MÉDIO | Guard `is_tq_weight` (TQ4_1S/TQ3_1S → path dp4a fundido, fora de mmvq/mmq). Re-adicionar na dispatch refatorada. |
| `set-rows.cu`, `fattn-vec.cuh` | 0 (auto) | BAIXO(silent) | Auto-mergeados (nosso +929/+372, upstream +88/+29). Build-verify. |

### P2 — Metal (não-Arc)

| Arquivo | Conf | Risco | Nota |
|---|---|---|---|
| `ggml-metal-device.cpp` | 1 | MÉDIO | Registro pipeline turbo. |
| `ggml-metal-device.h` | 1 | BAIXO | Assinatura. |
| `ggml-metal-ops.cpp` | 1 | MÉDIO | Dispatch op turbo/WHT. |
| `ggml-metal.metal` | 3 | MÉDIO | Kernels turbo. (`turbo-matrices.h` 8207L + `turbo-wht.h` = novos, sem conflito.) |

### P2 — Vulkan (não-Arc)

| Arquivo | Conf | Churn upstream | Risco | Nota |
|---|---|---|---|---|
| `ggml-vulkan/ggml-vulkan.cpp` | 7 | +3121 | ALTO | Reescrita massiva upstream. Re-hook dos tipos turbo. Baixa prioridade (não-Arc). |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | 2 | — | MÉDIO | Registro de shaders turbo no gerador. |
| `vulkan-shaders/dequant_funcs_cm2.glsl` | 1 | — | MÉDIO | dequant turbo cm2. |
| `vulkan-shaders/flash_attn_base.glsl` | 1 | — | MÉDIO | FA base turbo. |
| shaders novos (`dequant_turbo3_0.comp`, `dequant_tq4_1s.comp`, `mul_mat_vec_tq4_1s.comp`, `turbo_wht.comp`, ...) | 0 | — | BAIXO | Novos, sem conflito. |

### P3 — Non-turbo custom + infra + testes

| Arquivo | Conf | Risco | Nota |
|---|---|---|---|
| `common/arg.cpp` | 1 | MÉDIO | **Non-turbo:** nosso ngram-mod speculative (`speculative.type`, `ngram_size_n`) vs API nova upstream (`speculative.types.push_back`, struct `ngram_mod`). Adotar API nova upstream com nossos valores. |
| `common/speculative.cpp` | 1 | MÉDIO | Idem, mesma feature speculative. |
| `tools/server/server-task.cpp` | 1 | BAIXO | Parse de param (provável KV type string). |
| `tests/test-backend-ops.cpp` | 1 | BAIXO | Registro de tipos turbo nos testes. |
| `tests/test-quantize-fns.cpp` | 1 | BAIXO | Idem quantize fns. |
| `.github/workflows/build.yml` | 0* | BAIXO | *Listado sem markers (mudança upstream no build.yml compartilhado). Nosso CI turbo é `tqp-sycl.yml` (SEPARADO, não conflita → sobrevive ao sync). Take upstream no build.yml. |
| `.devops/nix/package.nix` | 1 | BAIXO | Take upstream. |
| `.gitignore` | 1 | BAIXO | Take-both. |

## Funcionalidade turbo — está preservada?

**Evidência textual: SIM (provisório).** Todos os símbolos turbo core sobrevivem ao trial-merge:
- Tipos (`block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s`, enums) — intactos, só precisam renumerar slot 42 (ADR-0003).
- KV turbo logic (auto-asymmetric, +3 rotation tensors, InnerQ) — em conflito mas **preservável** (upstream não sobrescreveu a semântica, só renomeou `n_layer`).
- Graph inverse-WHT (`ggml_turbo_wht`) — auto-mergeou intacto.
- SYCL turbo primitives (turbo-quants, turbo-wht, fattn turbo, set_rows) — arquivos novos intactos; hooks nos arquivos reescritos são cirúrgicos (listas de tipo).

**Evidência de runtime: PENDENTE.** Nenhum build/golden-test rodou (ADR-0002). **Não declarar turbo preservado até `tests/test-sycl-turbo` verde + `turbo-quality-gate.sh` dentro de 5% PPL.**

## Riscos remanescentes

1. **[ALTO] Re-integração SYCL não-validada** (mmvq +1538, cpy +706, dmmv +916 auto-merged): maior fonte de bug silencioso. Só build+golden pega.
2. **[ALTO] Refactor `n_layer`→`n_layer_all`/`n_layer_kv`** em `llama-kv-cache.cpp`: se a adaptação do bloco turbo/mem-size errar a var, KV turbo aloca errado → crash ou lixo.
3. **[MÉDIO] Decisão de numeração (ADR-0003)** pode quebrar GGUFs TQ existentes se Opção A e existirem tais arquivos.
4. **[MÉDIO] `llama-graph.cpp` semântica**: 912 linhas de churn upstream em volta dos hooks turbo — auto-merge textual não garante shapes/API iguais.
5. **[BAIXO] CUDA/Metal/Vulkan**: precisam compilar mas não são validados em Arc; podem ficar quebrados sem afetar o path SYCL.

## Build / validação (próxima sessão)

Ambiente (de `TURBO_HANDOFF.md`): oneAPI 2026.0 (`C:\Program Files (x86)\Intel\oneAPI`), VS 2022.
```
# configurar
cmake -B build -G Ninja -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
# build alvos
cmake --build build --config Release -j 8 --target llama-cli llama-server llama-bench llama-quantize test-sycl-turbo
```
Fixes conhecidos do shell do agente: PATH do vswhere (`...VisualStudio\Installer`), `set "NoDefaultCurrentDirectoryInExePath="`, `/EHsc`, rodar `.bat` via **PowerShell** `cmd /c`. Matar procs que seguram a DLL ggml-sycl antes de relinkar.

Gate: `tests/test-sycl-turbo` (golden cosine ~0.99999) + `bash scripts/turbo-quality-gate.sh` (PPL turbo3 < 1.05× baseline, ratio velocidade > 0.95) + CI `.github/workflows/tqp-sycl.yml` verde.
