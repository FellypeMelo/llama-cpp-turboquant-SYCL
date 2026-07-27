# UPSTREAM SYNC — 2026-07

> Mapa de zonas quentes + playbook determinístico de re-integração turbo.
> Companheiro dos ADR em `docs/pt-BR/decisions.md`. Ler ambos antes de retomar o sync.

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
| estado geral | **RESOLVIDO + VALIDADO (SYCL)** — 31 conflitos resolvidos, merge feito, build SYCL verde, golden gate verde. Ver "Resultado da execução" abaixo. |

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

---

## Resultado da execução (2026-07-09, sessão com toolchain)

**Merge feito e resolvido.** `git merge upstream/master` (fb30ba9a6) sobre `sync/upstream-2026-07`.
Os 31 conflitos foram resolvidos re-integrando a lógica turbo sobre a estrutura nova do upstream
(NÃO theirs/ours cego). ADR-0003 = Opção A aplicada (Q2_0=42, turbo 43–47, COUNT=48).

### Build SYCL — VERDE
- Toolchain: oneAPI 2026.0 (icx/icpx), Ninja, Win10 SDK 10.0.26100. Recipe correto em `docs/pt-BR/testing.md`.
- `cmake --build build-sync --target test-sycl-turbo llama-cli` → **exit 0**, `ggml-sycl.dll` (59 MB),
  `test-sycl-turbo.exe`, `llama-cli.exe` linkados. 0 erros.
- **3 quebras silenciosas de auto-merge que só o build pegou (corrigidas):**
  1. `ggml.c` static_assert `GGML_OP_COUNT == 97` → **98** (op turbo `GGML_OP_TURBO_WHT` + ops novos upstream).
  2. `ggml-sycl.cpp` chamava `get_sycl_env(...)` → upstream renomeou p/ `ggml_sycl_get_env(...)`.
  3. **`GGML_SYCL_FA_ALL_QUANTS`**: upstream passou a `#define`-ar em `common.hpp` (fork tinha OFF).
     Isso ativou o branch FA-vec que faz odr-use de `TURBO3_0 × {F16,Q4_0,Q4_1,Q5_0,Q5_1}` e
     `TURBO3_0 @ D256/512` — combos com `extern template` decl mas SEM instância explícita (só existem
     `tq3-tq3`, `q8_0-tq3`, `tq3-q8_0` @ D=64/128; turbo GRF-spilla @ D≥256 no Intel). → LNK2019.
     **Fix:** manter `GGML_SYCL_FA_ALL_QUANTS` OFF (config validada do fork; restaura path FA `#else`
     curado). Marcado `// TURBO BEGIN/END` em `common.hpp`. Custo: FA SYCL cobre F16/Q4_0/Q8_0 +
     turbo (não Q4_1/Q5_0/Q5_1 standard — que o fork nunca suportou). Re-ligar ALL_QUANTS exige gerar
     a matriz completa de instâncias turbo primeiro.

### Gate turbo — VERDE (evidência real, Arc B580)
`build-sync/bin/test-sycl-turbo.exe` sob oneAPI runtime → **exit 0**, todos PASSED:
- TURBO2/3/4_0 quant: MSE 0.0, **cosine 1.000000**.
- TQ3_1S weight mul: cosine 1.000000 · TQ4_1S: cosine 0.999843.
- `GGML_OP_TURBO_WHT` fwd + fwd→inv round-trip (gs=32/64/128): PASSED, max|diff| ~3e-7.
- Inverse-WHT value parity dev-vs-cpu (gs=64/128): max|dev-cpu| = 0.0.
- set_rows multi-group TURBO2/3/4 (ne00=1024): cosine 1.000000.
- FA turbo golden parity TURBO2/3/4 (D=128, n_q=1/8): cosine 0.999986–1.000000.
- FA turbo DECODE+GQA+padding-mask TURBO2/3/4: cosine 1.000000.
- Flash-Attention com KV cache TURBO3_0: graph_compute status 0, PASSED.

### Pendências honestas
- **`turbo-quality-gate.sh` (PPL fim-a-fim)**: NÃO rodado — o modelo validado (Qwen3-4B puro-atenção)
  e o wikitext não estão presentes nesta máquina; o único .gguf local é híbrido (Gated Delta Net,
  turbo não-validado nele por design). O golden `test-sycl-turbo` cobre paridade numérica turbo
  (quant+WHT+FA+KV) diretamente na GPU. Rodar o PPL gate quando o modelo puro-atenção estiver disponível.
- **Bench sem regressão >5%**: não medido nesta sessão (mesmo motivo — precisa do modelo de referência).
- **CUDA/Metal/Vulkan**: resolvidos (0 markers) por sub-agentes, **não compilados** (build SYCL-only).
  Vulkan: turbo3 FA nos paths scalar/coopmat1 ficou não-funcional pós-merge (upstream moveu o dequant
  de FA p/ `flash_attn_dequant.glsl` novo, sem case TURBO3_0 — fora do escopo dos 4 arquivos do agente);
  CM2 turbo3 preservado. Best-effort não-Arc; follow-up documentado.

## Checklist de touch-points turbo inline (meta: `git merge upstream` futuro quase-zero conflito)

Arquivos do upstream com hooks turbo inline (re-aplicar/conferir a cada sync). Marcados `// TURBO BEGIN/END`
onde prático. `arquivo:símbolo` → o que preservar:

| Arquivo | Símbolo / local | Hook turbo a preservar |
|---|---|---|
| `ggml/include/ggml.h` | `enum ggml_type` | slots 43–47 turbo (ADR-0003 A); realinhar se upstream ocupar 43+ |
| `ggml/src/ggml.c` | `type_traits[]`, `GGML_OP_NAME/SYMBOL[]`, `static_assert(GGML_OP_COUNT==N)` | entries turbo + op TURBO_WHT; **bumpar o assert** |
| `ggml/src/ggml-common.h` | `block_turbo2/3/4_0`, `block_tq3_1s/tq4_1s` | structs + static_asserts |
| `gguf-py/gguf/constants.py` | `GGMLQuantizationType`, `LlamaFileType`, block-size map | espelhar numeração |
| `include/llama.h` | `llama_ftype` | `MOSTLY_TQ3_1S=43/TQ4_1S=44` (take-both) |
| `ggml/src/ggml-sycl/common.hpp` | `GGML_SYCL_FA_ALL_QUANTS` | manter **OFF** (senão LNK2019 turbo FA-vec) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | `ggml_sycl_tq_convert_q8` (TQ_NATIVE), dmmv-exclusion, MUL_MAT TQ guard, SET_ROWS list, `GGML_OP_TURBO_WHT` dispatch+support | re-add nas listas/dispatch novos |
| `ggml/src/ggml-sycl/{mmvq,cpy,convert,set_rows,dmmv}.cpp` | dispatch por tipo | cases turbo (take-both c/ tipos novos upstream) |
| `ggml/src/ggml-sycl/fattn.cpp` | `ggml_sycl_flash_attn_ext_vec` (`#else` branch) | combos `FATTN_VEC_CASES_TURBO_D` (turbo×turbo, q8_0×turbo, turbo×q8_0 @ D64/128) |
| `ggml/src/ggml-sycl/fattn-vec.hpp` | `EXTERN_DECL_FATTN_VEC_CASES(*, TURBO3_0)` | extern decls turbo3 |
| `src/llama-kv-cache.cpp` | `ggml_mul_mat_aux`+InnerQ state, auto-asymmetric K, `+3` mem_size, rotation policy (`if(other)`/DeepSeek), adaptive-mode (`hparams.n_layer()`) | preservar; adaptar a renomes upstream |
| `src/llama-context.cpp` | turbo K/V head-dim padding (2×) | usa `hparams.n_layer()` (era campo) |
| `src/llama-graph.cpp` | `ggml_turbo_wht` fwd/inv, `get_turbo_innerq_scale_inv` | inverse-WHT no output FA (auto-merge; validar por build) |
| `src/llama-model-loader.cpp` | `llama_ftype_name` | cases `TQ3_1S/TQ4_1S` no estilo prefix novo |
