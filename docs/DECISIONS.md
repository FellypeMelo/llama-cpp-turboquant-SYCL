# DECISIONS (ADR) — TurboQuant SYCL fork

> Registro de decisões arquiteturais. Formato ADR curto. Português caveman, termos técnicos em inglês.

---

## ADR-0001 — Estratégia de sync com upstream (2026-07): MERGE, não rebase

**Data:** 2026-07-09
**Status:** Aceito (parcial — sync ainda não concluído, ver `docs/UPSTREAM_SYNC.md`)
**Branch:** `sync/upstream-2026-07` (criado de `feature/turboquant-kv-cache` @ `5874b9fe1`)

### Contexto
- merge-base fork↔upstream = `7fc1c4ef` (2026-04-21).
- Alvo: `upstream/master` = `fb30ba9a6` (2026-07-09).
- Fork = **230 commits** custom à frente do merge-base. Upstream = **1075 commits** à frente. Gap ~2.5 meses.
- Upstream reescreveu pesado as zonas onde o TurboQuant vive (especialmente backend **SYCL**: `ggml-sycl.cpp` +2017 linhas, `mmvq.cpp` +1538, `dmmv.cpp` +916, `cpy.cpp` +706; e `llama-graph.cpp` +912, `llama-kv-cache.cpp` +324, Vulkan `ggml-vulkan.cpp` +3121).
- Regra do dono: preservar funcionalidade turbo acima de tudo; resolver conflito = **re-integrar** lógica turbo sobre a base nova, não escolher theirs cego.

### Decisão
**Usar `git merge upstream/master` (um merge commit), NÃO `git rebase` dos 230 commits.**

Motivos:
1. Rebase de 230 commits replay-aria cada commit turbo contra 1075 commits upstream → conflitos repetidos no MESMO arquivo dezenas de vezes (ex.: cada commit que tocou `ggml-sycl.cpp` reconflita). Merge resolve cada arquivo **uma vez**, em bloco.
2. Merge preserva a história turbo intacta (não reescreve os 230 commits) — importante para auditoria e para o CI/release existente.
3. A re-integração turbo precisa de visão do estado FINAL de cada arquivo (ours vs theirs vs base-3), que o merge diff3 dá diretamente.

Trade-off aceito: histórico fica com um merge commit "grande"; mitigado por este ADR + `UPSTREAM_SYNC.md` documentando o que entrou.

---

## ADR-0002 — Esta sessão para no trial-merge + playbook; branch fica LIMPO (merge abortado)

**Data:** 2026-07-09
**Status:** Aceito

### Contexto
- Sessão escopada como **preparação e análise de risco**, explicitamente NÃO o rebase completo às cegas.
- Trial-merge executado: **31 arquivos em conflito** (bounded, caracterizado — ver mapa em `UPSTREAM_SYNC.md`). Muitos arquivos hot auto-mergearam textualmente (`ggml.c`, `llama-graph.cpp`, `ggml-common.h`).
- **Toolchain SYCL indisponível nesta sessão:** oneAPI 2026.0 está instalado em `C:\Program Files (x86)\Intel\oneAPI` mas `icx`/`icpx` não estão no PATH (setvars não sourced), e de todo modo **não dá para buildar uma árvore mid-merge** com 31 conflitos abertos.
- Regra do dono: **NÃO declarar sucesso sem validação**; perder turbo = falha total.

### Decisão
Após o trial-merge caracterizar o risco, **`git merge --abort`** — deixar a árvore de trabalho LIMPA (branch `sync/upstream-2026-07` de volta em `5874b9fe1`), e entregar um **playbook determinístico** de resolução em `UPSTREAM_SYNC.md`.

Motivos (escolha de engenharia sênior, deliberada):
1. **Nenhuma resolução minha seria validável nesta sessão** (sem build). Assar decisões semânticas turbo (re-integração de dispatch SYCL, refactor `n_layer`, guards de tipo) não-validadas numa árvore parcial que ficaria dias parada = risco de bug silencioso herdado — exatamente o que o dono proíbe.
2. **Uma árvore mid-merge trava o worktree principal**: git recusa trocar de branch com unmerged paths. Deixar o repo do dono preso num merge de 1075 commits por dias é hostil ao uso normal.
3. **Não se perde nada:** o merge é reproduzível em 1 comando (`git merge upstream/master`) e o playbook dá receita arquivo-por-arquivo. O "estado parcial" pedido está representado pelo **branch criado + playbook exaustivo**, não por uma árvore conflitada frágil.

Consequência: a resolução real (+ build + golden test) fica para a **próxima sessão com toolchain**. Gate de build/teste = **PENDENTE** (não declarado verde).

---

## ADR-0003 — Colisão de enum `ggml_type` no slot 42 (Q2_0 vs TurboQuant): RESOLVIDO Opção A

**Data:** 2026-07-09 (aceito 2026-07-09, sessão de execução do sync)
**Status:** **Aceito — Opção A** (dono confirmou: não há GGUF turbo salvos que precisem continuar carregáveis)

### Decisão do dono
Opção A escolhida. Alinhar com upstream: `Q2_0=42`, e mover os tipos turbo para 43–47:
`TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47, GGML_TYPE_COUNT=48`.
Motivo: não existem GGUFs quantizados como TQ3_1S/TQ4_1S em uso (path validado = pesos Q4_K_M + KV turbo runtime). Zero divergência permanente de numeração → syncs futuros não reconflitam Q2_0. Custo (mudança de ID de TQ3_1S/TQ4_1S) é inofensivo por não haver arquivos afetados.

Aplicado em: `ggml/include/ggml.h`, `gguf-py/gguf/constants.py` (`GGMLQuantizationType` + block-size map), coerente com `ggml.c`/`ggml-common.h` (designated initializers por nome). `include/llama.h` / `LlamaFileType` = take-both (sem renumerar; 42 fica livre no ftype).

---

### Contexto (original)

**Data:** 2026-07-09
**Status original:** Proposto (precisa confirmação do dono antes de finalizar o sync)

### Contexto
Única colisão *dura* de numeração do sync:
- **Upstream** adicionou `GGML_TYPE_Q2_0 = 42` (novo weight quant), `GGML_TYPE_COUNT = 43`.
- **Fork** já usava `42..46` para turbo: `TURBO2_0=42, TURBO3_0=43, TURBO4_0=44, TQ3_1S=45, TQ4_1S=46`, `COUNT=47`.
- (`llama_ftype` / `GGMLQuantizationType.MOSTLY_*` NÃO colidem: o fork pulou 41/42 e usou 43/44 → take-both limpo. A colisão é só no enum `ggml_type` e no espelho `GGMLQuantizationType`.)
- `ggml.c` usa designated initializers `[GGML_TYPE_XXX] = {...}` (por nome, não posição) → compila certo com qualquer numeração, desde que os nomes existam e `COUNT` seja grande o bastante.

### Ponto de decisão (hinge)
**Existe algum arquivo `.gguf` já quantizado como TQ3_1S/TQ4_1S que precise continuar carregável?**
(turbo2/3/4 são tipos de **KV cache runtime**, NÃO vão pro GGUF → renumerar é inofensivo pra eles. Só TQ3_1S/TQ4_1S são weight types gravados no GGUF.)

- **Se NÃO** (provável — path validado usa pesos Q4_K_M + KV turbo; TQ weight quant é experimental) → **Opção A (recomendada): alinhar com upstream.**
  `Q2_0=42, TURBO2_0=43, TURBO3_0=44, TURBO4_0=45, TQ3_1S=46, TQ4_1S=47, COUNT=48`.
  Vantagem: zero divergência permanente de numeração → syncs futuros não reconflitam Q2_0.
  Custo: IDs de TQ3_1S/TQ4_1S mudam (45→46, 46→47) — quebra GGUFs TQ existentes (assumidos inexistentes).

- **Se SIM** → **Opção B: preservar IDs turbo.**
  `TURBO2_0=42..TQ4_1S=46` (inalterados), `Q2_0=47`, `COUNT=48`.
  Vantagem: GGUFs turbo existentes continuam válidos.
  Custo: divergência permanente de Q2_0 vs upstream → todo sync futuro reconflita esse slot; GGUFs Q2_0 do fork ficam incompatíveis com upstream (Q2_0 é novo/inusado no fork, custo baixo).

Qualquer que seja a escolha, **numeração precisa ficar idêntica** em: `ggml/include/ggml.h`, `gguf-py/gguf/constants.py` (`GGMLQuantizationType` + block-size map), e coerente com `ggml.c`. `include/llama.h` e `LlamaFileType` = take-both (sem renumerar).

**Recomendação:** Opção A, salvo o dono confirmar que há GGUFs TQ em uso.

---

## ADR-0004 — `GGML_SYCL_FA_ALL_QUANTS` fica OFF no fork (link do FA turbo)

**Data:** 2026-07-09 (sessão de execução do sync)
**Status:** Aceito

### Contexto
Upstream passou a `#define GGML_SYCL_FA_ALL_QUANTS` por padrão em `ggml-sycl/common.hpp` (habilita
todos os quant types no flash-attention SYCL). O fork nunca definiu esse macro — sua config validada
usa o branch FA-vec `#else` (F16/Q4_0/Q8_0 + combos turbo curados).

O macro ON ativa o branch que faz **odr-use** de `ggml_sycl_flash_attn_ext_vec_case<D, TURBO3_0, V>`
para todo `V ∈ {F16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0}` e para `D ∈ {256,512}`. Esses combos têm `extern
template` decl em `fattn-vec.hpp` (suprime instanciação implícita) mas SÓ existem 3 instâncias
explícitas (`tq3-tq3`, `q8_0-tq3`, `tq3-q8_0`, em D=64/128 — turbo GRF-spilla em D≥256 no Intel,
ver `FATTN_VEC_CASES_TURBO_D`). Resultado com o macro ON: **LNK2019 unresolved externals** no link
de `ggml-sycl.dll`. (turbo2/turbo4 linkam por instanciação implícita — não têm extern decl.)

### Decisão
Manter `GGML_SYCL_FA_ALL_QUANTS` **desabilitado** (comentado, marcado `// TURBO BEGIN/END` em
`common.hpp`). Restaura o path FA `#else` validado do fork.

- **Custo:** FA SYCL cobre KV F16/Q4_0/Q8_0 (standard) + turbo2/3/4 — não Q4_1/Q5_0/Q5_1 standard.
  O fork nunca suportou esses no FA (não é regressão vs baseline do fork).
- **Divergência de upstream:** 1 linha, marcada e documentada (reconflita trivialmente em syncs futuros).
- **Reverter (re-ligar ALL_QUANTS)** exige gerar a matriz completa de instâncias FA-vec turbo
  (turbo × todos os tipos, D=64..512) — grande, e a maioria dos combos nunca ocorre em runtime
  (turbo K só pareia com turbo V ou q8_0). Fora de escopo até haver necessidade real.

**Evidência:** com o macro OFF, build SYCL linka (exit 0) e `test-sycl-turbo` passa verde na Arc B580.
