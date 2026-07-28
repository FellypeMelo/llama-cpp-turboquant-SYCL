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

---

## ADR-0005 — f16-K + turbo-V assimétrico: 3 rows curadas + 1 instance file (FA_ALL_QUANTS fica OFF)

**Data:** 2026-07-09
**Status:** Aceito
**Branch:** `feature/asymmetric-turbo-kv` (de `feature/turboquant-kv-cache` @ `5602f58d0`)

### Contexto
Assimetria = K preciso (q8_0 ou f16) + V turbo ("V is free, K is everything"). O default de prod
recomendado (`-ctk q8_0 -ctv turboN`) ja dispatcha no fork (rows `Q8_0 x TURBOx` em `fattn.cpp`). A
config genuinamente NOVA e `f16`-K + turbo-V (K de precisao maxima, sem quant loss). O router
(`ggml_sycl_get_best_fattn_kernel`) ja roteava esse par p/ `BEST_FATTN_KERNEL_VEC` (o gate `KV_is_turbo`
so olha se ALGUM lado e turbo), mas o dispatch em `ggml_sycl_flash_attn_ext_vec` NAO tinha as rows
`FATTN_VEC_CASES_TURBO_D(F16, TURBOx_0)` -> caia no `GGML_ABORT` "Not match KV type in vec"
(`fattn.cpp:166`), abort alcancavel em runtime real, nao so assert defensivo.

### Decisao
Adicionar as 3 rows curadas `FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16, GGML_TYPE_TURBO{2,3,4}_0)` no bloco
`#else` (marcadas `// TURBO BEGIN/END`), cap D em {64,128} como as outras rows turbo-V, mais UM arquivo
de instancia `template-instances/fattn-vec-instance-f16-tq3.cpp` p/ o unico combo extern-declarado:
`(F16, TURBO3_0)` em D=64/128 (espelha `q8_0-tq3.cpp`).

- `(F16, TURBO2_0)` e `(F16, TURBO4_0)` sao instanciados IMPLICITAMENTE em `fattn.cpp` (turbo2/turbo4
  nao aparecem no eixo type_V do `EXTERN_DECL_FATTN_VEC_CASES`) -> nenhum instance file necessario.
- Semantica de rotacao ja correta no grafo: a Q-WHT forward gateia em `k->type` turbo e a inverse-WHT
  gateia em `v->type` turbo (`llama-graph.cpp`), entao K=f16 deixa Q NAO-rotada enquanto V=turbo ainda
  aplica a inverse-WHT na saida. Nenhuma edicao de grafo necessaria.
- `GGML_SYCL_FA_ALL_QUANTS` continua **OFF** (ADR-0004). As rows novas NAO exigem religa-lo: so
  precisam do 1 instance file curado; nao houve LNK2019 (build linka exit 0). DLL relinkada:
  build-sync (JIT) 60.3 MB; build-perf (AOT bmg-g21) 262.5 MB — ambas linkam limpo.

### Evidencia (Arc B580, build-sync JIT)
- Golden `test-sycl-turbo`: 34 PASSED / 0 FAIL. Assimetricas `{q8_0,f16} x {turbo2,turbo3,turbo4}`
  golden decode + DECODE/GQA/padding-mask com cosine 1.000000 (f16 rel-MSE 0.0, q8_0 rel-MSE ~8e-4).
  RED capturado antes do fix: `fattn.cpp:166 Not match KV type: K=f16 V=turbo3`.
- e2e `test-e2e-turbo-kv.sh`: as 6 assimetricas geram texto COERENTE (keyword on-topic, sem repeticao
  5-gram, sem abort/NaN, sem corrupcao '?').

### Evidencia (Arc B580, build-perf AOT — verify gentil 2026-07-09, ninja -j4 + Idle)
- Reconfirma sob AOT (bmg-g21): Golden 34/34 (12 asym cosine 1.000000, f16 decode rel-MSE ~2-3e-5,
  q8_0 ~5e-4; sem regressao nos 21 simetricos). e2e ctest 1/1, 9/9 configs coerentes (Qwen3-4B Q4_K_M).
- Revisao adversarial read-only (4 lentes): CONFIRMED_GREEN, zero refutacao / zero issue critico.
- PENDENTE: PPL rescue Qwen2.5-7B Q4_K_M (precisao absoluta long-context) — modelo ausente no disco.

---

## ADR-0006 — Prefill assimetrico entra no Option A, atras de um limiar de colunas

**Data:** 2026-07-28
**Status:** Aceito (validado na Arc B580, build-perf AOT bmg-g21)

### Contexto
O Option A (ver TURBOQUANT_SYCL.md secao 4) resolveu o prefill turbo dequantizando K/V para f16
transitorio e reusando o TILE f16 provado. Mas o portao era **simetrico**:

```cpp
if (!(is_turbo(K->type) && is_turbo(V->type) && K->type == V->type)) return false;
```

O default de producao recomendado pelo proprio fork e `-ctk q8_0 -ctv turbo3`, que e assimetrico e
portanto **nunca** entrava nesse atalho. O router forca VEC para qualquer lado turbo
(`fattn.cpp:297-299`) e o VEC e orientado a decode, entao o prefill assimetrico rodava no kernel
errado. Medido antes da mudanca (B580, AOT, Qwen3-4B Q4_K_M): **pp8192 = 95,7 t/s contra 798,5 do
f16**, ou seja ~12% da velocidade do f16 — enquanto o simetrico ja estava em paridade.

### Decisao
Generalizar o shim: exigir **V turbo**, com K em `{f16, q8_0, turbo}`. Cada lado nao-f16 recebe um
shadow f16 contiguo; um lado que ja e f16 passa direto, porque o TILE le K e V por strides
independentes (`stride_K2 = nb11/2`, `stride_V2 = nb21/2`, `fattn-tile.hpp:806-807`).

A semantica de rotacao nao precisou de nenhuma edicao de grafo: os portoes da Q-WHT forward
(`k->type`) e da inverse-WHT de saida (`v->type`) sao decididos na construcao do grafo, antes do op
de FA, entao trocar `src[1]`/`src[2]` por shadows f16 dentro do shim nao pode dessincronizar nada.

Tres formas ficam **de fora**, cada uma por um motivo:
- `(turbo K, f16 V)` nao tem row curada em `FATTN_VEC_CASES_TURBO_D`, entao o decode aborta de
  qualquer jeito. Deixar o prefill passar so adiaria o abort.
- `(turbo K, q8_0 V)` roteado pelo TILE mede rel-MSE **8,3e-3** com turbo2-K contra o limite de
  6e-3 (turbo3 1,9e-3, turbo4 1,2e-3). No VEC, o mesmo par mede **0,0**. Fica no VEC.
- **turbo de larguras diferentes** (turbo2 K com turbo3 V, por exemplo) tambem nao tem instancia vec.
  A primeira versao deste patch aceitava esses pares por acidente: `f16able` inclui os tipos turbo e
  `symmetric_turbo` dava false, entao o prefill rodava no TILE e o abort so aparecia no primeiro
  token de decode — depois do prompt inteiro processado. Antes do patch o portao exigia
  `K->type == V->type` e o abort era imediato. Pego por revisao adversarial de pre-merge, nao por
  teste; corrigido com `if (is_turbo(K->type) && K->type != V->type) return false;`. Alcancavel
  direto por `--cache-type-k turbo2 --cache-type-v turbo3` e pelos modos `TURBO_LAYER_ADAPTIVE` 5/6.

### Limiar de colunas: 16, uniforme
O shim dequantiza o cache K/V **inteiro**. Sob cache unificado o `llama-server` default
(`n_parallel = 4`, `kv_unified = true`, `tools/server/server.cpp:146-150`) empacota todos os slots
num unico ubatch, entao 3 ou mais slots decodificando produzem `Q->ne[1] >= 3` com `K->ne[3] == 1`.
Com o divisor original `Q->ne[1] <= 2`, **cada passo de decode em lote** arrastaria o cache inteiro
por um dequant para servir 3-4 colunas: ~3x de trafego de memoria, com break-even so por volta de
12-16 colunas. Chunks reais de prefill sao do tamanho de `n_batch` e passam de 16 com folga.

O limiar vale para o **simetrico tambem**. A primeira versao deixava o simetrico no `>2` original
por conservadorismo, mas isso e incoerente: o break-even e propriedade do shim (um passe pelo cache
inteiro amortizado sobre N colunas), nao do tipo de K. Manter 3 no simetrico preservaria
conscientemente a mesma patologia que o proprio ADR documenta — sob `-np 4` com cache unificado, o
shadow f16 chega a ordem de 10x o trafego que o VEC leria. Custo da uniformizacao: prompts com menos
de 16 tokens passam a fazer prefill no VEC, o que e desprezivel em termos absolutos.

Ambos os riscos (o widening no decode em lote e a incoerencia do limiar) foram encontrados por
revisao adversarial read-only, **nao** por bench — o bench de prefill sozinho so teria mostrado o
ganho.

### Descoberta colateral: o Q tambem e quantizado quando K e q8_0
`fattn-vec.hpp:124`:

```cpp
constexpr bool Q_q8_1 = type_K != GGML_TYPE_F16 && !K_is_turbo;
```

Com `K=q8_0` e `V=turbo`, o proprio **Q** vira q8_1 para o produto KQ. K f16 e K turbo escapam
disso. Medido no golden, mesma entrada, so mudando o numero de colunas (e portanto o kernel):

| n_q | caminho | rel-MSE |
|----:|---------|--------:|
| 8   | VEC (Q em q8_1) | 0,0551 |
| 16  | f16 TILE        | 0,0038 |

O TILE e ~21x mais preciso para esse par. Isso **nao** motivou baixar o limiar, porque o custo de
trafego em decode em lote continua real — mas registra que, acima de 16 colunas, o assimetrico ganha
velocidade **e** precisao ao mesmo tempo.

### Evidencia (Arc B580, build-perf AOT bmg-g21, Qwen3-4B Q4_K_M)
Prefill, `llama-bench -fa 1 -r 2`, mesmo binario AOT antes e depois:

| teste  | antes | depois | ganho | f16 (teto) |
|--------|------:|-------:|------:|-----------:|
| pp512  | 630,24 | **2539,98** | 4,0x | 2582,04 |
| pp2048 | 299,21 | **1766,34** | 5,9x | 1768,90 |
| pp8192 |  95,69 |  **797,96** | 8,3x |  798,49 |

Paridade com f16 de 98,4% (pp512) a 99,9% (pp2048/pp8192). Supera ate o turbo3 simetrico
(2487 / 1736 / 791), porque dequantizar K em q8_0 custa menos que em turbo3.

- Golden `test-sycl-turbo` sob AOT: **58 PASSED / 0 FAILED** (inclui os novos casos assimetricos em
  n_q=8 e n_q=16 e o contexto profundo n_kv=8192).
- e2e `test-e2e-turbo-kv.sh` sob AOT: **9/9 configs coerentes** — os 3 simetricos mais os 6
  assimetricos `{q8_0,f16} x {turbo2,turbo3,turbo4}`, todos com keyword on-topic, sem repeticao
  degenerada, sem abort/NaN.

### Ajustes nos gates
- A tolerancia do golden passou a depender do **caminho** que a forma toma no device, nao so de
  `n_q`: TILE f16 ~6e-3; VEC com K quantizado nao-turbo (o caso `Q_q8_1`) precisa de bound largo;
  V q8_0 precisa de 2e-3 em vez de 1e-3 porque o AOT agenda esse dequant com erro maior que o JIT
  (mesmos casos: 0,0 sob build-sync, 1,2-1,4e-3 sob build-perf, cosine 0,999995 nos dois).
- Novo caso de **contexto profundo** `n_kv=8192`. Todo o golden rodava em `n_kv=256`, o que da
  `ntiles_KQ=2` e praticamente nao aciona split-KV; 8192 da `ntiles_KQ=64` e exercita
  `flash_attn_combine_results`, que e onde o decode long-context realmente vive.
- `scripts/turbo-quality-gate.sh` reescrito: media o baseline q8_0 ao vivo em vez do
  `BASELINE_PPL=6.111` hardcoded (que desacoplava o gate do modelo sob teste), cobre os pares
  assimetricos, usa `awk` no lugar de `bc` (ausente no Git Bash) e sai 77 (SKIP) com mensagem
  acionavel quando falta binario, modelo ou dataset.

### Adiado (com plano, sem codigo)
**GQA sharing no VEC (`ncols2 > 1`).** O decode quantizado perde para f16 em profundidade porque o
f16 vai para o TILE batched-GQA, que le cada row K/V uma vez para todo o grupo GQA, enquanto todo
tipo quantizado cai no VEC com `launch_fattn<D, cols_per_block, 1, ...>` (`fattn-vec.hpp:632`) — o
literal `1` e o `ncols2`. Isso re-dequantiza cada row uma vez **por cabeca** do grupo (~4x no modelo
de referencia).

Uma passada de design read-only estabeleceu que `launch_fattn` e todo o caminho split-KV /
`flash_attn_combine_results` **ja sao ncols2-aware**; o trabalho fica confinado ao decode de indice
de head/sequence do kernel VEC. O bloqueio e **pressao de registrador**: em D=128 com V turbo,
`nthreads_V = 2` (`fattn-vec.hpp:113-115`) infla o acumulador `VKQ` para ~64 dwords/lane, entao
`ncols2 = 2` so cabe se `nthreads_V` subir junto, e `ncols2 = 4` nao cabe. Spill de GRF no Intel e o
modo de falha conhecido. Nao entra sem medicao propria.

**Perplexidade do turbo simetrico.** Ao consertar o quality gate ele rodou pela primeira vez e reprovou
`turbo3 x turbo3`. Medido (wikitext-2, Qwen3-4B Q4_K_M, `-c 512 --chunks 32`, B580 AOT), PPL vs f16
9,054: `q8_0 x turbo3` 9,091 (+0,4%), `f16 x turbo3` 9,104 (+0,6%), `turbo4 x turbo4` 9,511 (+5,1%),
`turbo3 x q8_0` 11,781 (+30,1%), `turbo3 x turbo3` 11,920 (+31,7%), `turbo2 x turbo2` 12,821 (+41,6%).

As linhas `turbo3 x q8_0` e `q8_0 x turbo3` isolam a causa: turbo3 no V custa 0,4%, no K custa 30%
(~75x). Pre-existente, sem relacao com este ADR. Os exemplos de uso da doc foram corrigidos para
`-ctk q8_0 -ctv turbo3`. **O golden e estruturalmente cego a isso**: ele compara o device contra um
golden de CPU construido a partir dos MESMOS dados quantizados, entao o erro de quantizacao cancela dos
dois lados — ele mede fidelidade de kernel, nao qualidade de quantizacao. Consequencia pratica: o gate
de perplexidade nao e opcional, e enquanto o simetrico nao for tratado (ou removido da recomendacao) o
script sai 1 de proposito. Em aberto: o salto turbo4 (+5,1%) -> turbo3 (+31,7%) e abrupto para 1,1 bit,
e turbo4 usa PolarQuant enquanto turbo2/3 usam codebook de centroides — o codebook de 8 niveis do
turbo3 pode estar rendendo abaixo do esperado. Nao investigado.

**Divisor de `nsm` no Xe2.** `ggml-sycl.cpp:146` faz `nsm = max_compute_units / 16`. A B580 reporta
**160** compute units, dando `nsm = 10`, mas o BMG G21 tem 20 Xe-cores (160 vector engines / 8). Se
o divisor correto for 8, `nsm` esta subestimado em 2x — e ele alimenta `max_blocks_per_sm` e
`parallel_blocks` em `launch_fattn` (`fattn-common.hpp:1130,1155`) alem de `count-equal.cpp:48-59`.
Nao mexer sem confirmar a contagem real e sem gatear por arquitetura.
