# DECISIONS (ADR) — TurboQuant SYCL fork

> Registro de decisões arquiteturais. Formato ADR curto. Português caveman, termos técnicos em inglês.

---

## ADR-0001 — Estratégia de sync com upstream (2026-07): MERGE, não rebase

**Data:** 2026-07-09
**Status:** Aceito (parcial — sync ainda não concluído, ver `docs/pt-BR/upstream-sync.md`)
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

Trade-off aceito: histórico fica com um merge commit "grande"; mitigado por este ADR + `docs/pt-BR/upstream-sync.md` documentando o que entrou.

---

## ADR-0002 — Esta sessão para no trial-merge + playbook; branch fica LIMPO (merge abortado)

**Data:** 2026-07-09
**Status:** Aceito

### Contexto
- Sessão escopada como **preparação e análise de risco**, explicitamente NÃO o rebase completo às cegas.
- Trial-merge executado: **31 arquivos em conflito** (bounded, caracterizado — ver mapa em `docs/pt-BR/upstream-sync.md`). Muitos arquivos hot auto-mergearam textualmente (`ggml.c`, `llama-graph.cpp`, `ggml-common.h`).
- **Toolchain SYCL indisponível nesta sessão:** oneAPI 2026.0 está instalado em `C:\Program Files (x86)\Intel\oneAPI` mas `icx`/`icpx` não estão no PATH (setvars não sourced), e de todo modo **não dá para buildar uma árvore mid-merge** com 31 conflitos abertos.
- Regra do dono: **NÃO declarar sucesso sem validação**; perder turbo = falha total.

### Decisão
Após o trial-merge caracterizar o risco, **`git merge --abort`** — deixar a árvore de trabalho LIMPA (branch `sync/upstream-2026-07` de volta em `5874b9fe1`), e entregar um **playbook determinístico** de resolução em `docs/pt-BR/upstream-sync.md`.

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

**Divisor de `nsm` no Xe2 (ver tambem ADR-0007).** `ggml-sycl.cpp:146` faz `nsm = max_compute_units / 16`. A B580 reporta
**160** compute units, dando `nsm = 10`, mas o BMG G21 tem 20 Xe-cores (160 vector engines / 8). Se
o divisor correto for 8, `nsm` esta subestimado em 2x — e ele alimenta `max_blocks_per_sm` e
`parallel_blocks` em `launch_fattn` (`fattn-common.hpp:1130,1155`) alem de `count-equal.cpp:48-59`.
Nao mexer sem confirmar a contagem real e sem gatear por arquitetura.

---

## ADR-0007 - turbo4-K com V turbo mais barato (largura mista curada)

**Data:** 2026-07-28
**Status:** Aceito (validado na Arc B580, build-perf AOT bmg-g21)
**Branch:** `feature/turbo4-k-mixed-width`

### Contexto
As medicoes de perplexidade do ADR-0006 estabeleceram que turbo no V custa ~0,4% e turbo no K custa
~30%, e que **turbo4 e o joelho de qualidade no K** (turbo4 simetrico +5,1%, turbo3 simetrico +31,7%).
Isso implica um ponto de fronteira que o turbo simetrico nao alcanca: turbo4 no K com um V mais
barato. Essa combinacao nao existia.

Pior: ela **ja era alcancavel e abortava**. O router aceita qualquer par com algum lado turbo
(`fattn.cpp` `KV_is_turbo` -> `BEST_FATTN_KERNEL_VEC`) e `ggml_sycl_flash_attn_ext_supported` apenas
consulta o router, entao o backend **anunciava suporte** para `-ctk turbo4 -ctv turbo3` e depois caia
no `GGML_ABORT("Not match KV type in vec")` por falta de row curada. Nao e feature nova apenas; e
conserto de um abort alcancavel em producao.

### Decisao
Adicionar duas rows curadas, `FATTN_VEC_CASES_TURBO_D(TURBO4_0, TURBO3_0)` e `(TURBO4_0, TURBO2_0)`,
e um allowlist no shim de prefill para admitir exatamente esses dois pares de largura mista.

**Zero arquivos de instancia**, e e aqui que isto diverge do ADR-0005. `TURBO4_0` nao aparece no eixo
`type_K` do `EXTERN_DECL_FATTN_VEC_CASES` (`fattn-vec.hpp`), entao nenhum dos dois pares tem
`extern template` suprimindo a instanciacao implicita - ambos instanciam na TU do `fattn.cpp`,
exatamente como as rows `(TURBO4_0, TURBO4_0)` e `(TURBO4_0, Q8_0)` que ja enviam sem arquivo algum.
O ADR-0005 precisou de `fattn-vec-instance-f16-tq3.cpp` so porque `(F16, TURBO3_0)` **estava**
declarado extern. `GGML_SYCL_FA_ALL_QUANTS` continua OFF (ADR-0004); o link fecha sem LNK2019.

**Armadilha registrada no codigo:** nao adicionar `EXTERN_DECL_FATTN_VEC_CASES(D, GGML_TYPE_TURBO4_0)`
numa futura arrumacao - isso suprimiria a instanciacao implicita de todas as rows turbo4-K de uma vez
e produziria LNK2019.

Risco de recurso e nulo, e por um motivo estrutural: `nthreads_KQ` e `vec_dot_KQ` derivam so de
`type_K`; `nthreads_V`, `V_rows_per_thread` e `dequantize_V` derivam so de `type_V`; e os branches
`K_is_turbo`/`V_is_turbo` colapsam turbo2/3/4 num unico caso. Logo `(turbo4, turbo3)` tem footprint de
registrador e SLM **identico** ao `(turbo4, turbo4)` que ja envia. So muda o codebook dentro das duas
funcoes de dequant, e elas nunca se tocam no corpo do kernel.

### Larguras reversas ficam de fora
`(turbo3 K, turbo4 V)` custa exatamente os mesmos bytes que `(turbo4 K, turbo3 V)` - 118 B por 256
valores - mas poe o quantizador grosseiro no lado que domina o erro. Estritamente dominado: mesma
memoria, ~6x a penalidade de perplexidade. Mesmo argumento, pior, para qualquer par com turbo2 no K.

### Evidencia (Arc B580, build-perf AOT, Qwen3-4B Q4_K_M)
Perplexidade (wikitext-2, `-c 512 --chunks 32`, `TURBO_LAYER_ADAPTIVE=0`), e memoria KV @64k por
aritmetica exata de bytes/bloco (conferida contra a tabela da secao 6 do deep-dive em f16, turbo3 e
turbo4):

**Atencao ao modo adaptativo.** `-ctv turbo2` **auto-liga o modo 8** sem env nenhum
(`llama-kv-cache.cpp`: `type_v == TURBO2_0 && n_layer >= 8`), o que troca 4 camadas de fronteira por
q8_0/q8_0. Logo `turbo4 x turbo2` tem dois regimes distintos e a tabela precisa separar os dois. Os
demais pares nao auto-ligam nada.

| K x V | modo | PPL | vs f16 9,054 | KV @64k | vs f16 |
|-------|------|----:|-------------:|--------:|-------:|
| q8_0 x turbo3   | uniforme |  9,091 |  +0,4% | 3348 MiB | 2,75x |
| turbo4 x turbo4 | uniforme |  9,511 |  +5,1% | 2448 MiB | 3,76x |
| **turbo4 x turbo2** | **8 (default)** | **9,275** | **+2,4%** | **2176 MiB** | **4,23x** |
| **turbo4 x turbo3** | **uniforme** | **9,564** | **+5,6%** | **2124 MiB** | **4,34x** |
| turbo4 x turbo2 | 0 (opt-out) |  9,724 |  +7,4% | 1836 MiB | 5,02x |
| turbo3 x turbo3 | uniforme | 11,920 | +31,7% | 1800 MiB | 5,12x |

Leituras:
- `turbo4 x turbo3` **domina estritamente** o turbo4 simetrico: 15% menos KV por 0,5 ponto percentual.
- **`turbo4 x turbo2` no default e o melhor ponto dos dois**: +2,4% de PPL a 4,23x, ou seja melhor
  qualidade que `turbo4 x turbo3` com memoria praticamente igual. As 4 camadas q8_0 de fronteira
  compram mais qualidade do que custam em bytes. Uma versao anterior desta tabela reportou so o
  regime `TURBO_LAYER_ADAPTIVE=0` e fez esse par parecer pior do que e.
- Com `TURBO_LAYER_ADAPTIVE=0` o mesmo par vira o ponto de memoria maxima util: 5,02x contra os 5,12x
  do turbo3 simetrico, com +7,4% em vez de +31,7%.

Memoria calculada por aritmetica de bytes/bloco (por camada por lado @64k, 36 camadas,
n_embd_k_gqa 1024: f16 128 MiB, q8_0 68, turbo4 34, turbo3 25, turbo2 17), conferida contra a tabela
da secao 6 do deep-dive onde as duas se sobrepoem.

**Relacao com o gate de qualidade.** Contra o baseline q8_0 (9,091), `turbo4 x turbo3` fica +5,2% e
`turbo4 x turbo2` em LA=0 fica +7,0% - ou seja, ambos **estourariam o orcamento de 5%** de
`scripts/turbo-quality-gate.sh`. Por isso nao entraram no `CONFIGS` do gate: sao pontos oferecidos a
quem esta limitado por memoria, nao configuracoes que o projeto afirma estarem dentro do orcamento.
O par no default (+2,4% vs f16, +2,0% vs q8_0) passaria, mas depende do modo 8 e por ora fica fora
tambem, para o gate nao medir uma coisa e o usuario receber outra.

Com o K fixo em turbo4 a degradacao do V e suave (9,511 -> 9,564 -> 9,724 de 4 para 3 para 2 bits),
o que reconfirma que o V e quase de graca.

- Golden `test-sycl-turbo` sob AOT: **71 PASSED / 0 FAILED**. Os pares novos medem rel-MSE **0,0** em
  decode e em DECODE+GQA+padding-mask, e 1,5-2,7e-3 no caminho TILE.
- e2e `test-e2e-turbo-kv.sh`: **11/11 coerentes** (as 9 anteriores mais os dois pares novos).

### Nao e recomendacao de default
O default de producao continua `-ctk q8_0 -ctv turbo3` (+0,4%). Estes pares custam ~5 pontos
percentuais de perplexidade a mais; servem a quem precisa de contexto que nao cabe de outro jeito.

### Achado colateral: o modo 8 ignora o `--cache-type-v`
`llama-kv-cache.cpp`, branch do `adaptive_mode == 8`: nas camadas nao-fronteira ele faz
`layer_type_v = GGML_TYPE_TURBO2_0` **incondicionalmente**, descartando o tipo que o usuario pediu.
Medido: `-ctv` turbo2, turbo3 e turbo4 sob `TURBO_LAYER_ADAPTIVE=8` dao PPL identica ate a quarta casa
(9,2754), porque sao literalmente a mesma configuracao. O modo 8 so auto-liga quando `type_v` ja e
turbo2, entao o default nao e afetado - mas quem setar o env manualmente recebe um V diferente do que
pediu, sem aviso. Nao corrigido aqui (mexe em politica de modo adaptativo, fora do escopo deste ADR).

Isso tambem explica por que a tabela de memoria do deep-dive lista turbo2 em 816 MiB por lado e nao
612: aquele numero ja embute as 4 camadas q8_0 do modo 8 (4x68 + 32x17 = 816).

---

## ADR-0008 - head_dim fora de {64,128}: o fallback silencioso para CPU

**Data:** 2026-07-29
**Status:** Aceito, mas **parcialmente superado pelo ADR-0009** - D=256 passou a ser suportado.
O mecanismo do fallback silencioso descrito aqui continua valendo para head_dim fora de
{64,128,256}, e D=512 continua quebrado.

### O mecanismo, que e o ponto principal
`ggml_sycl_flash_attn_ext_supported` e literalmente
`ggml_sycl_get_best_fattn_kernel(...) != BEST_FATTN_KERNEL_NONE`. Quando o router devolve NONE, o
ggml **nao aborta e nao avisa**: ele agenda aquele no do grafo no backend CPU. O KV cache continua
na GPU em formato turbo; o que muda de lugar e a operacao de atencao, entao passa-se a pagar copia
GPU->CPU, atencao em CPU e copia de volta, a cada passo.

Resultado correto, sem uma linha de log, e roughly 10x mais lento. Essa e a pior classe de bug do
subsistema: nao quebra nada que qualquer gate consiga ver.

### Cobertura real de head_dim (atualizado pelo ADR-0009)
> Quando este ADR foi escrito, `FATTN_VEC_CASES_TURBO_D` cobria {64,128}. O ADR-0009 acrescentou
> 256. O paragrafo abaixo fica como registro do estado da epoca; leia o ADR-0009 para o atual.

`FATTN_VEC_CASES_TURBO_D` cobre D em {64,128}. Qualquer outro head_dim com turbo cai no caminho
acima. O `q8_0`/`f16` cobrem ate 512 via `FATTN_VEC_CASES_ALL_D`, entao a assimetria e so do turbo.

### Evidencia (Arc B580, llama-bench pp2048, modelo qwen35 9B Q4_K_M, head_dim 256)
| config | pp2048 | tg32 |
|--------|-------:|-----:|
| q8_0 x turbo3 |  139,37 | 40,67 |
| q8_0 x q8_0   | 1401,26 | 44,93 |
| f16 x f16     | 1426,91 | 41,61 |

O turbo fica em **10,2x mais lento** no prefill. O decode quase nao muda porque e limitado por banda
de pesos e a atencao pesa pouco ali.

### Tentativa de habilitar D=256, e por que foi revertida
Feita e desfeita em 2026-07-29. Tres camadas de bloqueio, nesta ordem:

1. **Link.** Adicionar `FATTN_VEC_CASE(256, ...)` quebra com LNK2019 nos 4 combos extern-declarados
   (`(F16,TURBO3_0)`, `(Q8_0,TURBO3_0)`, `(TURBO3_0,Q8_0)`, `(TURBO3_0,TURBO3_0)`), porque
   `EXTERN_DECL_FATTN_VEC_CASES` e invocada em D=256 e suprime a instanciacao implicita. Os pares com
   TURBO4_0 no K linkam sozinhos - nao estao na grade extern. Confirma o ADR-0004 na pratica.
2. **JIT.** Com os arquivos de instancia corrigidos o link passa e o **kernel nao compila no device**,
   derrubando o programa inteiro: D=128 para de funcionar junto. O comentario do macro culpava
   pressao de registrador por `nthreads_KQ=1`, o que **esta desatualizado** - o SYCL usa
   `128/cpy_nb` desde a cooperative-register rework. O teto real e **memoria local compartilhada**:
   `ne_combine = nwarps * V_cols_per_iter * D` da 8*8*256 = 16384 floats = **64 KB** em D=256, que e
   o limite por work-group no Xe2. Em D=128 sao 32 KB e cabe.
3. **Correcao.** Baixar `nthreads_V` resolve o orcamento de SLM (16 KB) mas quebra o resultado:
   golden com cosine **-0,34** e rel-MSE 1480. O dequant do V turbo assume que cada lane cobre um
   bloco de 128 inteiro, premissa que cai quando `V_cols_per_iter` diminui.

`nthreads_V` esta preso dos dois lados. D=256 exige trabalho de kernel, nao ajuste de constante.

### O caminho viavel
O **f16 TILE lida com D=256 corretamente** - o shim de prefill mediu rel-MSE 3,5e-4 nesses casos
antes da falha de JIT. Rotear turbo com D>128 para o TILE em vez do VEC evita o kernel VEC por
completo: sem SLM, sem premissa escondida, sem risco de GRF. Custo: o TILE e orientado a lote e
decode de 1 coluna nele e ineficiente - mas o baseline atual e atencao em CPU, que perde para quase
qualquer coisa na GPU. E a mesma estrutura que o CUDA documenta para o proprio teto do vec
(`ggml-cuda/fattn.cu`: "head_dim > 256 cannot use VEC (falls through to TILE)").

### Ponto cego dos gates
**Nenhum dos tres gates detecta o fallback para CPU.** O golden compara valores e a CPU produz os
valores certos; o e2e compara texto e o texto sai coerente; a perplexidade compara qualidade e a
qualidade nao muda. Todos passam com a atencao rodando em CPU. Um teste que afirme **onde** a
operacao rodou vale mais, aqui, do que mais um teste que afirme o que ela calculou.

### Licao de metodo
Tres interpretacoes foram feitas antes de medir neste dia, e as tres cairam na primeira medicao:
- "o modelo e 75% SSM, entao o KV nao e o gargalo" -> f16 faz 1427 t/s no mesmo modelo;
- "o fix assimetrico esta ativo e mensuravel neste workload" -> o turbo nem roda em head_dim 256;
- "o bloqueio de D=256 e pressao de registrador" (comentario no codigo) -> e SLM.

Medir custa minutos. Interpretar estrutura custou dois ciclos de build e uma conclusao publicada
errada.

---

## ADR-0009 - head_dim 256 no kernel VEC: quatro bugs, nenhum deles registrador

**Data:** 2026-07-29
**Status:** Aceito (validado na Arc B580, build-perf AOT bmg-g21)

### Contexto
O ADR-0008 registrou que head_dim fora de {64,128} com turbo faz a flash-attention inteira cair na
CPU, silenciosamente, ~10x mais lenta. A restricao estava documentada em `fattn.cpp` como pressao de
registrador vinda de `nthreads_KQ=1`. Essa atribuicao estava **errada**, e provavelmente e o motivo
de a restricao ter durado tanto: "precisa de reescrita de registrador" desencoraja tentar, enquanto
os bloqueios reais eram localizados. Alem disso `nthreads_KQ` nao e 1 no SYCL desde a
cooperative-register rework; usa `128/cpy_nb`.

### Os quatro bloqueios, na ordem em que apareceram
Cada um so ficou visivel depois que o anterior saiu do caminho. Foram resolvidos um por vez, cada um
com seu proprio veredito de teste, justamente para nao repetir o erro dos primeiros ciclos, em que
duas mudancas foram aplicadas juntas e a falha ficou inatribuivel.

1. **LNK2019.** Adicionar `FATTN_VEC_CASE(256, ...)` quebra o link nos 4 combos extern-declarados
   (`(F16,TURBO3_0)`, `(Q8_0,TURBO3_0)`, `(TURBO3_0,Q8_0)`, `(TURBO3_0,TURBO3_0)`), porque
   `EXTERN_DECL_FATTN_VEC_CASES` e invocada em D=256 e suprime a instanciacao implicita. Os pares com
   TURBO4_0 no K linkam sozinhos - nao estao na grade extern. Confirma o ADR-0004 na pratica.
   **Resolvido:** definicoes explicitas nos 4 arquivos de instancia.

2. **SLM.** O combine encenava `nwarps * V_cols_per_iter` vetores parciais de tamanho D de uma vez.
   Como `nthreads = max(128, D)`, o numero de warps cresce com D: em D=256 sao 16*8*256 = **128 KB**,
   o dobro do teto por work-group do Xe2. O kernel nao compila no device e derruba o programa
   inteiro - D=128 para de funcionar junto.
   **Resolvido:** encenar `combine_warp_chunk` warps por vez com a soma corrente em registrador. A
   reducao e elementwise em D, sem dependencia cruzada, entao o agrupamento e livre.
   **Validado independentemente:** forcar chunk=4 tambem em D=128 (2 passes em vez de 1) manteve os
   71 casos D=128 verdes, provando que o caminho multi-passe esta correto por si.

3. **Descompasso host/device.** `ggml_sycl_flash_attn_ext_vec_case_impl` calculava
   `nthreads` a partir de um helper de host que devolve 128 fixo, enquanto o kernel computa
   `max(128, D)`. Em D>128 o kernel acreditava ter o dobro das threads e dos warps que recebeu: o
   combine somava sobre 16 warps existindo 8, e `KQ[j*nthreads + k]` indexava alem do que as threads
   lancadas escreveram. Resultado: lixo (cosine ~0). Em D<=128 os dois lados dao 128, e por isso
   ficou invisivel desde sempre.
   **Este nao e um bug do turbo.** Atinge qualquer tipo de KV no vec acima de D=128, e
   `FATTN_VEC_CASES_ALL_D` sempre emitiu instancias em 256 e 512 para f16/q4_0/q5_0/q8_0.
   **Resolvido:** o host passa a computar a mesma formula.

4. **Precisao do TILE.** Com o VEC funcionando, sobrou um K turbo em D=256 pelo f16 TILE medindo
   rel-MSE 2,2e-2 (turbo3 x turbo3) e 6,2e-3 (turbo4 x turbo3) contra um limite de 6e-3, enquanto os
   **mesmos pares pelo VEC medem exatamente 0,0** e K preciso pelo TILE mede 2-3e-4. Dados e rotacao
   corretos; e a soma em f16 de 256 termos rotacionados que perde terreno.
   **Resolvido:** K turbo em D>=256 fica no VEC (o shim recusa), espelhando a exclusao de
   `(turbo K, q8_0 V)` do ADR-0006.

### Evidencia (Arc B580, build-perf AOT, qwen35 9B Q4_K_M, head_dim 256)
`llama-bench` pp2048, antes -> depois:

| config | antes | depois | vs f16 |
|--------|------:|-------:|-------:|
| q8_0 x turbo3 |  139,37 | **1474,95** | 99,9% |
| q8_0 x turbo4 |      -  | **1471,50** | 99,7% |
| q8_0 x q8_0   | 1401,26 | 1464,30 | 99,2% |
| f16 x f16     | 1426,91 | 1476,65 |    -   |

**10,6x** no par turbo. tg32 40,67 -> 44,97.

Repare que `q8_0 x q8_0` e `f16 x f16` tambem subiram sem que o caminho deles fosse tocado: e o
bloqueio 3 beneficiando todo tipo de KV em D=256. Confirmado direto pelo golden, que ganhou casos
`f16xf16`, `q8_0xq8_0` e `q4_0xq4_0` em D=256 - todos passam.

- Golden `test-sycl-turbo` sob AOT: **85 PASSED / 0 FAILED**.
- e2e `test-e2e-turbo-kv.sh` sob AOT: **11/11 coerentes**.
- DLL AOT: 253 -> 269,7 MB (**+6,6%**) pelos kernels D=256.

### head_dim 512 continua quebrado, e nao e do turbo
Reproduzido ao adicionar casos: `f16 x f16` e `q8_0 x q8_0` em D=512, n_q=1, dao cosine **0,039** e
**0,047**, rel-MSE na casa de 1e7. Ja estava quebrado **antes** do bloqueio 3 ser corrigido (o
descompasso ali era de 4x) e continua depois, entao ha uma terceira causa especifica de D=512 que nao
foi encontrada. `FATTN_VEC_CASES_ALL_D` emite essas instancias e o router as alcanca, entao qualquer
modelo de head_dim 512 no caminho vec do SYCL e afetado, com qualquer tipo de KV. Os casos foram
removidos do gate para ele nao nascer vermelho; a reproducao esta em comentario em
`tests/test-sycl-turbo.cpp` e reativa-la e uma edicao so.

### Divergencia com upstream, que vai conflitar
O upstream **ja corrigiu o bloqueio 3**, no commit `c1063ac9d` ("sycl: set fattn_vec_nthreads to 256
for Battlemage", PR #25205, 2026-07-14), que **nao** esta no nosso HEAD. A abordagem deles e melhor
na estrutura: apagam os dois helpers e passam `nthreads` como parametro de template, eliminando a
dupla fonte de verdade em vez de sincroniza-la. A semantica tambem difere - eles querem 256 threads
na Battlemage em **qualquer** D (e ganho de desempenho), enquanto aqui e `max(128, D)`.

Eles tambem bateram no mesmo teto de SLM e escolheram **capar**: o commit traz
`// 256 threads would overflow the 64 KB work-group local memory at D == 512, so keep 128 there`.
O fatiamento do bloqueio 2 remove essa necessidade em D=256; em D=512 nao ajuda, porque a causa la e
outra (ver secao acima).

**O proximo `git merge upstream/master` (ADR-0001) vai conflitar em `fattn-vec.hpp`, e nao e conflito
textual:** exige decidir se o combine fatiado composta com o `nthreads` por arquitetura deles. A
resolucao provavelmente certa e adotar a estrutura do upstream (parametro de template) e manter o
fatiamento por cima, o que permitiria remover o cap de D=512 deles - mas so depois de achar a
terceira causa.

Achado por revisao adversarial olhando `git log` do upstream, fora do escopo dos arquivos do diff.
Nenhuma das quatro lentes de codigo teria encontrado.
