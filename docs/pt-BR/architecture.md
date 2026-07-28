# TurboQuant+ no SYCL — Aprofundamento técnico

> Trazendo quantização rotacionada de 2/3/4 bits para o cache KV em GPUs Intel (Arc / Xe2) no `llama.cpp`, com até **7,5× menos memória de cache KV** que fp16 e **prefill em paridade com fp16**.

Este documento conta a história de engenharia por trás deste fork: o que é o TurboQuant, como ele foi portado para o backend SYCL, os bugs que precisaram ser resolvidos para chegar à corretude, e o retrato honesto de desempenho em hardware real (Intel Arc B580 / Battlemage).

**Hardware de validação:** Intel Arc B580 (Battlemage / Xe2, `bmg_g21`), 12 GB. **Modelo:** Qwen3-4B Q4_K_M (head_dim 128, GQA 4:1, 36 camadas).

---

## 1. O problema: o cache KV é o muro de memória

Em inferência de contexto longo, o cache KV — não os pesos — domina a VRAM. Um cache KV em fp16 para o Qwen3-4B em 64k de contexto ocupa **9,2 GB**, o que sozinho quase preenche uma placa de 12 GB e não deixa espaço para crescer o contexto. A mitigação usual, o KV em `q8_0`, só reduz isso pela metade (1,9×).

O **TurboQuant** comprime o cache KV bem mais forte — para ~2,1 bits/valor — mantendo a qualidade de geração intacta, ao *rotacionar* os vetores de chave/valor antes de quantizá-los.

## 2. A ideia central: rotacionar, depois quantizar

A quantização ingênua de baixa precisão em K/V de atenção falha porque suas distribuições por canal têm outliers pesados — alguns poucos componentes grandes explodem a escala de quantização e destroem os pequenos.

O TurboQuant aplica primeiro uma rotação **Walsh–Hadamard Transform (WHT)** a cada vetor de head-dim. Uma rotação Hadamard de sinal aleatório é (a) ortogonal (preserva norma, então os scores de atenção não mudam em expectativa) e (b) um *achatador de outliers* — ela mistura cada componente em cada saída, transformando uma distribuição espinhosa em uma quase-Gaussiana que quantiza limpo com um codebook fixo minúsculo.

A rotação é um butterfly rápido em registrador, O(D log D), sem multiplicação de matriz:

```
rotate(x):  x *= 1/‖x‖            # normaliza
            x  = SIGNS1 ⊙ x       # sinais aleatórios (fixos por posição)
            x  = hadamard_butterfly(x)   # radix-2, log2(D) estágios
            x *= 1/sqrt(D)        # escala ortonormal  (D=128 → 0.08838834)
            x  = SIGNS2 ⊙ x       # segunda camada de sinais aleatórios
```

Os valores quantizados então são apenas um índice em um **codebook Lloyd–Max ótimo** ajustado a `N(0, 1/D)` (a distribuição pós-rotação), escalado pela norma por vetor armazenada.

### Onde a rotação vive no grafo
Rotacionar dentro do kernel de atenção a cada token a cada passo seria desperdício e fácil de errar. Em vez disso, a rotação é encaixada no grafo de computação:

- **K / V:** rotacionados **uma vez** no momento da escrita no cache (o kernel turbo `set_rows` do SYCL rotaciona e depois quantiza; a norma corrigida é armazenada por bloco).
- **Q:** pré-rotacionado por um nó de grafo dedicado (`GGML_OP_TURBO_WHT`) antes do score de atenção.
- **Output:** des-rotacionado pela WHT inversa após a atenção.

Como a rotação é ortogonal, `softmax(Q·Kᵀ)` é invariante a ela, então o único custo é o butterfly barato — nunca uma re-quantização no loop quente.

## 3. Os três formatos

Os três usam block size 128 (= head_dim, um bloco por grupo de rotação, sem normas redundantes). Os tamanhos em bytes abaixo são a verdade **garantida por `static_assert`** (alguns comentários inline no header estão desatualizados, de um design anterior de 32 valores):

| Tipo | Layout (por 128 valores) | Bytes | bits/val | vs fp16 |
|------|-------------------------|------:|---------:|--------:|
| `turbo2` | norm(fp16) + índices de 2 bits | **34** | 2,13 | **7,5×** |
| `turbo3` | norm(fp16) + índices de 2 bits + sinais de 1 bit (3 bits) | **50** | 3,13 | **5,1×** |
| `turbo4` | norm(fp16) + rnorm(fp16) + índices PolarQuant de 4 bits | **68** | 4,25 | **3,8×** |

- `turbo2` — 4 centroides simétricos `{-0,133462, -0,039994, 0,039994, 0,133462}` (Lloyd–Max para `N(0,1/128)`). Compressão máxima; qualidade é marginal, melhor em camadas que toleram isso.
- `turbo3` — 8 níveis (magnitude de 2 bits + sinal de 1 bit). **Melhor equilíbrio** — qualidade quase-fp16 com ~5× de economia.
- `turbo4` — 16 níveis, PolarQuant empacotado em nibbles. Qualidade mais segura.

## 4. Dois caminhos de atenção — e por que o prefill precisou de um diferente

O flash-attention SYCL do `llama.cpp` tem dois kernels com pontos de otimização opostos:

- **VEC** (`fattn-vec.hpp`) — orientado a decode, 1–2 colunas por bloco. Dequantiza cada elemento de K/V dentro do produto interno.
- **TILE** (`fattn-tile.hpp`) — orientado a batch/prefill, GEMM em f16 com tiling.

O KV turbo era inicialmente roteado **exclusivamente para o VEC**. Isso estava correto para decode, mas foi catastrófico para o **prefill**: o prefill turbo rodava a ~394 t/s contra 1271 do f16/q8_0 — um **gap de 3,2×** que se alargava para **7×** em 8k de contexto, porque o VEC processa o prefill uma ou duas colunas por vez.

### A correção — dequantizar para f16 e reaproveitar o kernel TILE já comprovado ("Opção A")
Em vez de escrever um loader turbo-TILE nativo frágil (os blocos K/V têm 34/50/68 bytes, strides não-potência-de-dois — o scaffolding existente produzia lixo), o caminho de prefill foi resolvido **desacoplando o dequant da atenção**:

1. Um pequeno kernel SYCL com stride dequantiza as views turbo K/V (permutadas, não-contíguas) para um **scratch f16 contíguo** no pool de computação.
2. O kernel **f16 TILE** já testado em batalha roda sobre esse scratch.

É correto porque o turbo já armazena valores rotacionados (Q é rotacionado no grafo, output des-rotacionado), então a sombra f16 é numericamente a entrada real de atenção. Resultado: **o prefill salta para paridade com fp16 em qualquer comprimento**, sem nenhuma cirurgia de stride no kernel quente.

| config | pp512 antes → depois | pp8192 antes → depois |
|--------|---------------------|----------------------|
| turbo3 | 394 → **1244** (f16 1267) | 82 → **577** (f16 580) |
| turbo4 | → **1249** | |

O decode continua no VEC, então a economia de memória é totalmente preservada.

## 5. Os bugs entre "compila" e "correto"

KV rotacionado de baixa precisão não perdoa — um off-by-one num stride de bit empacotado corrompe o cache silenciosamente. As correções conquistadas com esforço:

- **O `set_rows` que matava a linha única** — o kernel turbo de escrita no cache só escrevia a linha 0 de cada batch, deixando o resto do cache KV como lixo. A geração parecia plausível e depois divergia. Corrigido para escrever todas as linhas.
- **Rotação dupla** — Q era rotacionado tanto pelo novo nó de grafo quanto por código inline remanescente do kernel antigo → Q rotacionado duas vezes. A rotação inline foi removida.
- **Leitura errada do half2-V** — o tile V era carregado como `half2` bruto, sem dequantizar o V turbo. O tipo do loader foi corrigido.
- **Guarda de tipo do TILE f16** — uma checagem de veracidade em `type_K` despachava errado f16/q8_0 pelo caminho turbo, produzindo lixo. A guarda foi corrigida.
- **Abort do turbo2 em KV misto** (`Not match KV type in vec`, `0xC0000409`) — um modo adaptativo por camada criava camadas de fronteira `K=turbo2 / V=q8_0` sem instância VEC correspondente. Corrigido adicionando os casos mistos K=turbo/V=q8_0 **e** um novo "modo 8" simétrico e adaptativo, para que toda camada fique com o mesmo tipo dos dois lados.
- **Crash no context-shift** (`0xC0000005`) — o caminho de shift do K quantizado desreferenciava um tensor de rotação nulo, e não existe cast SYCL turbo↔f32. O K-shift turbo ainda não está implementado, então `get_can_shift()` agora retorna `false` para turbo → o context-shift é **desabilitado graciosamente** em vez de travar.

Toda correção foi travada atrás de um **teste numérico golden** no device (`tests/test-sycl-turbo.cpp`): cosseno ≥ 0,99999 vs. a referência f16 em turbo2/3/4, KV misto e D∈{64,128}.

## 6. Desempenho — o retrato honesto

**Memória do cache KV (MiB, Qwen3-4B, via `common_memory_breakdown`):**

| ctx | f16 | q8_0 | turbo4 | turbo3 | turbo2 |
|----:|----:|-----:|-------:|-------:|-------:|
| 16k | 2304 | 1224 | 612 | 450 | 408 |
| 64k | 9216 | 4896 | 2448 | 1800 | 1632 |

Economia vs. fp16 @64k: **turbo2 5,6×** (modo adaptativo 8; turbo2 puro = 7,5×), **turbo3 5,1×**, **turbo4 3,8×** (contra 1,9× do q8_0). fp16 @64k quase esgota a B580 de 12 GB; o turbo roda 64k com 6+ GB livres (~256k de contexto alcançável).

**Prefill / decode (llama-bench, -fa 1):** o prefill acompanha o fp16 em qualquer comprimento (§4). O decode em contexto raso fica próximo do fp16 (turbo3 tg128 70,4 contra f16 75,6).

**Decode em profundidade — a limitação honesta:**

| profundidade | f16 | turbo3 | q8_0 |
|------:|----:|-------:|-----:|
| d0 | 75,9 | 70,8 | 72,3 |
| 8k | 57,6 | 28,9 | 26,1 |
| 32k | 33,9 | 10,1 | 8,9 |

O decode turbo fica mais lento em contexto profundo — **mas isso não é um defeito do TurboQuant.** O `q8_0` colapsa de forma **idêntica** (o turbo3 inclusive supera ele em profundidade). O gargalo é o dequant por elemento dentro do loop interno do VEC, inerente a **todo** decode de KV quantizado; o f16 só continua rápido porque pula o dequant. Então o turbo entrega velocidade de decode na classe do `q8_0` **com até 2,7× menos memória que o `q8_0`** — essa é a vitória. Fechar o gap para o f16 em profundidade exigiria um kernel de decode fundamentalmente diferente (o caminho VEC da CUDA também não fecha).

## 7. O que foi explorado e conscientemente adiado

- **Kernel de prefill XMX (joint_matrix)** — uma prova de conceito de Estágio 0 confirmou que o motor de matriz da Intel funciona na B580 (um tile DPAS fp16→f32, erro 1e-10, JIT, sub-group 16). **Parado**: como o prefill *já* está em paridade com fp16, o ganho projetado é marginal (~1,6% em pp512). O design completo + as formas DPAS suportadas estão documentados para uma revisita futura.
- **Reformulação cooperativa de registradores no VEC** — o `vec_dot` turbo foi reescrito para fatiar Q entre 8 lanes (na mesma convenção do f16/q8_0, 8× de folga de registrador rumo a D=256). Mantido como um refactor limpo; confirmou que o gargalo em profundidade é a *vazão* do dequant, não ocupação.
- **head_dim=64 nativo** e **K-shift turbo** — escopados, adiados.

## 8. Práticas de engenharia usadas aqui

- **Teste primeiro, em silício real** — uma referência golden de CPU + um teste de paridade numérica no device travam cada mudança; nada de "parece coerente" no olhômetro.
- **Reaproveitar em vez de reescrever** — a correção do prefill reaproveita o kernel f16 TILE já comprovado em vez de escrever à mão um loader de tile de baixa precisão frágil.
- **Falhar com segurança, não com estrondo** — caminhos não suportados (context-shift turbo, head dims fora de 128) recuam graciosamente em vez de abortar.
- **Benchmarking honesto** — o colapso em profundidade é reportado, tem a causa raiz identificada e é mostrado como comum a todo KV quantizado, não escondido.
- **Entrega reprodutível** — um pipeline de CI/CD dedicado (`.github/workflows/tqp-sycl.yml`) constrói automaticamente Windows + Linux SYCL a cada push e publica pacotes de release autocontidos.

## 9. Usando

```bash
# Cache KV turbo — requer flash-attention
llama-server -m model.gguf -ngl 99 --flash-attn on \
             --cache-type-k turbo3 --cache-type-v turbo3 -c 32768
```

Strings aceitas por `--cache-type-k` / `-v`: **`turbo2`**, **`turbo3`**, **`turbo4`**. Requer `--flash-attn on`. Em GPUs Intel, defina `SYCL_CACHE_PERSISTENT=1` uma vez para que o JIT do SYCL guarde os kernels compilados em disco (o primeiro lançamento compila todos os kernels — do contrário a partida parece lenta).

Binários pré-compilados: veja a [página de Releases](https://github.com/FellypeMelo/llama-cpp-turboquant-SYCL/releases).

## 10. Mapa do código-fonte

| Área | Arquivos |
|------|-------|
| Integração ao cache KV, modos adaptativos, guarda de shift | `src/llama-kv-cache.cpp` |
| Flash-attention SYCL (dispatch, prefill turbo) | `ggml/src/ggml-sycl/fattn.cpp` |
| Kernel de decode VEC / `vec_dot` cooperativo | `ggml/src/ggml-sycl/fattn-vec.hpp`, `fattn-common.hpp` |
| Kernel de prefill f16 TILE | `ggml/src/ggml-sycl/fattn-tile.hpp` |
| Primitivas turbo (centroides, WHT, dequant) | `ggml/src/ggml-sycl/turbo-quants.hpp`, `turbo-wht.cpp` |
| Rotação+quantização na escrita do cache | `ggml/src/ggml-sycl/set_rows.cpp` |
| Formatos de bloco / codebooks | `ggml/src/ggml-common.h`, `ggml/src/ggml.c` |
| Testes golden / stress | `tests/test-sycl-turbo.cpp` |
| CI/CD | `.github/workflows/tqp-sycl.yml` |

---

*TurboQuant é uma técnica de quantização de cache KV; este fork a implementa e valida no backend SYCL do `llama.cpp` para GPUs Intel. Construído sobre [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).*
