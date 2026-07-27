# Roadmap — fork TurboQuant SYCL

Este é um resumo prospectivo do que foi deliberadamente adiado, escopado mas não implementado, ou
ainda está pendente neste fork, destilado das notas de engenharia em `docs/pt-BR/architecture.md`,
`docs/pt-BR/decisions.md` (ADR-0005) e `docs/pt-BR/benchmarks.md`. Nada aqui é uma promessa ou um
cronograma — é uma lista honesta de trabalho de engenharia em aberto, sem ordem de prioridade a
menos que indicado.

## Gates de corretude/qualidade ainda não fechados

- **Gate de qualidade PPL/velocidade** (`scripts/turbo-quality-gate.sh`) — existe e está integrado à
  suíte de testes, mas ainda não foi executado ponta a ponta. Precisa de um modelo de referência
  local, puramente de atenção (o alvo validado é o Qwen3-4B), mais um dataset `wikitext-2-raw`;
  nenhum dos dois esteve presente na máquina de validação nas sessões documentadas até agora. O gate
  golden de similaridade de cosseno (ver `docs/pt-BR/testing.md`) passa e prova paridade numérica de
  kernel, mas isso não é a mesma afirmação que uma variação de perplexidade medida — até esse gate
  rodar, a história de qualidade se apoia no gate golden mais as amostras qualitativas de coerência
  e2e em `docs/pt-BR/benchmarks.md`.
- **Resgate de perplexidade em uma segunda razão de GQA** — o trabalho de K/V assimétrico
  (ADR-0005) foi validado no Qwen3-4B (GQA 4:1); uma execução de confirmação em um modelo com razão
  de GQA diferente (o alvo pretendido era um modelo da classe Qwen2.5, 7:1) está pendente. A lógica
  de upgrade automático-assimétrico do K tem um limiar de GQA≥6 que só um modelo nessa faixa
  realmente exercita.
- **Re-medir as configurações K/V assimétricas sob o build de perf AOT** — as seis combinações
  assimétricas `{q8_0,f16} x {turbo2,turbo3,turbo4}` hoje só têm benchmark no binário build-sync
  (JIT); o número de desempenho validado como manchete (+102,8% de prefill) vem de um binário
  separado, build-perf (F16+AOT), que ainda não teve as configurações assimétricas re-executadas
  contra ele.

## Escopado mas não implementado

- **Kernel de prefill XMX (`joint_matrix`)** — uma prova de conceito de Estágio 0 confirmou que o
  motor de matriz da Intel funciona para isso na B580 (um tile DPAS fp16→f32, erro 1e-10, JIT,
  sub-group 16). Está parado em vez de ser levado adiante: o prefill já está em paridade com fp16
  via o caminho de dequantização para f16 (ver `docs/pt-BR/architecture.md`, §4), então o ganho
  adicional projetado é marginal (~1,6% em pp512), e o retorno só cresceria de forma plausível em
  prefills muito mais longos (pp8192+). Revisitar se algum workload futuro tornar a vazão de
  prefill muito longo o gargalo.
- **Suporte nativo a head_dim=64** — escopado, não implementado. O turbo hoje só é validado em
  head_dim=128.
- **K-shift turbo** — o context-shift sobre um cache K quantizado em turbo não está implementado;
  hoje ele se desabilita graciosamente (`get_can_shift()` retorna `false` para tipos turbo) em vez
  de travar ou ser suportado. Implementá-lo exigiria um kernel de shift de K quantizado sem cast
  intermediário para f32, o que não existe hoje.
- **Reformulação cooperativa de registradores no VEC** — o `vec_dot` turbo foi reescrito
  experimentalmente para fatiar Q entre 8 lanes (na mesma convenção do f16/q8_0, com folga rumo a
  D=256). Mantido como um refactor limpo, não incorporado como ganho de desempenho: confirmou que a
  lentidão do decode em profundidade é um gargalo de *vazão* de dequant, não um problema de
  ocupação, então essa reformulação sozinha não fecharia o gap.

## Cobertura de backend

- **Suporte turbo em CUDA / Metal / Vulkan** — presente no código desde fases anteriores da
  história deste fork (ver `ggml/src/ggml-metal/turbo-matrices.h` e os caminhos de dispatch
  CUDA/Vulkan em `docs/pt-BR/upstream-sync.md`), mas o SYCL em GPU Intel Arc é o único backend
  coberto pelos gates em `docs/pt-BR/testing.md`. Os demais backends são best-effort e não são
  revalidados a cada sincronização com o upstream.
- **Regressão conhecida no Vulkan** — depois do merge com o upstream de 2026-07, o flash-attention
  turbo3 nos caminhos scalar e coopmat1 do Vulkan ficou não-funcional (o upstream moveu o código de
  dequant do flash-attention para um novo arquivo de shader sem caso `TURBO3_0`). O caminho coopmat2
  foi preservado. Esse é um backend best-effort, não-Arc, e não foi priorizado para correção.

## Follow-ups de configuração de build

- **Reabilitar `GGML_SYCL_FA_ALL_QUANTS`** (ver ADR-0004 em `docs/pt-BR/decisions.md` /
  `docs/en/decisions.md`) exige gerar a matriz completa de instâncias de flash-attention-vec turbo
  (turbo × todo tipo de KV, em toda dimensão de head suportada) antes que a macro possa voltar a ser
  ligada sem falha de link. Fora de escopo até haver necessidade concreta das combinações adicionais
  de tipos padrão que isso desbloquearia (Q4_1/Q5_0/Q5_1 pareados com turbo).

## Manutenção contínua

- **Re-sincronização com o upstream** — este fork acompanha o `ggml-org/llama.cpp` por meio de
  merges grandes periódicos, em vez de um snapshot único (ver `docs/pt-BR/upstream-sync.md`). O
  checklist de pontos de contato no final desse documento existe especificamente para que a
  superfície de conflito da próxima sincronização continue pequena; mantê-lo atualizado à medida
  que os hooks turbo evoluem é trabalho contínuo, não uma tarefa de uma vez só.
