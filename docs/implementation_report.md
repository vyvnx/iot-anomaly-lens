# Relatório de implementação

Data: 24/09/2026. Todas as execuções abaixo usaram **dados sintéticos**. Nenhum resultado do CICIoT2023
foi produzido.

## Ambiente utilizado (runs/environment.json)

Debian 12 no WSL2 (kernel 6.18), Python 3.11.2, AMD Ryzen 7 5800XT (o WSL2 expõe 4 núcleos físicos e
8 lógicos), 11,7 GiB de RAM, NVIDIA RTX 4060 Ti (8 GiB), não usada: o PyTorch instalado é
`2.14.0+cpu`. Versões principais: numpy 2.4.6, scikit-learn 1.9.1, torch 2.14.0+cpu, pyarrow 25.0.1,
duckdb 1.5.5, pandas 3.0.6 e matplotlib 3.11.2. O ambiente é gerenciado por uv 0.11.12; as versões
ficam em `uv.lock`, e `requirements.lock.txt` serve à alternativa pip. Essa alternativa não foi testada
nesta máquina.

## Decisões de implementação

- **Hash dos vetores no DuckDB:** o texto canônico usa a conversão double→texto do DuckDB. Antes de
  adotá-la, verificou-se que ela é de ida e volta exata em 300 mil valores aleatórios (inclusive
  subnormais e extremos) e que `-0.0` gera texto distinto, por isso é normalizado para `0.0`. Colisões
  são detectadas por grupo (texto mínimo ≠ máximo). O representante escolhido tem os valores relidos do
  registro original e o hash recalculado.
- **Staging em Parquet de strings:** permite contar valores não numéricos por coluna, com índice de
  linha determinístico por arquivo. Linhas com número errado de colunas são contadas como erro de
  parsing, com exemplos no relatório.
- **Propostas, nunca decisões:** `inspect` gera `label_mapping.proposed.yaml` e
  `feature_allowlist.proposed.yaml`, que só valem depois de copiados para `configs/` com
  `status: confirmed`.
- **Imutabilidade por entrada:** cada modelo/semente calibrado entra no `models_manifest.json` e não é
  mais alterado. Sementes novas podem ser acrescentadas depois.
- **Código congelado:** `evaluate` exige que a árvore de código-fonte tenha o mesmo hash registrado no
  `freeze`. Uma correção de código cria um protocolo novo.
- **Aquecimento no ajuste:** antes do primeiro ajuste cronometrado de cada família, um ajuste descartado
  em até 256 registros absorve a inicialização das bibliotecas (ver problema 3).

## Problemas reais encontrados e correções

1. **DuckDB — acesso a campo de struct sem nome** (`Binder Error: struct_extract with a string key
   cannot be used on an unnamed struct`), ao escolher o representante do grupo. Correção:
   `arg_min(..., struct_pack(f := arquivo, r := linha))`.
2. **f-string com chaves de struct** do DuckDB interpretadas pelo Python. Resolvido junto com o item 1.
3. **Tempo de ajuste distorcido pela inicialização:** na primeira fumaça, o AE da semente 42 levou 6,9 s
   e o da semente 43, 0,07 s. Um experimento isolado confirmou o custo de primeira execução
   (1,16 s contra 0,08 s repetindo a semente 42). Correção: aquecimento descartado e documentado na
   coluna `inclui` de `timings.csv`.
4. **Ordem das frações:** uma configuração regravada pelo PyYAML (chaves em ordem alfabética) era
   rejeitada. A validação passou a checar o conjunto de chaves e a normalizar para a ordem canônica,
   da qual dependem os desempates.
5. **Disco no agrupamento:** guardar os textos canônicos mínimo e máximo gerou 572 MB para 2 milhões de
   linhas. Guardar apenas a flag `hash_collision` reduziu para 74 MB.
6. **Mínimo por categoria invalidava a amostra:** o parâmetro entrava na impressão digital do `prepare`
   sem alterar a amostra. Foi excluído dela.
7. **Gráficos:** a faixa "DADOS SINTÉTICOS" se sobrepunha aos supertítulos e os rótulos do gráfico de
   FPR se sobrepunham. Corrigido após inspeção visual.

8. **Falta de memória no `prepare` real (24/09/2026).** Na primeira execução sobre os 45.019.234
   registros, o WSL caiu por falta de RAM depois de cerca de 32 min de agrupamento, com 4,6 GB já
   despejados em disco. A sequência foi reconstruída pelos horários dos arquivos; o log ficou vazio,
   porque o medidor só gravava ao final. Causa provável: estados de agregação sem tamanho fixo
   (`count(DISTINCT label)`, `min`/`max` do texto canônico de cerca de 400 bytes, `arg_min` sobre struct)
   em cerca de 40 milhões de grupos, que o DuckDB não consegue despejar em disco, somados a um
   `row_number()` global ordenado por string. O teste de escala com 2 milhões de linhas não reproduzia o
   problema. Correção:
   - passada única que grava num banco DuckDB em disco uma tabela estreita de inteiros (arquivo, linha,
     rótulo, flags, chave MD5 de 128 bits e verificação xxhash de 64 bits do texto canônico);
   - agregações só com estados de tamanho fixo;
   - amostragem por estrato com `arg_min(..., n)` (top-k), sem ordenação global;
   - o `vector_id` publicado continua sendo o SHA-256 do texto canônico, calculado para os registros amostrados.

   Efeito medido: com 2 milhões de linhas sintéticas, o pico de RSS caiu de 3,6 GiB para 1,0 GiB e o tempo
   subiu de 42–63 s para 115 s. Numa porção de desenvolvimento real (6 de 63 arquivos, 4.297.838
   registros; `configs/dev_subset.yaml`), `prepare` levou 49 s com pico de 786 MiB. A execução completa
   passou a ser acompanhada por um vigia que encerra o processo se a memória disponível cair abaixo de 1 GiB
   (`runs/logs/prepare_real_mem.log`), e o medidor passou a gravar o log em tempo real
   (`scripts/measure.py`).

9. **Verificação `save_load_scores_within_tolerance` com defeito (25/09/2026).** No primeiro protocolo real
   (`real-20260925T023151Z-5a424935`), o `verify-run` falhou nessa checagem. Os modelos recarregados
   reproduziam exatamente (diferença 0,0) as predições gravadas ao pontuar o teste inteiro, como no
   `evaluate`; mas a checagem recalculava só os primeiros 2.000 registros. No SGD One-Class SVM, a
   transformação Nyström em float32 varia até 2,6e-6 com o tamanho do lote (escores perto de zero), o que
   estourava a tolerância relativa. Nenhuma decisão mudou (0 de 43.996). Correção: comparar o teste inteiro
   e registrar a sensibilidade ao lote como informação (`batch_size_sensitivity_info`). Como o código mudou,
   o protocolo foi versionado: o anterior ficou marcado `SUPERSEDED.md` e preservado, e o novo
   (`real-20260925T023443Z-9436f126`) tem partições e artefatos congelados idênticos. A semente 42 foi reproduzida bit a bit
   (escores e decisões iguais nos três modelos).

## Testes (pytest, 41 aprovados)

`uv run pytest -q` executa `tests/test_units.py` e `tests/test_pipeline.py` com fixtures **sintéticas**
(`tests/fixtures/make_synthetic.py`). Os testes cobrem:

- rótulo desconhecido, atributo proibido (coluna de rótulo na allowlist), categoria ausente e categoria
  abaixo do mínimo;
- duplicatas entre arquivos, grupo conflitante, ausentes, não numéricos, infinitos e linha malformada,
  com contagens exatas;
- amostra idêntica com os mesmos registros distribuídos em 2 ou 3 arquivos de nomes diferentes;
  partições independentes da ordem das linhas;
- treino e validações só com benignos; conflito no espaço transformado preserva o treino;
- scaler e constantes inalterados quando valores exclusivos do teste mudam; overflow em float32;
- orientação dos escores e salvar/carregar nos três modelos; recusa de AE sem gargalo; restauração do
  melhor checkpoint;
- empates e amostra pequena na calibração; limiar inalterado quando rótulos/escores do teste mudam;
- métricas contra caso manual (contagens, recall, FPR, precisão, F1, acurácia balanceada, AUROC) e
  denominadores nulos;
- mesmos registros e ordem para os três modelos; métricas recompostas das predições;
- artefato congelado ou manifesto bruto alterado invalida a execução; artefato de modelo corrompido não
  é reutilizado;
- confirmatório bloqueado sem liberação e registrado quando liberado; `run-initial` nunca o avalia;
  saídas sintéticas marcadas e separadas.

Os testes verificam a correção do software. Nenhum exige recall mínimo.

## Execuções sintéticas registradas

| Execução | Configuração | Resultado |
|---|---|---|
| Fumaça | `configs/smoke.yaml`, 8.409 linhas, 3 arquivos, 2 sementes | `run-initial` completo em ≈ 15 s; `verify-run` OK; retomada reaproveitou tudo; confirmatório bloqueado (código 2) |
| Escala | 2.000.000 de linhas, 46 atributos + rótulo, 566 MB, 5 arquivos | `inspect` 23 s (RSS 1,5 GiB); `prepare` 42–63 s (RSS ≤ 3,6 GiB); `run-initial` com hiperparâmetros reais sobre 125.101 vetores: 29 s, `verify-run` OK |

Na execução de escala, o ajuste do AE levou 7,5 s (50 épocas, sem parada antecipada), o do IF 0,38 s
e o do SGD-OCSVM 0,27 s (convergiu em 7 iterações), com 41.461 benignos de treino. São números de
dados sintéticos e servem apenas para dimensionar recursos.
