# Protocolo resolvido, quantidades e decisões pendentes

> **Leitura:** as seções 1 a 4 registram a proposta inicial (24/09/2026), antes dos dados reais. Algumas
> regras foram substituídas pelas decisões das seções 5 a 7: política de conflitos v2, amostragem por
> categoria e partição sistemática. A execução vigente é o protocolo `real-20260925T023443Z-9436f126`.

Estado em 24/09/2026. Nenhum CSV real foi inspecionado; quantidades são **limites da configuração**,
não contagens obtidas. Valores numéricos são propostas da especificação, não decisões do orientador.

## 1. Protocolo resolvido (configs/initial.yaml)

| Etapa | Regra implementada |
|---|---|
| Entrada | Todos os CSVs importados (`data.files: all`), verificados por SHA-256 contra `raw_manifest.json` |
| Rótulo | Coluna detectada (única não numérica) ou definida em `data.label_column`; rótulo original preservado |
| Atributos | Allowlist confirmada pelo autor; rótulo e identificadores de linha/arquivo/captura proibidos |
| Inválidos | Registro removido se algum atributo aprovado for ausente, não numérico, NaN ou infinito; exclusões contadas por motivo (sobrepostas e combinação), rótulo e categoria |
| Duplicatas | Chave de agrupamento = MD5 (128 bits) do texto canônico dos atributos (ordem fixa, `-0.0`→`0.0`, sem rótulo), sobre **todos** os arquivos juntos; colisões verificadas por xxhash (64 bits) do mesmo texto; `vector_id` publicado = SHA-256 do texto canônico |
| Conflitos | Vetor com rótulos originais diferentes: grupo inteiro removido |
| Unidade | Um vetor distinto por grupo (representante: menor `arquivo#linha`), multiplicidade preservada |
| Amostra | Até 100.000 benignos e até 2.000 por rótulo original de ataque, sem reposição; prioridade MD5(`2026:chave`), menores valores por estrato (top-k por estrato) |
| Partição | `split.mode` **pendente** (ver D4); proposta: `random_unique_vectors`, semente 2027, arredondamento por maiores restos |
| Benignos | train 60% · validation_fit 10% · validation_calibration 10% · test_initial 10% · test_confirmatory 10% |
| Ataques | test_initial 50% · test_confirmatory 50%, estratificados por rótulo original |
| Pré-processamento | Remove constantes do treino; StandardScaler só em train; conversão float32 com checagem de overflow |
| Conflito no espaço efetivo | Vetores idênticos após o pré-processamento não cruzam partições: prioridade train > validation_fit > validation_calibration > test_initial > test_confirmatory; perdas registradas |
| Isolation Forest | 200 árvores, max_samples = min(256, n_train), max_features 1.0, sem bootstrap; escore = −score_samples |
| SGD One-Class SVM | Nyström RBF (γ = 1/d, min(256, n_train) componentes) + SGDOneClassSVM(ν=0.01, max_iter 2000, tol 1e-4, average); escore = −decision_function |
| Autoencoder | d→32→8→32→d, ReLU, saída linear, MSE, Adam 1e-3, lote 256, até 50 épocas, early stopping em validation_fit (paciência 5, min_delta 1e-5), melhor checkpoint restaurado; escore = MSE por registro |
| Calibração | limiar = quantile(escores de validation_calibration, 0.99, method="higher"); decisão = escore > limiar |
| Sementes | 42, 43, 44 (mesma amostra e partição) |
| Tempos | CPU, orçamento de threads = núcleos físicos (4 nesta máquina); aquecimento descartado; inferência com 1 aquecimento + 5 repetições, lote 4096 |
| Teste confirmatório | Nunca avaliado por `run-initial`; exige `--release-confirmatory`; exposição registrada em log só de acréscimo |

## 2. Quantidades esperadas (limites; confirmar após `prepare`)

| Partição | Benignos | Ataques |
|---|---|---|
| train | até 60.000 | 0 |
| validation_fit | até 10.000 | 0 |
| validation_calibration | até 10.000 | 0 |
| test_initial | até 10.000 | ≈ 50% de (≤ 2.000 × nº de rótulos de ataque) |
| test_confirmatory | até 10.000 | ≈ 50% de (≤ 2.000 × nº de rótulos de ataque) |

O número de rótulos de ataque só será conhecido na inspeção. Rótulos com menos de 2.000 vetores
distintos válidos entram com o que houver. Com 10.000 benignos de calibração, o quantil 0,99 deixa cerca
de 100 registros acima do limiar.

## 3. Recursos estimados

Medição **sintética** nesta máquina (2 milhões de linhas, 47 colunas, 566 MB de CSV, 5 arquivos):
`inspect` 23 s; `prepare` entre 42 e 63 s, com pico de RSS entre 3,0 e 3,6 GiB; staging de 307 MB;
treino, calibração, avaliação e relatório dos três modelos (uma semente, 125 mil vetores) em 29 s.
Numa extrapolação linear, sem garantia, cada 10 milhões de linhas reais custariam cerca de 2 min de
`inspect` e 4–5 min de `prepare`, e o staging ocuparia cerca de 55% do tamanho dos CSVs. O DuckDB fica
limitado a 60% da RAM disponível e grava em disco o que exceder. Não há necessidade prevista de GPU.

## 4. Decisões que precisam do autor

| ID | Decisão | Quando | Proposta |
|---|---|---|---|
| D1 | Quais arquivos CSV baixar/usar (plano de amostragem) | antes do download | Todos os CSVs da distribuição oficial. Se só parte for usada, a amostra não será probabilística em relação ao dataset completo (fica registrado) |
| D2 | Confirmar o mapeamento exato rótulo → benigno/categoria | após `inspect` | Revisar `label_mapping.proposed.yaml` contra o material oficial |
| D3 | Allowlist de atributos; em especial **IAT**, marcado como possível proxy do processo de captura | após `inspect` | Decidir com base no material oficial e na literatura. Registrar a justificativa no `feature_allowlist.yaml` |
| D4 | `split.mode` e `split.justification` | após `inspect` | `random_unique_vectors`, se não houver identificador confiável de captura, dispositivo ou período (`provenance_group` não foi implementado por falta desse identificador) |
| D5 | Mínimo de ataques por categoria em cada teste (`min_attack_per_category_per_test`) | antes de `freeze` | 100 |
| D6 | Executar as 3 sementes já na etapa inicial | antes de `run-initial` | Sim: o custo medido dos modelos é pequeno |
| D7 | Execução adicional do AE em GPU | opcional | Não fazer: o PyTorch instalado é CPU, e o AE treina em segundos |
| D8 | Data real do download manual (`--obtained-at`) | na importação | Informar, se conhecida; senão fica `null` |

Tudo o que não está nesta lista já está implementado conforme a especificação e não precisa de
confirmação.

## 5. Decisões registradas (24/09/2026)

| ID | Decisão do autor | Onde está gravada |
|---|---|---|
| D1 | Pasta oficial `MERGED_CSV` completa (63 arquivos), baixada em 24/09/2026 | `data/raw_manifest.json` |
| D2 | Mapeamento confirmado; VULNERABILITYSCAN → Recon conforme a classificação oficial; BENIGN fora das 7 categorias | `configs/label_mapping.yaml` (`validation`) |
| D3 | IAT excluído do experimento principal: "exclusão preventiva diante da suspeita de dependência do processo de captura". Decisão conservadora, não afirmação de vazamento comprovado; a coluna segue nos dados originais | `configs/feature_allowlist.yaml` (38 atributos) |
| D4 | `random_unique_vectors`, semente 2027, pela ausência de identificadores de captura, dispositivo e período | `configs/initial.yaml` (`split.justification`) |
| D5 | Mínimo de 100 ataques por categoria em cada teste: piso operacional, sem garantia de precisão estatística. Sem duplicar ataques; insuficiência bloqueia `verify-data` e é registrada para revisar a amostragem | `configs/initial.yaml` |
| D6 | Sementes 42, 43 e 44: primeiro a 42 para validar o funcionamento, depois 43 e 44, e todas entram nos resultados | sequência de comandos abaixo |

**Validação do mapeamento contra a página oficial** (consultada em 24/09/2026): a página diz "33 attacks
[...] classified into seven categories", mas lista 32 nomes (11 em DDoS) e não menciona "ICMP
fragmentation". Os CSVs têm 33 rótulos de ataque: 32 correspondem um a um à lista oficial, e
`DDOS-ICMP_FRAGMENTATION` foi mapeado para DDoS pelo prefixo do próprio rótulo. A divergência fica
registrada; a documentação oficial não a resolve.

**Sequência de execução (D6):**

```bash
uv run tc2-iot verify-data --config configs/initial.yaml
uv run tc2-iot freeze      --config configs/initial.yaml
uv run tc2-iot train       --protocol <id> --models all --seeds 42
uv run tc2-iot calibrate   --protocol <id>
uv run tc2-iot evaluate    --protocol <id> --split test_initial
uv run tc2-iot report      --protocol <id> && uv run tc2-iot verify-run --protocol <id>
# depois da validação da semente 42:
uv run tc2-iot train       --protocol <id> --models all --seeds 43 44
uv run tc2-iot calibrate   --protocol <id>
uv run tc2-iot evaluate    --protocol <id> --split test_initial
uv run tc2-iot report      --protocol <id> && uv run tc2-iot verify-run --protocol <id>
```

## 6. Política de conflitos v2 (decidida pelo autor em 24/09/2026, após `docs/auditoria_conflitos.md`)

Substitui a v1 (remover todo grupo conflitante e limitar a amostra por rótulo original), que não foi usada
em nenhum protocolo congelado. Fica registrada como `policy_version: 2` em `data_quality.json` e em
`protocol.json`.

| Situação | Regra | Justificativa registrada |
|---|---|---|
| A: benigno + ataque no mesmo vetor | grupo inteiro excluído; identificadores, rótulos e contagens preservados em `excluded_binary_ambiguous_rows.parquet` | ambiguidade binária na representação adotada, sem afirmar que os rótulos originais estejam errados. Essas observações podem ser as mais difíceis de distinguir, o que pode enviesar as métricas |
| B: ataques de categorias diferentes | mantidos como ataque; estrato `attack_category_ambiguous`; conjunto completo de rótulos e categorias preservado nos metadados | "ambíguo" não é uma 8ª categoria nem uma saída do modelo; recall reportado à parte, fora das 7 categorias e do macro |
| C: ataques de uma mesma categoria | mantidos no estrato da categoria, sem cota extra | categoria conhecida |

**Amostragem por estrato** (sem reposição; prioridade MD5(`2026:chave`); se faltar, usa-se o disponível e a
diferença fica em `strata_counts.csv`). Os tetos são desenho experimental, não distribuição natural da base:

| Estrato | Teto (dois testes somados) |
|---|---|
| Benigno | 100.000 (todas as partições) |
| DDoS | 24.000 |
| DoS | 8.000 |
| Recon | 10.000 |
| Web-based | 12.000 |
| Brute Force | 2.000 |
| Spoofing | 4.000 |
| Mirai | 6.000 |
| attack_category_ambiguous | 2.000 |

Como a amostragem é por categoria, e não por rótulo, a composição por rótulo original é reportada em
`sample_counts.csv` e não se presume que todos os 33 ataques estejam representados.

**Partição:** alocação sistemática dentro de cada estrato. Os vetores são ordenados por rótulo e depois pela
prioridade SHA-256(`2027:vector_id`), e cada um vai para a partição com maior déficit em relação à sua
fração. Assim, estrato e rótulos se dividem entre os testes com diferença de no máximo 1.

**Colisões após a transformação (P4):** o pré-processamento é ajustado só no treino. Se dois vetores
distintos ficarem idênticos no espaço que os modelos recebem (float32 padronizado, sem as colunas
constantes do treino):
- benigno + ataque: todos os membros são excluídos (coerente com P1);
- mesma classe binária em partições diferentes: fica só o membro da partição de maior prioridade
  (train > validation_fit > validation_calibration > test_initial > test_confirmatory);
- se o treino perder registros, o pré-processamento é reajustado e a verificação se repete até o treino
  estabilizar.

Cada remoção fica em `effective_space_exclusions.csv`, com motivo, partição e rótulo. A prévia aparece em
`verify-data`, antes do `freeze`.

## 7. Regra completa de identidade do vetor

1. Atributos: os 38 de `configs/feature_allowlist.yaml`, na ordem do cabeçalho dos CSVs (IAT excluído;
   `Label`, arquivo e linha nunca entram).
2. Cada valor textual do CSV é convertido pelo DuckDB para DOUBLE (IEEE 754, 64 bits), sem arredondamento.
   Registros com valor ausente, não numérico ou não finito em qualquer atributo são excluídos antes.
3. `-0.0` é normalizado para `0.0`; formas textuais diferentes do mesmo número (`0` e `0.0`) resultam no
   mesmo double.
4. Forma canônica: cada double é escrito no texto mais curto que o reconstrói exatamente (ida e volta
   verificada em 300 mil valores), e os valores são unidos por `|` na ordem do item 1.
5. Chave de agrupamento: MD5 (128 bits) da forma canônica, com verificação de colisão por xxhash (64 bits)
   do mesmo texto; nenhuma colisão foi encontrada. Identificador publicado: `vector_id` = SHA-256 da forma
   canônica.
6. Vetores idênticos em todos os arquivos selecionados formam um grupo. O representante é o registro de
   menor (arquivo, linha); a multiplicidade original é preservada.
