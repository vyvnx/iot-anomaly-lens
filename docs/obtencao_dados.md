# Obtenção dos CSVs do CICIoT2023

## Fonte oficial (e somente ela)

- Página do dataset: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Entrada de download indicada pela página: https://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/

Na consulta registrada na especificação (24/09/2026), a entrada exigia **formulário de acesso**, e o
endereço final dos arquivos não foi verificado. O cadastro é pessoal e deve ser feito **pelo autor**.
O software não preenche formulários, não adivinha URLs e não usa cópias do Kaggle, amostras de terceiros
nem versões pré-normalizadas. Qualquer mudança de origem exige decisão explícita e documentação.

## O que baixar

1. A distribuição **CSV de características extraídas** (não os PCAPs).
2. Metadados, README, exemplos e material suplementar oficiais que descrevam colunas e rótulos. Eles são
   necessários para confirmar o `label_mapping.yaml` e as descrições do `feature_catalog.csv`.

Anote a **data do download** para informar em `--obtained-at`. Se não souber, omita: o manifesto grava
`null` e nunca usa a data de importação no lugar dela.

Antes de baixar, confira o tamanho total informado pelo servidor. Esta máquina tinha 816 GiB livres em
24/09/2026 (`runs/environment.json`). O processamento cria um staging Parquet e arquivos intermediários
em `data/processed/`. Pela extrapolação sintética em [implementation_report.md](implementation_report.md),
esse espaço fica da ordem do tamanho dos CSVs.

## Onde colocar

Qualquer pasta serve. No WSL, prefira o sistema de arquivos Linux: ler de `/mnt/c/...` é bem mais lento.

```bash
# exemplo: download feito no windows, em "Downloads/CICIoT2023/CSV"
uv run tc2-iot import-data --config configs/initial.yaml \
    --source "/mnt/c/Users/<usuario>/Downloads/CICIoT2023/CSV" --mode copy --obtained-at 2026-09-25
```

- `--mode copy` copia para `data/raw/` (recomendado no WSL). `--mode link` cria links simbólicos para
  os originais, sem duplicar espaço.
- Os originais nunca são modificados. Se um arquivo já existir em `data/raw/` com conteúdo diferente,
  a importação é recusada.
- Arquivos `.csv` que sejam páginas HTML ou formulários salvos (download incompleto) são rejeitados.
- `data/raw_manifest.json` registra, por arquivo: nome relativo, tamanho, SHA-256, origem, modo,
  `obtained_at` e `registered_at`.

## Download automatizado (opcional)

Só use se o autor tiver URLs **autorizadas** e diretas, após o cadastro, sem tokens pessoais que não
possam ser registrados:

```bash
uv run tc2-iot download --config configs/initial.yaml --urls-file urls.txt --max-gib 20 --dry-run
```

O comando consulta os tamanhos, recusa tamanho desconhecido e limites excedidos, grava em `.part`,
retoma só com suporte correto a `Range`, detecta HTML e extrai `.zip` com proteção contra caminhos
fora da pasta. Query strings e credenciais são removidas da URL registrada.

## Depois da importação

```bash
uv run tc2-iot inspect --config configs/initial.yaml
```

Revise em `data/processed/inspection/`:

- `schema.json`, `raw_counts.csv`, `quality_report.json`;
- `label_mapping.proposed.yaml`: proposta automática por regras de nome. Confira cada rótulo contra o
  material oficial, corrija, copie para `configs/label_mapping.yaml` e mude `status: confirmed`.
  Rótulos sem mapeamento bloqueiam o experimento;
- `feature_allowlist.proposed.yaml` e `feature_catalog.csv`: resolva `pending_decisions` (ver
  [protocolo_e_decisoes.md](protocolo_e_decisoes.md)), copie para `configs/feature_allowlist.yaml` e
  mude `status: confirmed`.
