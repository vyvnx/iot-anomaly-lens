# tc2-iot — detecção de anomalias no CICIoT2023 (TC II)

Ferramenta experimental de linha de comando que compara **Isolation Forest**, **SGD One-Class SVM com
aproximação RBF por Nyström** e **Autoencoder** na detecção binária (0 = benigno, 1 = anômalo) do
CICIoT2023, com modelos ajustados **somente com tráfego benigno** e limiar calibrado para FPR ≈ 1%.

> Situação em 24/09/2026: software implementado e verificado com **dados sintéticos**. Nenhum CSV do
> CICIoT2023 foi obtido ou processado ainda. Veja [docs/progress.md](docs/progress.md).

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/obtencao_dados.md](docs/obtencao_dados.md) | Como obter e importar os CSVs oficiais |
| [docs/protocolo_e_decisoes.md](docs/protocolo_e_decisoes.md) | Protocolo resolvido, quantidades e **decisões pendentes do autor** |
| [docs/arquitetura.md](docs/arquitetura.md) | Requisitos, diagrama, módulos e artefatos (Cap. 3) |
| [docs/limitacoes.md](docs/limitacoes.md) | Limitações metodológicas (Caps. 2 e 4) |
| [docs/implementation_report.md](docs/implementation_report.md) | Relatório de implementação, testes e problemas reais |
| [docs/progress.md](docs/progress.md) | Concluído, em andamento e pendente |
| [docs/next_experiments.md](docs/next_experiments.md) | Plano de continuidade |

## Instalação

Requisitos: Python 3.11 ou 3.12. Preferência: [uv](https://docs.astral.sh/uv/). O `uv.lock` fixa as
versões resolvidas; o PyTorch instalado é a variante **CPU** (a comparação principal é em CPU).

```bash
# linux / wsl / macos
uv sync                      # cria .venv e instala as versões do uv.lock
uv run tc2-iot doctor        # ambiente -> runs/environment.json
uv run pytest -q             # testes (dados sintéticos)
```

```powershell
# windows powershell (execução nativa, sem wsl)
uv sync
uv run tc2-iot doctor
uv run pytest -q
```

Alternativa sem uv (venv + pip), usando o lock exportado:

```bash
python3.11 -m venv .venv && . .venv/bin/activate          # powershell: .venv\Scripts\Activate.ps1
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
pip install -e . --no-deps
```

## Fluxo com dados reais

Coloque os caminhos entre aspas: funcionam com espaços.

```bash
uv run tc2-iot import-data --config configs/initial.yaml --source "/caminho/dos/CSV"   # ou --mode link
uv run tc2-iot inspect     --config configs/initial.yaml
#   -> revisar data/processed/inspection/*.proposed.yaml, copiar para configs/ com status: confirmed
#   -> definir split.mode e split.justification em configs/initial.yaml
uv run tc2-iot prepare     --config configs/initial.yaml
uv run tc2-iot verify-data --config configs/initial.yaml
uv run tc2-iot run-initial --config configs/initial.yaml --dry-run    # mostra etapas, sem treinar
uv run tc2-iot run-initial --config configs/initial.yaml              # freeze → train → calibrate → evaluate test_initial → report → verify-run
```

Ou etapa por etapa, após `freeze` (que imprime o `<protocol_id>`):

```bash
uv run tc2-iot freeze    --config configs/initial.yaml
uv run tc2-iot train     --protocol <protocol_id> --models all --seeds 42 43 44
uv run tc2-iot calibrate --protocol <protocol_id>
uv run tc2-iot evaluate  --protocol <protocol_id> --split test_initial
uv run tc2-iot report    --protocol <protocol_id> --split test_initial
uv run tc2-iot verify-run --protocol <protocol_id>
# somente em etapa posterior, por decisão explícita do autor (registro de exposição permanente):
uv run tc2-iot evaluate  --protocol <protocol_id> --split test_confirmatory --release-confirmatory
```

`download --urls-file urls.txt --max-gib N` existe para URLs **autorizadas** com tamanho conhecido;
não adivinha endereços nem contorna o formulário de acesso.

## Verificação com dados sintéticos

```bash
uv run python tests/fixtures/make_synthetic.py tests/output/smoke/source
uv run tc2-iot import-data --config configs/smoke.yaml --source tests/output/smoke/source
uv run tc2-iot run-initial --config configs/smoke.yaml
```

Saídas sintéticas ficam em `tests/output/` (a configuração recusa misturar pastas), com `data_kind=synthetic`
em `protocol.json` e faixa **"DADOS SINTÉTICOS"** em relatórios e figuras. Elas verificam o software,
não a capacidade de detecção no CICIoT2023.

## Códigos de saída

`0` sucesso · `1` `verify-run` com falhas · `2` bloqueado por entrada pendente (mapeamento, allowlist,
modo de partição, arquivo ausente, liberação confirmatória), com a mensagem `BLOQUEADO: ...`.
