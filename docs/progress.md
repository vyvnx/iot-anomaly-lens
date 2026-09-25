# Progresso

Atualizar a cada etapa com datas reais.

## Concluído

- **24/09/2026:** leitura da especificação v1.0; inspeção do ambiente (`runs/environment.json`).
- **24/09/2026:** projeto Python com uv, `uv.lock` e `requirements.lock.txt`; PyTorch CPU.
- **24/09/2026:** implementação completa da CLI (doctor, import-data, download, inspect, prepare,
  verify-data, freeze, train, calibrate, evaluate, report, verify-run e run-initial).
- **24/09/2026:** 38 testes automatizados aprovados, com dados **sintéticos**; fluxo completo de fumaça
  e teste de escala sintético (2 milhões de linhas).
- **24/09/2026:** documentação técnica (README e docs/).

- **24/09/2026:** autor baixou a pasta oficial `MERGED_CSV` (63 arquivos `Merged01..63.csv`,
  8,66 GiB), colocada em `data/MERGED_CSV/`; importada com `--mode link` para `data/raw/`
  (`data/raw_manifest.json`, SHA-256 por arquivo, `obtained_at` = 2026-09-24 informado pelo autor).

- **24/09/2026:** `inspect` real concluído (343,6 s; RSS 1,1 GiB): 45.019.234 registros, 39 atributos +
  `Label`, 34 rótulos (BENIGN + 33 de ataque), todos presentes nos 63 arquivos; 9 linhas truncadas
  (Merged42–52) descartadas como erro de parsing; 991 `Rate` infinitos; 670 `Std`/`Variance` ausentes;
  nenhuma coluna de proveniência. Log: `runs/logs/inspect_real.log`.

- **24/09/2026:** decisões D2–D6 do autor registradas (`docs/protocolo_e_decisoes.md` §5).
- **24/09/2026:** primeira tentativa de `prepare` real esgotou a RAM do WSL; corrigida (ver
  implementation_report, problema 8).
- **24/09/2026:** `prepare` real concluído (534 s; pico RSS 5,6 GiB; log `runs/logs/prepare_real.log`):
  45.019.234 registros; 991 removidos por inválidos; 14.732.490 vetores distintos; 399.007 grupos
  conflitantes (17.085.116 registros) removidos; amostra de 165.193 vetores (100.000 benignos, 65.193
  ataques). `verify-data` aprovado: todas as categorias com ≥ 1.000 ataques em cada teste.

- **24/09/2026:** auditoria dos conflitos (`docs/auditoria_conflitos.md`) e política v2 decidida pelo autor
  (`docs/protocolo_e_decisoes.md` §6); implementada com 41 testes aprovados.
- **24/09/2026:** `prepare` v2 real (544,8 s; pico RSS 5,4 GiB; `runs/logs/prepare_real_v2.log`): amostra de
  168.000 vetores, todos os estratos no teto; `verify-data` aprovado com 18 exclusões por colisão numérica
  da mesma classe após a transformação e nenhuma ambiguidade binária após a transformação.

- **25/09/2026:** `freeze` definitivo; resultado idêntico à prévia. Defeito na verificação save/load
  corrigido e protocolo versionado (implementation_report, problema 9). Protocolo vigente: `real-20260925T023443Z-9436f126`.
- **25/09/2026:** sementes 42, 43 e 44 dos três modelos treinadas, calibradas e avaliadas **somente em
  test_initial**; `verify-run` aprovado; test_confirmatory não avaliado. Síntese em
  `runs/real-20260925T023443Z-9436f126/results_initial.md`.

## Em andamento

- Redação dos capítulos a partir dos artefatos do protocolo.

## Pendente

1. Redigir os capítulos a partir de `results_initial.md` e dos artefatos do protocolo.
2. Etapa 13–19/10: metodologia completa e análise. O confirmatório só é liberado por decisão do autor.
