# Auditoria dos grupos com rótulos conflitantes (24/09/2026)

Script: `scripts/audit_conflicts.py --config configs/initial.yaml`. Não altera a amostra, não congela
protocolo, não usa modelos nem métricas. Tempo: 245,5 s; pico de RSS: 6,1 GiB (`runs/logs/audit_real.log`).
Artefatos em `data/processed/audit/`: `auditoria_conflitos.json`, `conflitos_k38_*.csv` (sem IAT, protocolo
atual), `conflitos_k39_*.csv` (com IAT), `efeito_iat_por_rotulo.csv`, `previa_freeze.json`,
`previa_freeze_colisoes.csv`, `particoes_previstas.csv`.

A auditoria reproduz os números do `prepare`: 45.018.243 registros válidos, 14.732.490 vetores distintos,
399.007 grupos conflitantes e nenhuma colisão de hash.

## Identidade do vetor

- **Colunas:** os 38 atributos de `configs/feature_allowlist.yaml`, na ordem do cabeçalho. IAT fica fora
  da chave e dos atributos, mas é preservado nos CSVs e no staging.
- **Tipo:** texto do CSV convertido para DOUBLE (IEEE 754, 64 bits) pelo DuckDB, sem arredondamento.
  `-0.0` é normalizado para `0.0`.
- **Forma canônica:** o texto mais curto que reconstrói exatamente o mesmo double, com os valores unidos
  por `|`.
- **Chave:** MD5 (128 bits) da forma canônica. As colisões são verificadas por xxhash (64 bits). O
  `vector_id` publicado é o SHA-256 da forma canônica.
- **Fora da chave:** rótulo, arquivo e linha.

## Situações

| Situação | Definição |
|---|---|
| A | o grupo contém BENIGN e ao menos um ataque (ambiguidade binária) |
| B | só ataques, de duas ou mais categorias (binário sem ambiguidade; categoria ambígua) |
| C | só ataques, todos da mesma categoria (binário e categoria sem ambiguidade; só o rótulo original é ambíguo) |

## Protocolo atual (38 atributos, sem IAT)

| Situação | Grupos | Registros |
|---|---|---|
| A | 4.205 | 44.242 |
| B | 346.557 | 16.770.753 |
| C | 48.245 | 270.121 |
| **Total** | **399.007** | **17.085.116** |

Por categoria (grupos distintos que contêm a categoria / registros da categoria nesses grupos):

| Categoria | A grupos | A registros | B grupos | B registros | C grupos | C registros |
|---|---|---|---|---|---|---|
| Benigno | 4.205 | 9.104 | — | — | — | — |
| DDoS | — | — | 345.922 | 12.453.295 | 45.976 | 259.482 |
| DoS | — | — | 345.922 | 4.314.681 | 43 | 105 |
| Recon | 3.216 | 6.468 | 616 | 995 | 1.982 | 9.736 |
| Spoofing | 1.467 | 28.451 | 616 | 1.733 | 236 | 781 |
| Web-based | 62 | 196 | 40 | 46 | 8 | 17 |
| Brute Force | 16 | 23 | 3 | 3 | — | — |
| Mirai | — | — | — | — | — | — |

Um grupo com várias categorias aparece em cada linha correspondente, por isso as colunas de grupos não
somam o total.

Principais combinações (`conflitos_k38_por_combinacao.csv` tem as 166):

- B: `DDOS-SYNONYMOUSIP_FLOOD + DDOS-SYN_FLOOD + DOS-SYN_FLOOD` (80.009 grupos, 6.352.626 registros),
  `DDOS-UDP_FLOOD + DOS-UDP_FLOOD` (108.019 grupos, 5.184.954), `DDOS-TCP_FLOOD + DOS-TCP_FLOOD`
  (108.401 grupos, 5.091.069). Juntos com as demais variantes DDoS+DoS, somam 345.922 dos 346.557
  grupos B.
- C: `DDOS-SYNONYMOUSIP_FLOOD + DDOS-SYN_FLOOD` (45.940 grupos, 259.362 registros);
  `RECON-OSSCAN + RECON-PORTSCAN` (1.698 grupos, 8.148).
- A: o maior é `BENIGN + VULNERABILITYSCAN` (2.144 grupos, 6.116 registros). Os demais misturam BENIGN com
  Spoofing, Recon, Web-based e Brute Force em até 9 rótulos por grupo. Nenhum envolve DDoS, DoS ou Mirai.

## Efeito da exclusão de IAT

| | Com IAT (39) | Sem IAT (38) |
|---|---|---|
| Vetores distintos | 20.382.520 | 14.732.490 |
| Grupos conflitantes | 529.808 | 399.007 |
| Registros em conflito | 13.776.104 | 17.085.116 |
| A: grupos / registros | 855 / 25.155 | 4.205 / 44.242 |
| B: grupos / registros | 449.889 / 13.249.666 | 346.557 / 16.770.753 |
| C: grupos / registros | 79.064 / 501.283 | 48.245 / 270.121 |

Por registro: 13.776.104 já estavam em conflito com IAT e continuam sem ele. **3.309.012 passaram a
conflitar só por causa da exclusão** (19,4% dos conflitos atuais). Nenhum registro deixou de conflitar,
porque vetores idênticos com IAT continuam idênticos sem ele.

| Categoria | Conflito com IAT | Conflito sem IAT | Criado pela exclusão |
|---|---|---|---|
| Benigno | 2.692 | 9.104 | 6.412 |
| DDoS | 10.082.698 | 12.712.777 | 2.630.079 |
| DoS | 3.659.818 | 4.314.786 | 654.968 |
| Recon | 7.757 | 17.199 | 9.442 |
| Spoofing | 22.933 | 30.965 | 8.032 |
| Web-based | 187 | 259 | 72 |
| Brute Force | 19 | 26 | 7 |
| Mirai | 0 | 0 | 0 |

Interpretação limitada aos dados: IAT torna os vetores mais distintos (+5,65 milhões de vetores). Isso é
compatível com a suspeita da decisão D3, mas não a comprova. A maior parte dos conflitos DDoS × DoS
existe mesmo com IAT.

## Partições previstas (amostra atual, antes do `freeze`)

| Partição | Benigno | DDoS | DoS | Recon | Web-based | Brute Force | Spoofing | Mirai | Ataques | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 60.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60.000 |
| validation_fit | 10.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10.000 |
| validation_calibration | 10.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10.000 |
| test_initial | 10.000 | 12.000 | 4.000 | 5.000 | 5.597 | 1.000 | 2.000 | 3.000 | 32.597 | 42.597 |
| test_confirmatory | 10.000 | 12.000 | 4.000 | 5.000 | 5.596 | 1.000 | 2.000 | 3.000 | 32.596 | 42.596 |
| **Total** | **100.000** | 24.000 | 8.000 | 10.000 | 11.193 | 2.000 | 4.000 | 6.000 | **65.193** | **165.193** |

## Prévia das colisões no `freeze`

Transformações ajustadas só no treino: remoção de colunas constantes (`Telnet` e `SMTP` são constantes nos
60.000 benignos de treino, restando 36 atributos) e StandardScaler, com conversão para float32.

- **Remoção de constantes:** não cria nenhum vetor idêntico entre partições.
- **Arredondamento para float32:** 13 grupos (26 registros) de vetores distintos em float64 ficam idênticos
  entre partições. Onze são pares benigno-benigno; um é `RECON-PINGSWEEP` (test_confirmatory) × `BENIGN`
  (train); outro é `DDOS-TCP_FLOOD` (test_confirmatory) × `DOS-TCP_FLOOD` (test_initial).
- **Mesmo conjunto:** um par `DDOS-UDP_FLOOD` × `DOS-UDP_FLOOD`, ambos em test_confirmatory, fica idêntico
  em float32 sem cruzar partições.

Lista completa em `previa_freeze_colisoes.csv`. Com a regra atual (preservar treino; depois
validation_fit > validation_calibration > test_initial > test_confirmatory), o `freeze` removeria 13
registros da partição de menor prioridade de cada grupo e registraria cada perda em `protocol.json`. Essa
aplicação depende da decisão do autor.
