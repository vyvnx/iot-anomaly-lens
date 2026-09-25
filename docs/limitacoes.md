# Limitações conhecidas

Válidas para qualquer execução deste protocolo. Os relatórios de cada protocolo repetem as aplicáveis.

1. **Unidade de análise:** vetores de atributos distintos, não a distribuição natural do tráfego. A
   multiplicidade original é preservada em `partitions.parquet`, mas não pondera as métricas.
2. **Remoção de conflitos:** grupos com rótulos originais conflitantes são removidos inteiros, o que
   altera a distribuição (impacto em `data_quality.json`).
3. **Independência:** a partição aleatória de vetores distintos não comprova independência temporal,
   por dispositivo ou entre observações relacionadas. Não há teste em dispositivos inéditos.
4. **Representatividade:** se só parte dos arquivos oficiais for usada, a amostragem interna não torna
   a amostra probabilística em relação ao CICIoT2023 completo.
5. **Calibração:** o quantil 0,99 é um critério experimental, sem garantia estatística de FPR futura.
   Empates podem deixar a FPR de calibração abaixo de 1%, e a FPR de teste pode diferir entre modelos.
   Nesse caso, eles não foram comparados à mesma FPR de teste.
6. **Precisão e F1** dependem da proporção amostrada de ataques e não se extrapolam para prevalências reais.
7. **Configuração fixa:** sem busca de hiperparâmetros. Uma configuração por família não comprova
   superioridade universal do algoritmo.
8. **Sementes:** as repetições (42, 43, 44) medem a variabilidade de inicialização e ajuste, não a incerteza
   populacional. Não há teste de significância nem intervalos de confiança automáticos.
9. **Padronização:** não elimina a influência de extremos. A distância no espaço padronizado é uma
   escolha do experimento.
10. **SGD One-Class SVM:** o scikit-learn é linear; a não linearidade vem da aproximação de Nyström (256
    landmarks do treino). O ν = 0,01 não garante FPR de 1%.
11. **Tempos:** medidas offline em lote, em CPU, com 4 threads (núcleos físicos que o WSL2 expõe nesta
    máquina), sem leitura de disco. Não equivalem a operação em tempo real. Sementes não garantem
    igualdade entre versões e plataformas.
12. **Descrições de atributos:** as de `feature_catalog.csv` são interpretação do autor pela nomenclatura
    e devem ser conferidas no material oficial.
13. **Proveniência:** `provenance_group` não foi implementado, porque não se conhece identificador
    confiável de captura nos CSVs. Se a inspeção encontrar um, será preciso implementá-lo.
