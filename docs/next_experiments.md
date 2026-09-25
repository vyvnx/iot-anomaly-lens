# Próximos experimentos (plano, sem antecipar resultados)

1. **Experimento inicial real:** `run-initial` com `configs/initial.yaml`, depois das decisões D1–D5
   ([protocolo_e_decisoes.md](protocolo_e_decisoes.md)).
2. **Avaliação confirmatória:** `evaluate --split test_confirmatory --release-confirmatory` no mesmo
   protocolo, uma única vez, depois de fixar a análise. Se o teste inicial motivar mudanças, a análise
   inicial passa a ser exploratória e cada mudança deve ser registrada. A confirmação continua reservada.
3. **Variantes possíveis, cada uma como protocolo novo, definidas sem consultar o confirmatório:**
   - SGD One-Class SVM linear (`models.sgd_one_class_svm.variant: linear`), como experimento adicional;
   - outra transformação dos atributos (por exemplo, logarítmica), como experimento separado;
   - efeito da inclusão ou exclusão do atributo IAT, se a decisão D3 deixar dúvida.
4. **Autoencoder em GPU:** só se houver motivo. Exige PyTorch com CUDA, relato separado e nunca misturado
   à comparação principal em CPU.
5. **Inferência estatística:** definir antes a unidade de independência. Não usar bootstrap de linhas.
