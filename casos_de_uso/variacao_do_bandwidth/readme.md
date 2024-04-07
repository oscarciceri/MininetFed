# Caso de Uso 2 - Dispositivos com Largura de Banda Heterogênea

O objetivo deste teste é avaliar o impacto da largura de banda no tempo de treinamento. Foram criados quatro experimentos, todos baseados na configuração do experimento anterior, mas com **100% da capacidade de CPU para todos os clientes**. A única diferença entre os quatro experimentos é a largura de banda disponível:

1. **Experimento 1**: Largura de banda de **20 Mbps** para todos os clientes.
2. **Experimento 2**: Largura de banda de **10 Mbps** para todos os clientes.
3. **Experimento 3**: Largura de banda de **5 Mbps** para todos os clientes.
4. **Experimento 4**: Cenário heterogêneo com **1 cliente a 10 Mbps** e os demais a **20 Mbps**.

Como o **TrainerHar** é propositalmente intensivo no uso de rede, espera-se um impacto significativo no tempo de treinamento. No entanto, a curva de precisão por rodada deve ser novamente bastante semelhante.

A **Figura 1** confirma as expectativas desses experimentos. Observa-se que:

- A diferença entre o tempo de finalização do experimento com **20 Mbps** e **10 Mbps** é pouco mais de **100 segundos**.
- A diferença entre o experimento com **10 Mbps** e o de **5 Mbps** é de cerca de **300 segundos**, demonstrando o caráter não linear da redução da largura de banda.
- O experimento heterogêneo fica entre o com **20 Mbps** e o com **10 Mbps** em termos de tempo de conclusão.

![Figura 1](https://example.com/figuras/rede_heterogenea.png)
