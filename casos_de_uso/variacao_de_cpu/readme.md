# Caso de Uso 1 - Dispositivos com Processamento Heterogêneo

Este primeiro caso de uso examina o impacto da disponibilidade de CPU no tempo de treinamento. Foram configurados três experimentos utilizando o **TrainerHar**, a função de seleção **Random** e a função de agregação **FedAvg**. Em todos os cenários, há um total de seis clientes e as condições de rede são idênticas. Aqui estão os detalhes dos experimentos:

1. **Experimento 1**: 100% da capacidade de processamento do host é disponibilizada para todos os clientes.
2. **Experimento 2**: Apenas 50% da capacidade de processamento é fornecida a todos os clientes.
3. **Experimento 3**: Três clientes têm 100% de capacidade, enquanto os outros três têm 50%, simulando um cenário heterogêneo.

A expectativa é que o gráfico de acurácia por rodada seja bastante semelhante entre todos os experimentos. No entanto, o tempo de conclusão do experimento com todos os clientes operando a 50% de capacidade deve ser um pouco menos que o dobro em relação ao primeiro experimento com 100%. O terceiro experimento deve apresentar um resultado intermediário.

As expectativas são confirmadas ao analisar a **Figura 1**. O tempo de finalização do experimento com 100% de disponibilidade de CPU é cerca de 150 segundos, enquanto o experimento com 50% de capacidade leva pouco mais de 250 segundos. O experimento com três clientes operando a 50% e três clientes operando a 100% fica na casa dos 180 segundos.

![Figura 1](https://example.com/figuras/cpu_heterogeneo.png)
