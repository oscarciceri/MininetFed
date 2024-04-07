# Caso de Uso 2 - Algoritmo de Aprendizagem com Compressão de Modelos

Neste experimento, exploramos o uso do algoritmo de aprendizagem **FedSketch** com seleção aleatória de clientes e a técnica de agregação **FedAvg**. O FedSketch é uma abordagem que comprime as atualizações dos modelos em estruturas de dados probabilísticas chamadas "sketches" antes de serem enviadas. Essa compressão reduz a quantidade de dados transmitidos, ao mesmo tempo que melhora a privacidade do processo.

## Detalhes do Experimento:

- **Algoritmo de Aprendizagem**: FedSketch
- **Seleção de Clientes**: Aleatória
- **Agregação**: FedAvg
- **Dataset**: MotionSense
- **Número de Clientes**: 24
- **Rounds de Treinamento**: 50

## Resultados:

Na **Figura 1**, podemos observar os resultados de acurácia alcançados:

- **FedSketch**: 85%
- **FedAvg**: 83%

![Figura 1](https://example.com/figuras/fedsketch.png)

Além disso, na **Figura 2**, apresentamos a quantidade de bytes transmitida durante o processo de treinamento para os algoritmos FedAvg e FedSketch. Vale ressaltar que, diferentemente dos casos de uso anteriores, a rede neural utilizada aqui é mais complexa e possui um maior número de parâmetros. A compressão dos pesos em sketches resultou em uma significativa redução no uso de banda durante o processo.

![Figura 2](https://example.com/figuras/fedsketchbytes.png)

Concluímos que a ferramenta foi capaz de executar o processo de treinamento mesmo com um algoritmo de aprendizagem customizado, demonstrando sua flexibilidade e eficácia.
