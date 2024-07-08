Problema: Ferramenta não treina algumas vezes com o trainer CKKSFED e o agregador CKKSFED

### Sintoma constatado:

Alguns clientes se mantem vários rounds com a exata mesma acurácia

### Pontos de falha possíveis na ordem que eu acho onde pode estar o problema:

- Separação dos dados
- agregação dos pesos
- rede neural
- geração da matriz de distâncias
- transmissão da matriz de distâncias

# Entendendo o problema

O que eu quer saber:

- Em qual caso isso ocorre?
  1. Com o trainer sozinho?
  - Não ocorre
  2. Com o trainer com apredizado federado sem clusterização?
  - Ocorre
  3. Com o trainer com aprendizado federado com clusterização sem encriptar?
  4. Com o trainer com aprendizado federado com clusterização com a matriz encriptada?
