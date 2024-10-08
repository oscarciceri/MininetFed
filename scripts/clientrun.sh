# #!/bin/bash

# # Gera 3 letras aleatórias
# random_letters=$(cat /dev/urandom | tr -dc 'a-z' | fold -w 3 | head -n 1)

# # Executa o script python
# python3 client/client.py 172.17.0.2 $random_letters

# Atribui o argumento a nome, ou usa 'statipo00' como padrão se nenhum argumento for fornecido
nome=${1:-statipo00}

# Atribui o segundo argumento a valor, ou usa '0' como padrão se nenhum argumento for fornecido
valor=${2:-0}

mode=${2:-client}

# Executa o script python
# python3 client/client.py 172.20.72.17 $nome $valor $mode
python3 client/client.py 10.0.0.1 $nome $valor $mode
