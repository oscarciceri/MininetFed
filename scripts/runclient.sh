#!/bin/bash

# Gera 3 letras aleatórias
random_letters=$(cat /dev/urandom | tr -dc 'a-z' | fold -w 3 | head -n 1)

# Executa o script python
python3 client/client.py 172.17.0.2 $random_letters