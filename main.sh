#!/bin/bash

# Verifica se pelo menos um argumento foi passado
if [ $# -eq 0 ]; then
    echo "Uso: sh main.sh <arquivo1.py> <arquivo2.py> ..."
    exit 1
fi

# Itera sobre cada argumento (nome de arquivo Python)
for file in "$@"; do
    # Verifica se o arquivo existe
    if [ -f "$file" ]; then
        echo "Executando: sudo python3 $file -s"
        sudo python3 "$file"
    else
        echo "Erro: Arquivo $file não encontrado."
    fi
done
