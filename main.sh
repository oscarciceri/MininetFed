#!/bin/bash

# Verifica se pelo menos um argumento foi passado
if [ $# -eq 0 ]; then
    echo "Uso: sh main.sh <arquivo1.py> <arquivo2.py> ..."
    exit 1
fi

# Define o diretório raiz do projeto (o local deste script)
RAIZ=$(dirname "$(realpath "$0")")

# Itera sobre cada argumento (nome de arquivo Python)
for file in "$@"; do
    # Verifica se o arquivo existe
    if [ -f "$file" ]; then
        # Extrai o diretório do arquivo
        FILE_DIR=$(dirname "$(realpath "$file")")

        echo "Executando: sudo PYTHONPATH=$RAIZ python3 $file -s"
        
        # Executa o arquivo Python com o PYTHONPATH definido para a raiz
        sudo PYTHONPATH="$RAIZ" python3 "$file"
    else
        echo "Erro: Arquivo $file não encontrado."
    fi
done
