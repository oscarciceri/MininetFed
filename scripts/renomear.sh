#!/bin/bash

# Verifica se o usuário forneceu o caminho da pasta como argumento
if [ -z "$1" ]; then
  echo "Uso: $0 <caminho_da_pasta>"
  exit 1
fi

# Atribui o caminho da pasta a uma variável
directory="$1"

# Verifica se o diretório existe
if [ ! -d "$directory" ]; then
  echo "Erro: O diretório '$directory' não existe."
  exit 1
fi

# Itera sobre todos os arquivos no diretório
for file in "$directory"/*; do
  # Verifica se é um arquivo regular
  if [ -f "$file" ]; then
    # Obtém o nome do arquivo sem o caminho completo
    filename=$(basename "$file")
    
    # Substitui os 3 primeiros caracteres por "sensor"
    new_filename="sensor${filename:12}"

    # Renomeia o arquivo
    mv "$file" "$directory/$new_filename"
  fi
done

echo "Renomeação concluída!"
