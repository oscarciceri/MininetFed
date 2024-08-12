#!/bin/bash

# Exclui todo o conteúdo da pasta "temp/ckksfed_fhe/pasta"
rm -rf temp/ckksfed_fhe/pasta/*

# Ativa o ambiente virtual
source env_key_gen/bin/activate

# Navega até o diretório "temp/ckksfed_fhe"
cd temp/ckksfed_fhe

# Executa o script Python
python3 key_gen.py
