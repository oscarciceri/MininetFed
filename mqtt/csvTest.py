import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def arrays_to_csv(arrays, headers, filename):
    # Verifica se o número de arrays corresponde ao número de cabeçalhos
    if len(arrays) != len(headers):
        raise ValueError("O número de arrays deve ser igual ao número de cabeçalhos")

    # Cria um dicionário onde a chave é o cabeçalho e o valor é o array correspondente
    data = {headers[i]: arrays[i] for i in range(len(headers))}

    # Cria um DataFrame pandas a partir do dicionário
    df = pd.DataFrame(data)

    # Escreve o DataFrame em um arquivo CSV
    df.to_csv(filename, sep=';', index=False)

# Exemplo de uso da função
arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
headers = ['Média', 'Tempo', 'Cliente']
arrays_to_csv(arrays, headers, 'meu_arquivo.csv')


def plot_csv(filename):
    # Lê o arquivo CSV em um DataFrame pandas
    df = pd.read_csv(filename, sep=';')

    # Cria um gráfico para cada coluna
    for column in df.columns:
        plt.figure()  # Cria uma nova figura
        plt.plot(df[column])  # Plota a coluna
        plt.xlabel('Rodadas')  # Define o nome do eixo x
        plt.ylabel(column)  # Define o nome do eixo y como o nome da coluna
        plt.title(f'Gráfico de {column}')  # Define o título do gráfico
        plt.show(block=False)  # Mostra o gráfico sem bloquear
    input()

# Exemplo de uso da função
plot_csv('meu_arquivo.csv')
