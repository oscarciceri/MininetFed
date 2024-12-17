import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import zip_longest
import math


class Graphics:
    def __init__(self, data_frames, save, experiments_folder):
        self.dfs = data_frames
        self.save = save  # salva gráficos (não implementado)
        self.experiments_folder = experiments_folder
        # dfs = [{'name': name, 'df': df} for name, df in zip(names, data_frames)]

    #     # Gráfico 1: Delta T vs Round
    # def deltaT_per_round(self):
    #     plt.figure(figsize=(10, 6))
    #     for item in self.dfs:
    #         df = item['df']
    #         plt.plot(df['round'], df['deltaT'], label=item['name'])

    #     plt.xlabel('round')
    #     plt.ylabel('Delta T (milisegundos)')
    #     plt.title('Gráfico de Delta T vs round')
    #     plt.legend()
    #     plt.show()

    #     # Gráfico 2: acurácia média vs round
    # def mean_acc(self):
    #     # plt.clf()
    #     plt.figure(figsize=(10, 6))
    #     for item in self.dfs:
    #         df = item['df']
    #         # for column in df.columns[2:]:  # Ignorando as colunas 'round' e 'round Delta T'
    #         plt.plot(df['round'], df['mean_accuracy'], label=item['name'])
    #     plt.xlabel('round')
    #     plt.ylabel('Acurácia')
    #     plt.title('Gráfico de Acurácia vs round')
    #     plt.legend()
    #     plt.show()

    def n_clients_absolute(self):
        # plt.clf()
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            # for column in df.columns[2:]:  # Ignorando as colunas 'round' e 'round Delta T'
            plt.plot(df['round'], df['n_selected'], label=item['name'])
        plt.xlabel('round')
        plt.ylabel('N clientes selecionados')
        plt.title('Número de clientes selecionados por round')
        plt.legend()
        plt.show()

    def n_clients_relative(self, relative_to):
        if len(self.dfs) > 1:
            # plt.clf()
            plt.figure(figsize=(10, 6))
            ref_item = next(
                (d for d in self.dfs if d['name'] == relative_to), None)
            ref_df = ref_item['df']
            for item in self.dfs:
                if item['name'] == ref_item['name']:
                    continue
                df = item['df']
                plt.plot(df['round'], ref_df['n_selected'] -
                         df['n_selected'], label=item['name'])

            # Adicionando uma linha pontilhada em Y=0
            plt.axhline(0, color='black', linestyle='dashed')

            # Definindo a escala de Y para números inteiros
            y_min = int(min(ref_df['n_selected'] - df['n_selected']))
            y_max = int(max(ref_df['n_selected'] - df['n_selected']))
            plt.yticks(np.arange(y_min, y_max+1, step=1))

            plt.xlabel('round')
            plt.ylabel('Diferença no Nº clientes selecionados')
            plt.title(
                f'Gráfico de Nº clientes selecionados relativo à "{ref_item["name"]}"')
            plt.legend()
            plt.show()

    # def network_consumption(self):
    #     plt.figure(figsize=(10, 6))
    #     for item in self.dfs:
    #         df = item['netdf']
    #         plt.scatter(df['segs'], df['recived'], label=item['name'], marker='o', s=20)

    #     plt.xlabel('Tempo de execução (segundos)')
    #     plt.ylabel('bytes recived by the broker')
    #     plt.title('Gráfico de bytes recebidos no decorrer do tempo')
    #     plt.legend()
    #     plt.show()

    def network_consumption(self):
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['netdf']
            plt.scatter(df['segs']/60, df['recived']/2**30,
                        label=item['name'], marker='o', s=20)

        plt.xlabel('Execution time (minutes)', fontsize=18)
        plt.ylabel('Training network traffic (GBytes)', fontsize=18)
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.show()

    # Gráfico 1: Delta T vs Round
    def deltaT_per_round(self):
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            plt.plot(df['round'], df['deltaT'], label=item['name'])

        plt.xlabel('round', fontsize=18)
        plt.ylabel('Delta T (milisegundos)', fontsize=18)
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.show()

    # Gráfico 2: acurácia média vs round
    def mean_acc(self):
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            plt.plot(df['round'], df['mean_accuracy'], label=item['name'])
            # print(item)
        plt.xlabel('Round', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.show()

    def energy_consumption(self):
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            for label in df:
                if 'energy_consumption' in label:
                    plt.plot(
                        df['round'], df[label], label=f"{item['name']} {label.replace('_energy_consumption','')}")
            # print(item)
        plt.xlabel('Round', fontsize=18)
        plt.ylabel('Energy consumed (escala ?????)', fontsize=18)
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.show()

    def mean_acc_k_folds(self):
        plt.figure(figsize=(10, 6))
        possible_colors = ['blue', 'green', 'red', 'purple', 'orange']
        experiments = {}

        for item in self.dfs:
            df = item['df']
            if (item['experiment'] in experiments) is False:
                experiments[item['experiment']] = []

            experiments[item['experiment']].append(df['mean_accuracy'])

        for idx, experiment in reversed(list(enumerate(experiments.__reversed__()))):
            experiments_matrix = experiments[experiment]

            max_tamanho = max(len(line) for line in experiments_matrix)

            acc = np.zeros(max_tamanho)
            acc_qtd = np.zeros(max_tamanho)
            mean = np.zeros(max_tamanho)
            error = np.zeros(max_tamanho)
            error_qtd = np.zeros(max_tamanho)
            std_d = np.zeros(max_tamanho)

            for i in range(max_tamanho):
                for line in experiments_matrix:
                    if len(line) > i:
                        acc[i] += line[i]
                        acc_qtd[i] += 1

            # print(acc)
            # print(acc_qtd)
            for i in range(max_tamanho):
                mean[i] = acc[i] / acc_qtd[i]

            for i in range(max_tamanho):
                for line in experiments_matrix:
                    if len(line) > i:
                        error[i] += (line[i] - mean[i])**2
                        error_qtd[i] += 1

            for i in range(max_tamanho):
                if (error_qtd[i] - 1.0) > 0:
                    std_d[i] = math.sqrt(error[i] / (error_qtd[i] - 1.0))

            # plt.figure(figsize=(10, 6))
            # plt.fill_between(range(max_tamanho), mean -
            #                  std_d, mean+std_d, alpha=.3, color=possible_colors[idx % len(possible_colors)])
            plt.fill_between(range(max_tamanho), mean -
                             std_d, mean+std_d, alpha=.3)
            # plt.plot(range(max_tamanho), mean,
            #          label=f"{experiment}", color=possible_colors[idx % len(possible_colors)])
            plt.plot(range(max_tamanho), mean,
                     label=f"{experiment}")

            plt.xlabel('Round', fontsize=18)
            plt.ylabel('Accuracy', fontsize=18)
            plt.title('Mean accuracy in K-folds with standard deviation')
            plt.legend(fontsize=16)
            plt.tick_params(labelsize=16)
        plt.savefig(f"experiments_out/{'out'}.pdf")
        # plt.show()
