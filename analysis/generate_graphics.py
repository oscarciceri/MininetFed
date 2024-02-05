import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



class Graphics:
    def __init__(self,data_frames, save, experiments_folder):
        self.dfs = data_frames
        self.save = save # salva gráficos (não implementado)
        self.experiments_folder = experiments_folder
        # dfs = [{'name': name, 'df': df} for name, df in zip(names, data_frames)]
        
        # Gráfico 1: Delta T vs Round
    def deltaT_per_round(self):
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            plt.plot(df['round'], df['deltaT'], label=item['name'])
            
        plt.xlabel('round')
        plt.ylabel('Delta T')
        plt.title('Gráfico de Delta T vs round')
        plt.legend()
        plt.show()
        
        # Gráfico 2: acurácia média vs round
    def mean_acc(self):
        # plt.clf()
        plt.figure(figsize=(10, 6))
        for item in self.dfs:
            df = item['df']
            # for column in df.columns[2:]:  # Ignorando as colunas 'round' e 'round Delta T'
            plt.plot(df['round'], df['mean_accuracy'], label=item['name'])
        plt.xlabel('round')
        plt.ylabel('Acurácia')
        plt.title('Gráfico de Acurácia vs round')
        plt.legend()
        plt.show()
    
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
            ref_item =  next((d for d in self.dfs if d['name'] == relative_to), None)
            ref_df = ref_item['df']
            for item in self.dfs:
                if item['name'] == ref_item['name']:
                    continue
                df = item['df']
                plt.plot(df['round'], ref_df['n_selected'] - df['n_selected'], label=item['name'])
            
            # Adicionando uma linha pontilhada em Y=0
            plt.axhline(0, color='black', linestyle='dashed')
            
            # Definindo a escala de Y para números inteiros
            y_min = int(min(ref_df['n_selected'] - df['n_selected']))
            y_max = int(max(ref_df['n_selected'] - df['n_selected']))
            plt.yticks(np.arange(y_min, y_max+1, step=1))
            
            plt.xlabel('round')
            plt.ylabel('Diferença no Nº clientes selecionados')
            plt.title(f'Gráfico de Nº clientes selecionados relativo à "{ref_item["name"]}"')
            plt.legend()
            plt.show()
    
    def network_consumption(self):
        pass