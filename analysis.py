from analysis.config import Config
from analysis.process_log import File
from analysis.generate_graphics import Graphics
# from .federated.experiment import Experiment
from client import Trainer

import sys
import os

if __name__ == '__main__':
    # total args
    n = len(sys.argv)

    #  check args
    if (n < 3):
        print("correct use: sudo python3 analysis.py <experiments_folder> <graphics.yaml>")
        exit()

    FOLDER = sys.argv[1]
    config = Config(sys.argv[2])
    
    
    experiments_analysis = config.get("experiments_analysis")
    if experiments_analysis != None:
        
        csv =  experiments_analysis.get("save_csv")
        dfs = []
        
            
        
        for experiment in experiments_analysis["from"]:
            exp_files =  experiment.get("files")
            experiment_name = experiment["experiment"]
            if exp_files != None:
                for fileName in exp_files:
                    name = fileName.split(".")[0]
                    f = File(f"{FOLDER}/{experiment_name}/{name}")
                    df = f.get_dataframe()
                    netdf = f.get_net_dataframe()
                    if csv:
                        f.save_to_csv()
                    dfs.append({'name':f"{experiment_name}/{name}" ,'df':df,'netdf':netdf})
            else:
                for fileName in os.listdir(f"{FOLDER}/{experiment_name}"):
                    if fileName.endswith(".log"):
                        name = fileName.split(".")[0]
                        f = File(f"{FOLDER}/{experiment_name}/{name}")
                        df = f.get_dataframe()
                        netdf = f.get_net_dataframe()
                        if csv:
                            f.save_to_csv()
                        dfs.append({'name':f"{experiment_name}/{name}" ,'df':df, 'netdf':netdf})
        
        plot = Graphics(dfs,experiments_analysis.get("save_graphics"),FOLDER)
        
        for graphic in experiments_analysis["graphics"]:
            if graphic['type'] == 'mean_acc':
                plot.mean_acc()
            elif graphic['type'] == 'deltaT_per_round':
                plot.deltaT_per_round()
            elif graphic['type'] == 'n_clients_absolute':
                plot.n_clients_absolute()
            elif graphic['type'] == 'n_clients_relative':
                relative_to = f"{graphic['relative_to']['experiment']}/{graphic['relative_to']['file'].split('.')[0]}"
                plot.n_clients_relative(relative_to)
            elif graphic['type'] == 'network_consumption':
                plot.network_consumption()
    
        
    import matplotlib.pyplot as plt
    import pandas as pd
    # import seaborn as sns
    datasets_analysis = config.get("datasets_analysis")
    if datasets_analysis != None:
        trainer = Trainer(datasets_analysis["id"],datasets_analysis["mode"])
        
        for graphic in datasets_analysis["graphics"]:
            # Distribuição de classes
            if 'class_distribution' == graphic["type"]:
                plt.figure(figsize=(10, 6))
                plt.hist(trainer.y_train, bins=30)
                plt.title('Distribuição de Classes')
                plt.show()

            # Histograma
            if 'histogram' == graphic["type"]:
                plt.figure(figsize=(15,10))
                plt.hist(trainer.x_train, bins=30)
                plt.title('Histograma')
                plt.show()

            # Boxplot
            if 'boxplot'  == graphic["type"]:
                plt.figure(figsize=(10, 6))
                plt.boxplot(trainer.x_train)
                plt.title('Boxplot')
                plt.show()

            # Matriz de correlação
            if 'correlation_matrix' == graphic["type"]:
                df = pd.DataFrame(trainer.x_train)
                corr = df.corr()
                cax = plt.matshow(corr, cmap='coolwarm')
                plt.colorbar(cax)
                plt.title('Matriz de Correlação')
                plt.show()
        # # Distribuição de classes
        # if 'class_distribution' in datasets_analysis["graphics"]:
        #     plt.figure(figsize=(10, 6))
        #     sns.countplot(x=trainer.y_train)
        #     plt.title('Distribuição de Classes')
        #     plt.show()

        # # Histograma
        # if 'histogram' in datasets_analysis["graphics"]:
        #     trainer.x_train.hist(bins=30, figsize=(15,10))
        #     plt.title('Histograma')
        #     plt.show()

        # # Boxplot
        # if 'boxplot' in datasets_analysis["graphics"]:
        #     plt.figure(figsize=(10, 6))
        #     trainer.x_train.boxplot()
        #     plt.title('Boxplot')
        #     plt.show()

        # # Matriz de correlação
        # if 'correlation_matrix' in datasets_analysis["graphics"]:
        #     plt.figure(figsize=(10, 6))
        #     sns.heatmap(trainer.x_train.corr(), annot=True, fmt=".2f")
        #     plt.title('Matriz de Correlação')
        #     plt.show()
    
