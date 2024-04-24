from analysis.config import Config
from analysis.process_log import File
from analysis.generate_graphics import Graphics
from analysis.dataset_analysis_graphics import DatasetAnalysisGraphics
# from .federated.experiment import Experiment
from client import Trainer


from collections import OrderedDict
import sys
import os


def analysis(analysis_yaml_path):
    
    # FOLDER = sys.argv[1]
    config = Config(analysis_yaml_path)
    
    
    
    experiments_analysis = config.get("experiments_analysis")
    FOLDER = config.get("experiments_folder")
    if experiments_analysis != None:
        
        csv =  experiments_analysis.get("save_csv")
        dfs = []
        
            
        
        for experiment in experiments_analysis["from"]:
            exp_files =  experiment.get("files")
            experiment_name = experiment["experiment"]
            alias = experiment.get("alias")
            if exp_files != None:
                for idx, fileName in exp_files:
                    name = None
                    filepath = f"{experiment_name}/{fileName.split('.')[0]}"
                    if idx == 0:
                        name = alias
                    elif alias is not None:
                        name = f"{alias} ({idx})"
                    else:
                        name = filepath
                        
                    f = File(f"{FOLDER}/{filepath}")
                    df = f.get_dataframe()
                    netdf = f.get_net_dataframe()
                    if csv:
                        f.save_to_csv()
                    dfs.append({'name':name ,'df':df,'netdf':netdf})
            else:
                idx = 0
                for fileName in os.listdir(f"{FOLDER}/{experiment_name}"):
                    if fileName.endswith(".log"):
                        name = None
                        filepath = f"{experiment_name}/{fileName.split('.')[0]}"
                        if idx == 0:
                            name = alias
                        elif alias is not None:
                            name = f"{alias} ({idx})"
                        else:
                            name = filepath
                            
                        f = File(f"{FOLDER}/{filepath}")
                        df = f.get_dataframe()
                        netdf = f.get_net_dataframe()
                        if csv:
                            f.save_to_csv()
                        dfs.append({'name':name ,'df':df,'netdf':netdf})
                        idx += 1
                        
        
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
        trainers = OrderedDict()
        for id in datasets_analysis["id"]:
            trainers[id] = Trainer(id,datasets_analysis["mode"])
        plot = DatasetAnalysisGraphics(trainers)
            
        for graphic in datasets_analysis["graphics"]:
            # Distribuição de classes
            if 'class_distribution_per_client' == graphic["type"]:
                plot.class_distribution(graphic.get("y_labels"))
                
            if 'class_distribution_complete' == graphic["type"]:
                plot.class_distribution_all(graphic.get("y_labels"))

            # Histograma
            if 'histogram' == graphic["type"]:
                plot.histogram()

            # Boxplot
            if 'boxplot'  == graphic["type"]:
                plot.boxplot()

            # Matriz de correlação
            if 'correlation_matrix' == graphic["type"]:
                plot.correlation_matrix()
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
    
if __name__ == '__main__':
    # total args
    n = len(sys.argv)

    #  check args
    if (n < 2):
        # print("correct use: sudo python3 analysis.py <experiments_folder> <graphics.yaml>")
        print("alternative: correct use: sudo python3 analysis.py <graphics.yaml> ...")
        exit()

    for analysis_yaml_path in sys.argv[1:]:
        analysis(analysis_yaml_path)
    