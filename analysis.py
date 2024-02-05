from analysis.config import Config
from analysis.process_log import File
from analysis.generate_graphics import Graphics
# from .federated.experiment import Experiment
# from .client.trainer import Trainer

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
                    if csv:
                        f.save_to_csv()
                    dfs.append({'name':f"{experiment_name}/{name}" ,'df':df})
            else:
                for fileName in os.listdir(f"{FOLDER}/{experiment_name}"):
                    if fileName.endswith(".log"):
                        name = fileName.split(".")[0]
                        f = File(f"{FOLDER}/{experiment_name}/{name}")
                        df = f.get_dataframe()
                        if csv:
                            f.save_to_csv()
                        dfs.append({'name':f"{experiment_name}/{name}" ,'df':df})
        
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
    
        
        
        
        
        
        
        
        
    datasets_analysis = config.get("datasets_analysis")
    print(datasets_analysis)
    
    
    # files = []
    
    # for fileName in sys.argv[1:]:
    #     file = File(fileName.split('.')[0])
    #     file.save_to_csv()
    #     files.append(file)