import re
import pandas as pd
import csv
from datetime import datetime
import sys

class File:
    def __init__(self, name):
        self.name = name
        self.data = pd.DataFrame(columns=['round', 'deltaT', 'mean_accuracy'])
        
        with open(self.name + '.log', 'r') as file:
            self.content = file.readlines()
        self.processContent()
    
    def processContent(self):
        round_start_time = None
        round_end_time = None
        mean_accuracy = -1
        for line in self.content:
            if 'METRIC' in line:
                if 'round:' in line:
                    round_number = int(re.search('round: (\d+)', line).group(1))
                    
                    if round_start_time is not None:
                        round_end_time = datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
                        deltaT = round_end_time - round_start_time
                        self.save_data(round_number - 1, deltaT.total_seconds()*1000, mean_accuracy)
                        
                    round_start_time = datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
                    
                elif 'mean_accuracy:' in line:
                    mean_accuracy = float(
                        re.search('mean_accuracy: (\d+\.\d+)', line).group(1))
                    
                elif 'stop_condition' in line and round_start_time is not None:
                    round_end_time = datetime.strptime(line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
                    deltaT = round_end_time - round_start_time
                    self.save_data(round_number,deltaT.total_seconds()*1000,  mean_accuracy)
                    self.save_to_csv()
                        
    def save_data(self, round, deltaT, mean_accuracy):
        new_data = pd.DataFrame({'round': [round], 'deltaT': [deltaT], 'mean_accuracy': [mean_accuracy]})
        if self.data.empty:
            self.data = new_data
        else:
            self.data = pd.concat([self.data, new_data], ignore_index=True)

    def save_to_csv(self):
        self.data.to_csv(self.name + '.csv', index=False)


if __name__ == '__main__':
    # total args
    n = len(sys.argv)

    #  check args
    if (n < 2):
        print("correct use: sudo python3 trainingLog.py <1.log> ...")
        exit()

    files = []
    
    for fileName in sys.argv[1:]:
        files.append(File(fileName.split('.')[0]))
   


# # --------------------------------------------------
    import pandas as pd
    import matplotlib.pyplot as plt
    dfs = []
    for fileName in sys.argv[1:]:
        name = fileName.split('.')[0]
        dfs.append({'name':name ,'df':pd.read_csv(name + '.csv')})
    

    # # Gráfico 1: Delta T vs Round
    # plt.figure(figsize=(10, 6))
    # plt.plot(df['round'], df['deltaT'])
    # plt.xlabel('round')
    # plt.ylabel('Delta T')
    # plt.title('Gráfico de Delta T vs round')
    # plt.show()

    # Gráfico 2: Acurácia de cada cliente e acurácia média vs round
    plt.figure(figsize=(10, 6))
    for item in dfs:
        df = item['df']
        for column in df.columns[2:]:  # Ignorando as colunas 'round' e 'round Delta T'
            plt.plot(df['round'], df[column], label=item['name'])
    plt.xlabel('round')
    plt.ylabel('Acurácia')
    plt.title('Gráfico de Acurácia vs round')
    plt.legend()
    plt.show()
