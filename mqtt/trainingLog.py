import re
import csv
from datetime import datetime
# import sys

# # total args
# n = len(sys.argv)

# # check args
# if (n != 2):
#     print("correct use: sudo python3 trainingLog.py <metric>")
#     exit()

# TARGETMETRIC = sys.argv[1]

with open('meu_arquivo.log', 'r') as file:
    log = file.readlines()

csv_data = []
round_start_time = None
trainers_accuracy = {}
mean_accuracy = {}

def printOnCSV(line, round):
  global trainers_accuracy
  global mean_accuracy
  global round_start_time
  global csv_data
  
  delta_t = (datetime.strptime(line.split(
                    ' - ')[0], '%Y-%m-%d %H:%M:%S,%f') - round_start_time).total_seconds() * 1000  # convert to milliseconds
  for round_number, accuracies in trainers_accuracy.items():
      row = [round, delta_t]
      for trainer_number, accuracy in accuracies.items():
          row.append(accuracy)
      row.append(mean_accuracy[round_number])
      csv_data.append(row)
  trainers_accuracy = {}
  mean_accuracy = {}

  

for line in log:
    if 'METRIC' in line:
        if 'round:' in line:
            round_number = int(re.search('round: (\d+)', line).group(1))
            if round_start_time is not None:
                printOnCSV(line, round_number - 1)
            round_start_time = datetime.strptime(
                line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
            trainers_accuracy[round_number] = {}
        elif 'mean_accuracy:' in line:
            mean_accuracy[round_number] = float(
                re.search('mean_accuracy: (\d+\.\d+)', line).group(1))
        elif 'accuracy:' in line and round_start_time is not None:
            trainer_number = int(re.search('(\d+) loss', line).group(1))
            accuracy = float(re.search('accuracy: (\d+\.\d+)', line).group(1))
            trainers_accuracy[round_number][trainer_number] = accuracy
        elif 'stop_condition' in line and round_start_time is not None:
          printOnCSV(line,  round_number)
          
          

header = ['Round'] + ['Round Delta T'] + \
    [f'Cliente {i+1}' for i in range(len(csv_data[0])-3)] + ['Acurácia Média']

with open('training_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(csv_data)
# --------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Supondo que seus dados estejam em um arquivo chamado 'dados.csv'
df = pd.read_csv('training_log.csv')

# Gráfico 1: Delta T vs Round
plt.figure(figsize=(10, 6))
plt.plot(df['Round'], df['Round Delta T'])
plt.xlabel('Round')
plt.ylabel('Delta T')
plt.title('Gráfico de Delta T vs Round')
plt.show()

# Gráfico 2: Acurácia de cada cliente e acurácia média vs Round
plt.figure(figsize=(10, 6))
for column in df.columns[2:]:  # Ignorando as colunas 'Round' e 'Round Delta T'
    plt.plot(df['Round'], df[column], label=column)
plt.xlabel('Round')
plt.ylabel('Acurácia')
plt.title('Gráfico de Acurácia vs Round')
plt.legend()
plt.show()
