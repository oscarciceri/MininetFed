# import re
# import csv
# from datetime import datetime

# with open('meu_arquivo.log', 'r') as file:
#     log = file.readlines()

# csv_data = []
# round_start_time = None
# trainers_accuracy = {}
# mean_accuracy = {}

# for line in log:
#     if 'METRIC' in line:
#         if 'round:' in line:
#             round_number = int(re.search('round: (\d+)', line).group(1))
#             if round_start_time is not None:
#                 delta_t = (datetime.strptime(line.split(
#                     ' - ')[0], '%Y-%m-%d %H:%M:%S,%f') - round_start_time).total_seconds() * 1000  # convert to milliseconds
#                 for round_number, accuracies in trainers_accuracy.items():
#                     row = [delta_t]
#                     for trainer_number, accuracy in accuracies.items():
#                         row.append(accuracy)
#                     row.append(mean_accuracy[round_number])
#                     csv_data.append(row)
#                 trainers_accuracy = {}
#                 mean_accuracy = {}
#             round_start_time = datetime.strptime(
#                 line.split(' - ')[0], '%Y-%m-%d %H:%M:%S,%f')
#             trainers_accuracy[round_number] = {}
#         elif 'accuracy:' in line and round_start_time is not None:
#             trainer_number = int(re.search('(\d+) loss', line).group(1))
#             accuracy = float(re.search('accuracy: (\d+\.\d+)', line).group(1))
#             trainers_accuracy[round_number][trainer_number] = accuracy
#         elif 'mean_accuracy:' in line:
#             mean_accuracy[round_number] = float(
#                 re.search('mean_accuracy: (\d+\.\d+)', line).group(1))

# header = ['Round Delta T'] + \
#     [f'Cliente {i+1}' for i in range(len(csv_data[0])-2)] + ['Acurácia Média']

# with open('training_log.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     writer.writerows(csv_data)
