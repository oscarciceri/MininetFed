import random

# Power-of-choice (POC): Seleciona aleatoriamente um número K de clientes
class Poc:
    def __init__(self):
      self.trainers_per_round = 12
    
    def select_trainers_for_round(self, trainer_list, metrics):
      if len(trainer_list) <= self.trainers_per_round:
          return trainer_list
      else:
          return random.sample(trainer_list, self.trainers_per_round)