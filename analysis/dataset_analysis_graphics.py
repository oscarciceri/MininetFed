import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DatasetAnalysisGraphics:
  def __init__(self, id,trainer) -> None:
    self.trainer = trainer
    self.id =id
    
    
  def class_distribution(self, y_labels=None):
      # Contando a ocorrência de cada classe em y_train
      classes, counts = np.unique(self.trainer.y_train, return_counts=True)

      plt.figure(figsize=(10, 6))
      plt.bar(classes, counts)

      if y_labels is not None:
          plt.xticks(classes, y_labels)

      plt.xlabel('Classe', fontsize=18)
      plt.ylabel('Quantidade de Casos', fontsize=18)
      plt.title(f'Quantidade de casos por Classe nos Dados de Treino (cliente {self.id})', fontsize=16)
      plt.tick_params(labelsize=16)

      plt.show()

      # Contando a ocorrência de cada classe em y_test
      classes, counts = np.unique(self.trainer.y_test, return_counts=True)

      plt.figure(figsize=(10, 6))
      plt.bar(classes, counts, color="green")

      if y_labels is not None:
          plt.xticks(classes, y_labels)

      plt.xlabel('Classe', fontsize=18)
      plt.ylabel('Quantidade de Casos', fontsize=18)
      plt.title(f'Quantidade de Itens de uma Dada Classe nos Dados de Teste (cliente {self.id})', fontsize=16)
      plt.tick_params(labelsize=16)

      plt.show()





  # Histograma
  def histogram(self):
      plt.figure(figsize=(15,10))
      plt.hist(self.trainer.x_train, bins=30)
      plt.title('Histograma')
      plt.show()

  # Boxplot
  def boxplot(self):
      plt.figure(figsize=(10, 6))
      plt.boxplot(self.trainer.x_train)
      plt.title('Boxplot')
      plt.show()

  # Matriz de correlação
  def correlation_matrix(self):
      df = pd.DataFrame(self.trainer.x_train)
      corr = df.corr()
      cax = plt.matshow(corr, cmap='coolwarm')
      plt.colorbar(cax)
      plt.title('Matriz de Correlação')
      plt.show()