#servidor
def cka_unecrypted(X,Y,XTX,YTY):
  # Implements linear CKA as in Kornblith et al. (2019)
  X = X.copy()
  Y = Y.copy()
  # Calculate CKA
  YTX = Y.T.dot(X)
  return (YTX ** 2).sum() * XTX * YTY

def cka_encrypted(X,Y,XTX,YTY,HE):
  X = X.copy()
  Y = Y.copy()
  # Calculate CKA
  if len(X)==len(Y)==1:
    YTX = X[0] @ Y[0]
  else:
    YTX = [~(X[i]*Y[i]) for i in range(len(X[i]))]
    for i in range(1,len(YTX)):
        YTX[0] += YTX[i]
    YTX = HE.cumul_add(YTX[0])

  bottom = XTX * YTY

  HE.relinearize(bottom)
  HE.rescale_to_next(bottom)
  square = YTX * YTX
  HE.relinearize(square)
  top = HE.cumul_add(square,False,1)
  HE.relinearize(top)
  result = bottom * top
  HE.relinearize(result)
  return result


#X: ativação
#Y: ativação transposta de outro participante

def cka(X, Y, XTX, YTY , HE = None, crypt=False): 
  if crypt:
    res = cka_encrypted(X,Y, XTX, YTY, HE)
  else:
    res = cka_unecrypted(X,Y, XTX, YTY)
  return res




#servidor
# Rodar função teste no Cliente antes de enviar o seu modelo e mandar os resultados para a função de agregação
distance_matrix = []
for i in range(len(encrypted_vectors)):
  client_distance = []
  for j in range(len(encrypted_vectors_transposed)):
    client_distance.append(cka(encrypted_vectors[i], encrypted_vectors_transposed[j], VTVS[i], VTVS[j], HE , crypt=True))
  distance_matrix.append(client_distance)
  
# Para cada cliente, mandar junto com o modelo agregado a sua linha correspondente da matriz de distâncias: distance_matrix[i]
# O cliente vai desemcriptar a sua linha de distâncias, identificar quais clientes fazem parte de seu cluster dependendo da distância
  
  

import numpy as np

class FedAvg:
      
    def __init__(self):
      pass
    
    def aggregate(self, all_trainer_samples, all_weights):
        scaling_factor = list(np.array(all_trainer_samples) / np.array(all_trainer_samples).sum())
        
        # scale weights
        for scaling, weights in zip(scaling_factor, all_weights):
            for i in range(0, len(weights)):
                weights[i] = weights[i] * scaling
        
        # agg weights
        agg_weights = []
        for layer in range(0, len(all_weights[0])):
            var = []
            for model in range(0, len(all_weights)):
                var.append(all_weights[model][layer])
            agg_weights.append(sum(var))

        return agg_weights