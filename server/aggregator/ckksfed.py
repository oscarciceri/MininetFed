import sys
import numpy as np

from .fed_avg import FedAvg

import torch #Precisa importar isso para o Pyfhel funcional
from Pyfhel import Pyfhel, PyCtxt
import time

ENCRYPT = True

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
  # [0 0 0] [1 0 0] [0 1 0] [1 1 0] [0 0 1] 
  HE.relinearize(bottom)
  HE.rescale_to_next(bottom)  #
  square = YTX * YTX
  HE.relinearize(square)
  top = HE.cumul_add(square,False,1)
  HE.relinearize(top)
  
  # HE.rescale_to_next(top)  #
  # top, bottom = HE.align_mod_n_scale(bottom,top) #
  
  # print(bottom, file=sys.stderr)
  # print("AAAAAAAAAAAAAAAAAAAAAa", file=sys.stderr)
  # HE.relinearize(bottom)
  # print(bottom,file=sys.stderr)
  # print("AAAAAAAAAAAAAAAAAAAAAa", file=sys.stderr)
  # print(top,file=sys.stderr)
  # HE.relinearize(top)
  # print("AAAAAAAAAAAAAAAAAAAAAa", file=sys.stderr)
  # print(top,file=sys.stderr)
  # print("AAAAAAAAAAAAAAAAAAAAAa", file=sys.stderr)
  # time.sleep(1)
  result = bottom * top
  HE.relinearize(result)
  return result


#X: ativação
#Y: ativação transposta de outro participante

def decode_value(HE,value):
   
   return PyCtxt(pyfhel=HE, bytestring=value.encode('cp437'))


def decode_array(HE, encrypted_array):
    out = []
    # print(len(encrypted_array), file=sys.stderr)
    # print(encrypted_array, file=sys.stderr)
    for element in encrypted_array:
        # print(type(element), file=sys.stderr)
        b = element.encode('cp437')
        # print(type(b), file=sys.stderr)
        # print(b, file=sys.stderr)
        c_res = PyCtxt(pyfhel=HE, bytestring=b)
        out.append(c_res)
    return out
    
  

def cka(X, Y, XTX, YTY , HE = None, crypt=False): 
  if crypt:
    X = decode_array(HE,X)
    Y = decode_array(HE,Y)
    XTX = decode_value(HE,XTX)
    YTY = decode_value(HE,YTY)
    res = cka_encrypted(X,Y, XTX, YTY,HE)
  else:
    res = cka_unecrypted(X,Y, XTX, YTY)
  return res




#servidor
# Rodar função teste no Cliente antes de enviar o seu modelo e mandar os resultados para a função de agregação
# def get_distance_matrix(encrypted_vectors,encrypted_vectors_transposed, VTVS, HE):
#   distance_matrix = []
#   for i in range(len(encrypted_vectors)):
#     client_distance = []
#     for j in range(len(encrypted_vectors_transposed)):
#       client_distance.append(cka(encrypted_vectors[i], encrypted_vectors_transposed[j], VTVS[i], VTVS[j], HE , crypt=True))
#     distance_matrix.append(client_distance)
  
# Para cada cliente, mandar junto com o modelo agregado a sua linha correspondente da matriz de distâncias: distance_matrix[i]
# O cliente vai desemcriptar a sua linha de distâncias, identificar quais clientes fazem parte de seu cluster dependendo da distância
 

  


class Ckksfed:
      
    def __init__(self):
        dir_path = "temp/ckksfed_fhe/pasta"
        self.HE_f = Pyfhel() # Empty creation
        self.HE_f.load_context(dir_path + "/context")
        self.HE_f.load_public_key(dir_path + "/pub.key")
        # self.HE_f.load_secret_key(dir_path + "/sec.key")
        self.HE_f.load_relin_key(dir_path + "/relin.key")
        self.HE_f.rotateKeyGen()
        # self.HE_f.load_rotate_key(dir_path + "/rotate.key")
        
    def get_distance_matrix(self, client_training_responses):
      self.distance_matrix = {}
      for client_i in client_training_responses:
        client_distance = {}
        for client_j in client_training_responses:
          client_distance[client_j] = cka(client_training_responses[client_i]["training_args"][0],
                                    client_training_responses[client_j]["training_args"][1], 
                                    client_training_responses[client_i]["training_args"][2], 
                                    client_training_responses[client_j]["training_args"][2], 
                                    self.HE_f , crypt=ENCRYPT)
        self.distance_matrix[client_i] = client_distance  
    
    def aggregate(self,client_training_responses, trainers_list):
        
        self.get_distance_matrix(client_training_responses)
        
        # for client_i in client_training_responses:
        #   print( client_training_responses[client_i]["training_args"][3])
        
        weights_dict = {}
        if len(client_training_responses[trainers_list[0]]["training_args"][3]) == 0:  
          fed_avg = FedAvg()
          weights = fed_avg.aggregate(client_training_responses)
          weights_dict = {c: weights for c in trainers_list}
        else:
          aggregated_clusters = set()
          for client_i in trainers_list:
            cluster = client_training_responses[client_i]["training_args"][3].tolist()
              
            if tuple(cluster) in aggregated_clusters:
              continue
            
            aggregated_clusters.add(tuple(cluster))
            fed_avg = FedAvg()
            weights = fed_avg.aggregate({c: client_training_responses[c] for c in cluster})
            weights_dict = weights_dict | {c: weights for c in cluster}
        
        agg_response = {}
        for client in trainers_list:
            agg_response[client] = {"weights": weights_dict[client]}
        agg_response['all'] = {"distances": self.distance_matrix, "clients": trainers_list}
            
        # for client in client_training_responses:
        #   agg_response[client] = {"weights": weights[client], "distances": self.distance_matrix[client]}
        # print(sys.getsizeof(agg_response), file=sys.stderr)
        return agg_response