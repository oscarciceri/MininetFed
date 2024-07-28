import sys
import numpy as np

from .fed_avg import FedAvg
from .fed_sketch import FedSketchAgg

import torch  # Precisa importar isso para o Pyfhel funcional
from Pyfhel import Pyfhel, PyCtxt
import time
from .sketch_utils import compress, decompress, get_params, set_params, set_params_fedsketch, differential_garantee_pytorch, delta_weights, get_random_hashfunc

# ENCRYPT = True # Setado automaticamente pelo cliente

# HE.rescale_to_next(YTX)
# HE.mod_switch_to_next(YTX)
# YTX.set_scale(2 ** 30)


def cka_unecrypted(X, Y, XTX, YTY):
  # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()
    # Calculate CKA

    YTX = Y.T.dot(X)
    # print("XTX REAL", file=sys.stderr)
    # print(XTX, file=sys.stderr)
    # print("YTY REAL", file=sys.stderr)
    # print(YTY, file=sys.stderr)
    # print("YTX REAL", file=sys.stderr)
    # print(YTX, file=sys.stderr)
    # print("BOTTOM REAL", file=sys.stderr)
    # print(XTX * YTY, file=sys.stderr)
    # print("SQUARE REAL", file=sys.stderr)
    # print(YTX**2, file=sys.stderr)
    # print("TOP REAL", file=sys.stderr)
    # print((YTX ** 2).sum(), file=sys.stderr)
    return (YTX ** 2).sum() * XTX * YTY


def cka_encrypted(X, Y, XTX, YTY, HE):
    X = X.copy()
    Y = Y.copy()
    # print("X", file=sys.stderr)
    # print(HE.decrypt(X), file=sys.stderr)
    # print("Y", file=sys.stderr)
    # print(HE.decrypt(Y), file=sys.stderr)
    # Calculate CKA
    if len(X) == len(Y) == 1:
        YTX = X[0] @ Y[0]
    else:
        YTX = [~(X[i]*Y[i]) for i in range(len(X[i]))]
        for i in range(1, len(YTX)):
            YTX[0] += YTX[i]
        YTX = HE.cumul_add(YTX[0])
    HE.rescale_to_next(YTX)
    # print("YTX", file=sys.stderr)
    # print(HE.decrypt(YTX), file=sys.stderr)
    # print("XTX", file=sys.stderr)
    # print(HE.decrypt(XTX), file=sys.stderr)
    # print("YTY", file=sys.stderr)
    # print(HE.decrypt(YTY), file=sys.stderr)
    bottom = XTX * YTY
    HE.relinearize(bottom)
    # HE.rescale_to_next(bottom)
    # print("BOTTOM", file=sys.stderr)
    # print(HE.decrypt(bottom), file=sys.stderr)
    square = YTX * YTX
    HE.relinearize(square)
    # print("SQUARE", file=sys.stderr)
    # print(HE.decrypt(square), file=sys.stderr)
    top = HE.cumul_add(square, False, 1)
    HE.relinearize(top)
    # print("TOP", file=sys.stderr)
    # print(HE.decrypt(top), file=sys.stderr)
    result = top * bottom
    HE.relinearize(result)
    # print("RESULT", file=sys.stderr)
    # print(HE.decrypt(result), file=sys.stderr)
    return result


def decode_value(HE, value):

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


def encrypt_array(HE_f, array):
    CASE_SELECTOR = 1          # 1 or 2

    case_params = {
        1: {'l': 256},         # small l
        2: {'l': 65536},       # large l
    }[CASE_SELECTOR]
    l = case_params['l']

    return [HE_f.encrypt(array[j:j+HE_f.get_nSlots()]) for j in range(0, l, HE_f.get_nSlots())]


def encrypt_value(HE_f, value):
    return HE_f.encrypt(value)


def cka(X, Y, XTX, YTY, HE=None, crypt=False):
    if crypt:
        # res = cka_unecrypted(np.array(X), np.array(Y), XTX, YTY)
        # print(f"valor:{res}", file=sys.stderr)
        # XTX = encrypt_value(HE, XTX)
        # YTY = encrypt_value(HE, YTY)
        # X = encrypt_array(HE, X)
        # Y = encrypt_array(HE, Y)

        X = decode_array(HE, X)
        Y = decode_array(HE, Y)
        XTX = decode_value(HE, XTX)
        YTY = decode_value(HE, YTY)
        res = cka_encrypted(X, Y, XTX, YTY, HE)
        # res = HE.decrypt(res)
        # print(f"cript:{res}", file=sys.stderr)

        # print(f"Encriptado:{res}", file=sys.stderr)
    else:
        res = cka_unecrypted(np.array(X), np.array(Y), XTX, YTY)
    return res


# servidor
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
        self.fedsketch = True
        dir_path = "temp/ckksfed_fhe/pasta"
        self.HE_f = Pyfhel()  # Empty creation
        self.HE_f.load_context(dir_path + "/context")
        self.HE_f.load_public_key(dir_path + "/pub.key")
        # REMOVER DEPOIS DE TESTAR ---------------------------------------------------
        # self.HE_f.load_secret_key(dir_path + "/sec.key")
        self.HE_f.load_relin_key(dir_path + "/relin.key")
        # self.HE_f.rotateKeyGen()
        self.HE_f.load_rotate_key(dir_path + "/rotate.key")
        # self.HE_f.relinKeyGen()

    def get_distance_matrix(self, client_training_responses, ENCRYPT):
        self.distance_matrix = {}
        dist_uncript = {}
        for client_i in client_training_responses:
            client_distance = {}
            client_uncript = {}
            for client_j in client_training_responses:
                client_distance[client_j] = cka(client_training_responses[client_i]["training_args"][0],
                                                client_training_responses[client_j]["training_args"][1],
                                                client_training_responses[client_i]["training_args"][2],
                                                client_training_responses[client_j]["training_args"][2],
                                                self.HE_f, crypt=ENCRYPT)
                # client_uncript[client_j] = self.HE_f.decrypt(
                #    client_distance[client_j])[0]

            self.distance_matrix[client_i] = client_distance
            # dist_uncript[client_i] = client_uncript
        # print("matriz decriptada:", dist_uncript, file=sys.stderr)

    # def get_distance_matrix(self, client_training_responses):  # versão com múltiplos valores por linha (problema: mismatch)
    #   n_cli = len(client_training_responses)
    #   mask = np.zeros(n_cli*2)
    #   mask[n_cli] = 1
    #   mask_keep = np.zeros(n_cli*2)
    #   mask_keep[:(n_cli - 1)] = 1

    #   mask = self.HE_f.encodeFrac(mask)
    #   mask = self.HE_f.encryptPtxt(mask)
    #   mask_keep = self.HE_f.encodeFrac(mask_keep)
    #   mask_keep = self.HE_f.encryptPtxt(mask_keep)

    #   self.distance_matrix = {}
    #   self.distance_matrix["index"] = {}
    #   pos = 0
    #   for client_i in client_training_responses:
    #     i = n_cli

    #     # relação entre o nome do cliente e a posição na linha de distâncias que ele vai ficar
    #     self.distance_matrix["index"][client_i] = pos
    #     pos += 1

    #     for client_j in client_training_responses:

    #       client_distance = cka(client_training_responses[client_i]["training_args"][0],
    #                                 client_training_responses[client_j]["training_args"][1],
    #                                 client_training_responses[client_i]["training_args"][2],
    #                                 client_training_responses[client_j]["training_args"][2],
    #                                 self.HE_f , crypt=ENCRYPT)

    #       if client_i in self.distance_matrix:
    #         print("DIST antes",self.distance_matrix[client_i], file=sys.stderr)
    #         print("MASK antes",mask_keep << i, file=sys.stderr)

    #         (mask_keep,self.distance_matrix[client_i]) = self.HE_f.align_mod_n_scale(mask_keep,self.distance_matrix[client_i])
    #         # (self.distance_matrix[client_i], mask_keep) = self.HE_f.align_mod_n_scale(self.distance_matrix[client_i],mask_keep)
    #         print("DIST",self.distance_matrix[client_i], file=sys.stderr)
    #         print("MASK",mask_keep << i, file=sys.stderr)

    #         self.distance_matrix[client_i] = (mask_keep << i) *  self.distance_matrix[client_i]
    #         print("2",self.distance_matrix[client_i],file=sys.stderr)
    #         self.distance_matrix[client_i] = ~(self.distance_matrix[client_i])
    #         next_val = (client_distance >> (i - n_cli)) * (mask << i)
    #         next_val = ~(next_val)
    #         self.distance_matrix[client_i] += next_val
    #         print("3",self.distance_matrix[client_i],file=sys.stderr)
    #       else:
    #         self.distance_matrix[client_i] = client_distance
    #       i-= 1

    def aggregate(self, client_training_responses, trainers_list):

        # modificar para self.matriz = self.get_distance_matrix(client_training_responses)
        print(client_training_responses[trainers_list[0]]["training_args"][4])
        self.get_distance_matrix(
            client_training_responses, client_training_responses[trainers_list[0]]["training_args"][4])
        # print(self.distance_matrix, file=sys.stderr)

        # for client_i in client_training_responses:
        #   print( client_training_responses[client_i]["training_args"][3])
        if self.fedsketch == False:
            fed_avg = FedAvg()
        else:
            fed_avg = FedSketchAgg()
        weights_dict = {}
        if len(client_training_responses[trainers_list[0]]["training_args"][3]) == 0:

            weights = fed_avg.aggregate(client_training_responses)
            # print("Pesos Agregados",file=sys.stderr)
            # print(weights,file=sys.stderr)
            weights_dict = {c: weights for c in trainers_list}
        else:
            aggregated_clusters = set()
            for client_i in trainers_list:
                cluster = client_training_responses[client_i]["training_args"][3]

                if tuple(cluster) in aggregated_clusters:
                    continue

                aggregated_clusters.add(tuple(cluster))
                weights = fed_avg.aggregate(
                    {c: client_training_responses[c] for c in cluster})
                # print("Pesos Agregados",file=sys.stderr)
                # print(weights,file=sys.stderr)
                weights_dict = weights_dict | {c: weights for c in cluster}

        agg_response = {}
        for client in trainers_list:
            agg_response[client] = {"weights": weights_dict[client]}
        agg_response['all'] = {
            "distances": self.distance_matrix, "clients": trainers_list}
        agg_response['encrypted'] = client_training_responses[trainers_list[0]
                                                              ]["training_args"][4]
        # for client in client_training_responses:
        #   agg_response[client] = {"weights": weights[client], "distances": self.distance_matrix[client]}
        # print(sys.getsizeof(agg_response), file=sys.stderr)
        return agg_response
