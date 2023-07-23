import flwr as fl
import torch
import lerdados as aux
import modelo as modelo
import argparse
import numpy as np
import timeit
import lerconfig as cfg



cid = cfg['id']
nb_clients = cfg['nc']
train_batch_size = cfg['trbs']
test_batch_size = cfg['tsbs']
epochs = cfg['nr']

train_loader, test_loader = aux.load_data(
    train_batch_size=train_batch_size,
    test_batch_size=test_batch_size,
    cid=cid,
    nb_clients=nb_clients + 1,
)


client = modelo.FlowerClient(
    cid=cid,
    train_loader=train_loader,
    test_loader=test_loader,
    epochs=epochs,
    device=torch.device("cpu"),
)

client_address = f"{cfg['ip']}:8083"
fl.client.start_client(server_address=client_address,  client=client)
