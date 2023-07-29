import flwr as fl
import torch
import lerdados as aux
import modelo as modelo
import argparse
import numpy as np
import timeit
import lerconfig as cfg



cid = 1
nb_clients = 10
train_batch_size = 32
test_batch_size = 32
epochs = 10



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

client_address = "10.0.0.1:8083"
fl.client.start_client(server_address=client_address,  client=client)
