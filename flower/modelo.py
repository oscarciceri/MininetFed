
import timeit
from collections import OrderedDict
from torch import Tensor, optim

from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn

import flwr as fl


class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(40, 64, 1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out


def train(model, train_loader, epochs, cid, device: torch.device = torch.device("cpu")):
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    print(
        f"Training {epochs} epoch(s) w/ {len(train_loader)} mini-batches each")
    for epoch in range(epochs):
        print()
        loss_epoch: float = 0.0
        num_examples_train: int = 0
        correct: int = 0
        criterion = torch.nn.CrossEntropyLoss()
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            num_examples_train += len(data)
            optimizer.zero_grad()
            output = model(data.unsqueeze(1).permute(0, 2, 1))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            loss_epoch += loss.item()
            if batch_idx % 10 == 8:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Acc: {:.6f} (Cliente {})\t\t\t\t".format(
                        epoch,
                        num_examples_train,
                        len(train_loader) * train_loader.batch_size,
                        100.0
                        * num_examples_train
                        / len(train_loader)
                        / train_loader.batch_size,
                        loss.item(),
                        correct / num_examples_train,
                        cid,
                    ),
                    end="\r",
                    flush=True,
                )
    return num_examples_train


def test(model, test_loader, device: torch.device = torch.device("cpu")):
    model.eval()
    test_loss: float = 0
    correct: int = 0
    num_test_samples: int = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_test_samples += len(data)
            output = model(data.unsqueeze(1).permute(0, 2, 1))
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= num_test_samples

    return (num_test_samples, test_loss, correct / num_test_samples)


class FlowerClient(fl.client.Client):
    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = Model(40, 5).to(device)
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

    def get_weights(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights) -> None:
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, ins: fl.common.GetParametersIns) -> fl.common.GetParametersRes:
        weights = self.get_weights()
        parameters = fl.common.ndarrays_to_parameters(weights)
        status = fl.common.Status(code=fl.common.Code.OK, message="success")
        return fl.common.GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        np.random.seed(123)
        weights = fl.common.parameters_to_ndarrays(
            ins.parameters)
        fit_begin = timeit.default_timer()
        self.set_weights(weights)
        num_examples_train: int = train(
            self.model, self.train_loader, epochs=self.epochs, device=self.device, cid=self.cid
        )

        weights_prime = self.get_weights()
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin

        status = fl.common.Status(code=fl.common.Code.OK, message="success")
        return fl.common.FitRes(
            status=status,
            parameters=params_prime,
            num_examples=num_examples_train,
            metrics={'duration':fit_duration}
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:

        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        self.set_weights(weights)

        (num_examples_test, test_loss, accuracy) = test(
            self.model, self.test_loader, device=self.device)
        print(f"Client {self.cid} - Evaluate on {num_examples_test} samples: Average loss: {test_loss:.4f}, Accuracy: {100*accuracy:.2f}%\n")

        status = fl.common.Status(code=fl.common.Code.OK, message="success")
        return fl.common.EvaluateRes(
            status=status,
            loss=float(test_loss),
            num_examples=num_examples_test,
            metrics={"accuracy": float(accuracy)}
        )
