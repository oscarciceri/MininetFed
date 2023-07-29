import numpy as np
import torch
from torch.utils.data import  Dataset, SubsetRandomSampler
import pandas as pd

class LoadHar():
	def __init__(self):
		self.root = "flw/data/pml-training.csv"
		self.df = pd.read_csv(self.root, low_memory=False)
		self.parts = ["belt", "arm", "dumbbell", "forearm"]
		self.variables = ["roll_{}", "pitch_{}", "yaw_{}", "total_accel_{}", 
					   "accel_{}_x", "accel_{}_y", "accel_{}_z", "gyros_{}_x",
					   							   "gyros_{}_y", "gyros_{}_z"]
		self.var_list, self.labels, self.mean, self.std = self.normalize_data()
		self.length = self.var_list.size()[1]

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		step = self.var_list[:,idx]
		step = torch.unsqueeze(step, 0)
		target = self.labels[idx]

		return step, target, idx

	def normalize_data(self):
		var_list, labels = self.build_dataset()
		var_std = var_list.std(dim=1, keepdim=True)
		var_mean = var_list.mean(dim=1, keepdim=True)
		var_list = (var_list - var_mean) / var_std

		return var_list, labels, var_mean, var_std

	def build_dataset(self):
		var_list = []
		for part in self.parts:
			for var in self.variables:
				var_list.append(list(self.df[var.format(part)]))
		var_list = torch.tensor(var_list)
		labels = torch.tensor([ord(char) for char in list(self.df["classe"])])
		labels -= 65

		return var_list, labels

	def split_ind(self, val_split, shuffle=True):

		random_seed = 42
		indices = list(range(self.length))
		split = int(np.floor(val_split * self.length))
		if shuffle:
			np.random.seed(random_seed)
			np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_indices)
		val_sampler = SubsetRandomSampler(val_indices)

		return train_sampler, val_sampler
	
class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, transform=None):
        self.data, self.labels = LoadHar().build_dataset()
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        X = self.data[:, ID]
        y = self.labels[ID]
        if self.transform:
            X = self.transform(X)
        return X, y


def dataset_partitioner(dataset, batch_size, client_id, number_of_clients):

    np.random.seed(123)
    dataset_size = len(dataset)
    nb_samples_per_clients = dataset_size // number_of_clients
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    start_ind = client_id * nb_samples_per_clients
    end_ind = start_ind + nb_samples_per_clients
    data_sampler = SubsetRandomSampler(dataset_indices[start_ind:end_ind])
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler
    )
    return data_loader


def load_data( train_batch_size, test_batch_size, cid, nb_clients):   
    train_dataset = Dataset([e for e in range(19622)])
    test_dataset = Dataset([e for e in range(19622)])
    train_loader = dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    test_loader = dataset_partitioner(
        dataset=test_dataset,
        batch_size=test_batch_size,
        client_id=cid,
        number_of_clients=nb_clients,
    )

    return (train_loader, test_loader)