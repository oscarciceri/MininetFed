import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from Pyfhel import Pyfhel


from sklearn.cluster import AgglomerativeClustering

def get_params(model):
  param_dict = {}
  for name, param in model.named_parameters():
    if param.requires_grad:
        param_dict[name] = param.clone() #copy.deepcopy(param.clone())
  return param_dict

def set_params_fedsketch(model, data):
  for name, param in model.named_parameters():
        param.data = data[name]




class LeNet5(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6*num_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6*num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6*num_channels, 16*num_channels, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16*num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400*num_channels, 120*num_channels)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120*num_channels, 84*num_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84*num_channels, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out




class TrainerCkksfed():
    # ID = 0
    def __init__(self,ext_id, mode, id_name) -> None:
        
        CASE_SELECTOR = 1          # 1 or 2

        case_params = {
            1: {'l': 256},         # small l
            2: {'l': 65536},       # large l
        }[CASE_SELECTOR]
        self.l = case_params['l']
                
        
        self.id_name = id_name
        self.cluster_distance_threshold = 0.8
        self.cluster = []
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dir_path = "temp/ckksfed_fhe/pasta"
        self.num_samples = 500
        self.num_tests = 250
        self.epochs = 1
        self.cost = nn.CrossEntropyLoss()
        self.learning_rate = 0.01
        self.metric_names = ["accuracy"]
        
        self.external_id = ext_id
        self.mode = mode # client
        
        self.id = int(ext_id) + 1
        self.nc = self.id
        self.dataloader_train, self.dataloader_test = self.split_data()
        self.model = self.define_model()
        self.model_keys = list(get_params(self.model).keys())
        
        
        self.stop_flag = False
        self.args = None
        
        self.HE_f = Pyfhel() # Empty creation
        self.HE_f.load_context(dir_path + "/context")
        self.HE_f.load_public_key(dir_path + "/pub.key")
        self.HE_f.load_secret_key(dir_path + "/sec.key")
        self.HE_f.load_relin_key(dir_path + "/relin.key")
        # self.HE_f.rotateKeyGen()
        # self.HE_f.load_rotate_key(dir_path + "/rotate.key")
        
    
    def encrypt_array(self, array):
        return [self.HE_f.encrypt(array[j:j+self.HE_f.get_nSlots()]) for j in range(0,self.l,self.HE_f.get_nSlots())]
    
    def encrypt_value(self, value):
       return self.HE_f.encrypt(value)

    def decrypt_value(self, value):
       return self.HE_f.decrypt(value)

    def decrypt_array(self, encrypted_array):
        out = np.array()
        for element in encrypted_array:
            decrypted_part = self.HE_f.decryptFrac(element)
            out = np.concatenate((out, decrypted_part))
        
        return out
    
    def set_args(self,args):
        self.args = args
    
    def set_nc(self,clients):
        self.nc= clients
    
    def get_num_samples(self):
        return self.num_samples
    
    def define_model(self, n_channels=1, n_classes=10):
        return LeNet5(n_classes,n_channels )


    def sample_random_dataloader(self,dataset,num_samples, batch_size):
        indices = torch.randperm(len(dataset))[:num_samples]

        sample = torch.utils.data.Subset(dataset, indices)
        #random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=num_samples)
        dataloader = torch.utils.data.DataLoader(sample, batch_size=batch_size,shuffle=True,num_workers=2)
        return dataloader
    
    # def sample_random_dataloader(self, dataset, num_samples, batch_size, unbalanced=False):
    #     # Se o modo for 'unbalanced', altere a distribuição das classes
    #     if unbalanced:
    #         targets = np.array(dataset.targets)
    #         classes, class_counts = np.unique(targets, return_counts=True)
    #         num_classes = 10
    #         # Gere uma distribuição desbalanceada sem exceder o número de amostras por classe
    #         unbalanced_counts = [np.random.randint(1, count+1) for count in class_counts]
    #         indices = []
    #         for i in range(num_classes):
    #             class_indices = np.where(targets == classes[i])[0]
    #             class_subset = np.random.choice(class_indices, unbalanced_counts[i], replace=False)
    #             indices.extend(class_subset)
    #         np.random.shuffle(indices)
    #     else:
    #         # Se não for 'unbalanced', apenas selecione aleatoriamente
    #         indices = np.random.choice(len(dataset), num_samples, replace=False)

    #     # Crie um Subset com os índices selecionados
    #     subset = torch.utils.data.Subset(dataset, indices)
    #     # Crie o DataLoader com o Subset
    #     return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

        
    def split_data(self):
        #cliente
        train_dataset = MNIST(root = './data',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.Resize((32,32)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = (0.1307,), std = (0.3081,))]),
                                                download = True)
        test_dataset = MNIST(root = './data',
                                                train = False,
                                                transform = transforms.Compose([
                                                        transforms.Resize((32,32)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),
                                                download=True)

        dataloader_train = self.sample_random_dataloader(train_dataset, self.num_samples, 32)
        dataloader_test = self.sample_random_dataloader(test_dataset, self.num_tests, 32)
        
        

        return dataloader_train, dataloader_test

    def train_model(self):
        train_loader = self.dataloader_train
        num_epochs = self.epochs
        model = self.model
        cost = self.cost
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        total_step = len(train_loader)
        # print(total_step)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                #Forward pass
                outputs = model(images)

                loss = cost(outputs, labels)
                #actv_last.append(outputs.detach().clone().flatten())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % (total_step/num_epochs) == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # return model, loss.item()

    def eval_model(self):
        model = self.model
        test_loader = self.dataloader_test
        actv_last = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                actv_last.append(outputs.detach().clone().flatten())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
            print(torch.cat(actv_last, axis=0))
            concat_actv = np.array(torch.cat(actv_last, axis=0))
            concat_actv -= np.mean(concat_actv)
            actv = [concat_actv, concat_actv.T,1/np.sqrt((concat_actv.T.dot(concat_actv)**2).sum())]
            # return 100 * correct / total, actv
            return correct / total
    
    def get_training_args(self):
        model = self.model
        test_loader = self.dataloader_test
        actv_last = []
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                actv_last.append(outputs.detach().clone().flatten())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            concat_actv = np.array(torch.cat(actv_last, axis=0))
            concat_actv -= np.mean(concat_actv)
            
            # #Encriptando os vetores
            # XTX = self.encrypt_array(1/np.sqrt((concat_actv.T.dot(concat_actv)**2).sum()))
            # concat_actv_T = self.encrypt_array(concat_actv.T)
            # concat_actv = self.encrypt_array(concat_actv)
            
            #Valores não encriptados
            XTX = 1/np.sqrt((concat_actv.T.dot(concat_actv)**2).sum())
            concat_actv_T = concat_actv.T                
            
            actv = [concat_actv, concat_actv_T,XTX,self.cluster]

            return actv
        
    
    def all_metrics(self):
        acc = self.eval_model()
        # metrics = dict(zip(self.metric_names, [acc]))
        # if len(self.cluster) != 0:
        #     metrics["cluster"] = self.cluster
        return dict(zip(self.metric_names, [acc]))

    
    def get_weights(self):
        return list(get_params(self.model).values())
    
    def update_weights(self, weights):
        w = [torch.from_numpy(x) for x in weights]
        set_params_fedsketch(self.model, dict(zip(self.model_keys,w)))
    
    
    def agg_response_extra_info(self, agg_response):   
        data_matrix = []

        name_dict = {}
        pos_dict = {}
        for idx ,i in enumerate(agg_response["distances"]):
            name_dict[idx] = i
            pos_dict[i] = idx
            line = []
            for j in agg_response["distances"][i]:
                
                # Valor encriptado
                # line.append(self.decrypt_value(agg_response["distances"][i][j]))
                
                # Valor não encriptado
                line.append(agg_response["distances"][i][j])
            data_matrix.append(line)

        data_matrix = np.array(data_matrix) - 1
        data_matrix = abs(data_matrix)
        
        model = AgglomerativeClustering(
            metric='precomputed', n_clusters=2, linkage='complete').fit(data_matrix)

        self.cluster.clear()
        my_cluster_num = model.labels_[pos_dict[self.id_name]]
        for idx, cluster_num in enumerate(model.labels_):
            if cluster_num == my_cluster_num:
                self.cluster.append(name_dict[idx])
            
                
    
    def set_stop_true(self):
        self.stop_flag = True
    
    def get_stop_flag(self):
        return self.stop_flag
    

        

if __name__ == '__main__':
  pass



