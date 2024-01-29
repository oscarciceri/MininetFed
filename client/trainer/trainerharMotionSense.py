import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd

class TrainerHarMotionSense:
        
    def __init__(self,num_id) -> None:
        self.folder = "client/data"
        self.id = num_id
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_data()
        input_shape = self.x_train.shape[1:]
        self.num_samples = self.x_train.shape[0]
        n_classes = len(np.unique(self.y_train))
        self.model = self.define_model(input_shape, n_classes)
        self.stop_flag = False
    
    def get_num_samples(self):
        return self.num_samples
    
    def define_model(self, input_shape=(28, 28, 1), n_classes=10):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        opt = SGD(learning_rate=0.01)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

        
    def split_data(self):
        x_train, y_train, x_test, y_test = self.load_data()

        return x_train, y_train, x_test, y_test

    def train_model(self):
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=64, epochs=10, verbose=3)

    def eval_model(self):
        acc = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=False)[1]
        return acc
    
    def all_metrics(self):
        metrics_names = self.model.metrics_names
        values = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=False)
        return dict(zip(metrics_names, values))

    
    def get_weights(self):
        return self.model.get_weights()
    
    def update_weights(self, weights):
        self.model.set_weights(weights)
    
    def set_stop_true(self):
        self.stop_flag = True
    
    def get_stop_flag(self):
        return self.stop_flag
    
    def load_data(self):
        # Definir o caminho da pasta
        folder_path = self.folder

        # Carregar os dados de treinamento
        x_train = pd.read_csv(os.path.join(folder_path, 'MotionSense_x_train.csv'), sep=',', decimal='.')
        y_train = pd.read_csv(os.path.join(folder_path, 'MotionSense_y_train.csv'), sep=',', decimal='.')

        # Carregar os dados de teste
        x_test = pd.read_csv(os.path.join(folder_path, 'MotionSense_x_test.csv'), sep=',', decimal='.')
        y_test = pd.read_csv(os.path.join(folder_path, 'MotionSense_y_test.csv'), sep=',', decimal='.')

        # print(self.external_id,self.id)
        # Selecionar linhas com base no self.id
        mask_train = y_train['id'] == self.id
        mask_test = y_test['id'] == self.id
        
        x_train = x_train[mask_train]
        y_train = y_train[mask_train]['act']

        x_test = x_test[mask_test]
        y_test = y_test[mask_test]['act']
        print(y_test.shape,x_test.shape)
        print(y_train.shape,x_train.shape)
        # Converter os dataframes para numpy arrays antes de usá-los no treinamento do modelo
        x_train = x_train.values
        y_train = y_train.values
        x_test = x_test.values
        y_test = y_test.values

        return x_train, y_train, x_test, y_test



        

# if __name__ == '__main__':
#     trainer = TrainerCifar()
#     for l in trainer.model.layers:
#         print(l.name)
#         print(l.get_weights())