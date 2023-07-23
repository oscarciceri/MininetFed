from tabnanny import verbose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import Smart_home
import load_data
import sys
import tensorflow as tf
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Flatten,TimeDistributed, Input,RepeatVector,GlobalAveragePooling1D,BatchNormalization,AlphaDropout,Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from matplotlib import pyplot as plt
from typing import Dict
import numpy as np
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import random
import os.path
from pathlib import Path
import csv
import json,codecs
from functools import reduce
import psycopg2
from paho.mqtt import client as mqtt_client
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import uuid
import sqlite3
import datetime
import gc

broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"

class FederatedClient(object):
    def __init__(self, chosen_building_id,building_class, x_train, y_train, x_test, y_test, model, target_scaler, features_scalers):
        self.chosen_building_id = chosen_building_id
        self.building_class = building_class
        self.client_id = str(uuid.uuid4())
        #self.total_round = total_round
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_size = len(x_train.index)
        self.model = model
        self.target_scaler = target_scaler
        self.features_scalers = features_scalers
        self.round = 1
        #self.client_paho_id = f'python-mqtt-{random.randint(0, 1000)}'

    def publish_introduction(self, client):
        client.subscribe(self.client_id)
        msg = {
            'resp_topic' : self.client_id,
            'building_class' : self.building_class
            }
        #print(len(json.dumps(msg).encode('utf-8')))
        print('Publicando introduction')
        result = client.publish(topic_introduction, json.dumps(msg))
        response = subscribe.simple(self.client_id, hostname=broker)
        self.total_round = int(json.loads(response.payload)['total_round'])
        self.connection_db = sqlite3.connect(str(json.loads(response.payload)['connection_address']), timeout=120)
        
        #self.cursor_db = self.connection_db.cursor()
        #self.cursor_db.execute("""
        #                    SELECT * FROM aggregate_models WHERE ROUND=? ORDER BY Timestamp DESC;
        #                    """, [0])
        #linha = self.cursor_db.fetchmany(1)
        #self.cursor_db.close()
        #model_data = json.loads(linha[0][1])

        #self.model.set_weights([np.asarray(i) for i in model_data['Weights']])
        print(self.total_round )

    def publish_choice(self, client):
        print("aqui publish choice")
        random_number = random.uniform(0,1)
        client.subscribe(self.client_id)
        msg = {
            'resp_topic' : self.client_id,
            'random_number' : random_number
            }
        print("publishing choice")
        result = client.publish(topic_choice, json.dumps(msg))
        print(result)
        #if result[0] != 0:
        #    result = client.publish(topic_choice, json.dumps(msg))
        #    print(result)
        #    if result[0] != 0:
        #            client.reconnect()
        response = subscribe.simple(self.client_id, hostname=broker)
        print(response.payload)
        self.chosen = int(json.loads(response.payload)['chosen'])
        self.round = int(json.loads(response.payload)['round'])
        print(self.chosen)
        print(self.round)

    def publish_agg(self,client):
        
        client.subscribe(self.client_id)
        msg = {
                'resp_topic' : self.client_id,
            }
        if self.chosen == 1:
            es = EarlyStopping(monitor='val_loss', patience=5)
            model_name = 'Modelos/best_model'+ self.client_id  + '.h5'
            checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            history = self.model.fit(self.x_train, self.y_train, epochs=20, validation_split=0.10, batch_size=32,verbose=0,callbacks=[es,checkpoint])
            self.model = load_model(model_name)
            pred = self.model.predict(self.x_test)
            pred_rescaled = self.target_scaler.inverse_transform(pred)
            y_test_rescaled =  self.target_scaler.inverse_transform(self.y_test)
            self.mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
            self.rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
            self.score = r2_score(y_test_rescaled, pred_rescaled)
            print(f'MAE:{self.mae}')
            print(f'RMSE:{self.rmse}')
            print(f'R2:{self.score}')
            weights = [i.tolist() for i in self.model.get_weights()]
            agg_data = {
                'Size' : self.train_size,
                'Weights' : weights,
                'Cluster' : self.building_class
            }
            #print(len(json.dumps(msg).encode('utf-8')))
            
            ct = datetime.datetime.now()
            self.cursor_db = self.connection_db.cursor()
            self.cursor_db.execute("""
                            INSERT INTO models (MODEL, ROUND, Timestamp)
                            VALUES (?, ?, ?)
                            """, (json.dumps(agg_data ), self.round, ct))
            self.connection_db.commit()
            self.cursor_db.close()
            result = client.publish(topic_agg, json.dumps(msg))
            print(result)
            #if result[0] != 0:
            #    result = client.publish(topic_choice, json.dumps(msg))
            #    print(result)
            #    if result[0] != 0:
            #        #client.reinitialize(client_id = self.client_id, clean_session = True, userdata = None)
            #        time.sleep(1)
            #        result = client.publish(topic_choice, json.dumps(msg))
            print("Pesos publicados")
            response = subscribe.simple(self.client_id, hostname=broker)

        else:
            result = client.publish(topic_agg, json.dumps(msg))
            print(result)
            #if result[0] != 0:
            #    result = client.publish(topic_choice, json.dumps(msg))
            #    print(result)
            #    if result[0] != 0:
            #        #client.reinitialize(client_id = self.client_id, clean_session = True, userdata = None)
            #        time.sleep(1)
            #        result = client.publish(topic_choice, json.dumps(msg))
            print("Pesos publicados")
            print(result)
            response = subscribe.simple(self.client_id, hostname=broker)
        
        print("Pesos recebidos")
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                            SELECT * FROM aggregate_models WHERE ROUND=? AND CLUSTER=? ORDER BY Timestamp DESC;
                            """, [self.round,self.building_class])

        linha = self.cursor_db.fetchmany(1)
        print(self.building_class)
        if linha != []:
            model_data = json.loads(linha[0][1])
            print("client w")
            for i in model_data['Weights']:
                
                print(np.asarray(i).shape)
            model.set_weights([np.asarray(i) for i in model_data['Weights']])
        self.cursor_db.close()
        gc.collect()
        



    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    print("Connected to MQTT Broker!")
                else:
                    print("Failed to connect, return code %d\n", rc)
        # Set Connecting Client ID
        client = mqtt_client.Client(self.client_id)
        #client = mqtt_client.Client(client_paho_id,transport='websockets')
        #client.username_pw_set(username, password)
        client.on_connect = on_connect
        client.connect(broker, port,keepalive=3600)
        return client

    def run(self):
        client = self.connect_mqtt()
        self.publish_introduction(client)
        while self.round < self.total_round :
            print('aqui antes choice')
            self.publish_choice(client)
            print('aqui depois choice')
            print(self.round)
            #self.chosen = 1
            self.publish_agg(client)
            #self.round+=1
            print('aqui depois agg')
            gc.collect()
            if self.chosen == 1:
                print(f'MAE:{self.mae}')
                print(f'RMSE:{self.rmse}')
                print(f'R2:{self.score}')
            if self.round == self.total_round:
                pred = self.model.predict(self.x_test)
                pred_rescaled = self.target_scaler.inverse_transform(pred)
                y_test_rescaled =  self.target_scaler.inverse_transform(self.y_test)
                self.mae = mean_absolute_error(y_test_rescaled, pred_rescaled)
                self.rmse = mean_squared_error(y_test_rescaled, pred_rescaled, squared=False)
                self.score = r2_score(y_test_rescaled, pred_rescaled)
                model_path = './cluster_models/balanced_clusters/1_step_models/'+ self.building_class +'.h5'
                if not os.path.isfile(model_path):
                    self.model.save(model_path)
                print(f'MAE:{self.mae}')
                print(f'RMSE:{self.rmse}')
                print(f'R2:{self.score}')
                with open('Resultados/balanced_clusters/GenomeDatasetClustered_1_step/no_aggregation.csv','a') as fd:
                    result_csv_append = csv.writer(fd)
                    result_csv_append.writerow([self.mae, self.score, self.rmse, self.chosen_building_id])
                    break
                
                
            client.loop_start()
        client.loop_stop()
        client.disconnect()



if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_seed(0)
    random.seed(datetime.datetime.now().timestamp())
    chosen_building_id = str(sys.argv[1])
    clean_path = '/home/nocs/Dataset_Genome_Project_2/Dataset/Clean/'
    dataset_path = '/home/nocs/Federated PubSub/emqx/Genome_Separate_Buildings/'
    diff = False
    n_steps = 10
    k = 0
    horizon = 1 
    
    #x_train,x_test, y_train, y_test, target_scaler, features_scalers, chosen_building_id, building_class = load_data.load_data_genome(clean_path,n_steps, horizon, k,diff)
    
    #building_list = os.listdir(dataset_path)
    #building_list = [x.split('.')[0] for x in building_list]
    #chosen_building_id = random.choice(building_list)
    building_class = chosen_building_id.split('_')[1]
    building = pd.read_csv(dataset_path + chosen_building_id)
    #while len(building.index) < 1000:
    #    chosen_building_id = random.choice(building_list)
    #    building_class = chosen_building_id.split('_')[1]
    #    building = pd.read_csv(dataset_path + chosen_building_id + '.csv')
    x_train,x_test, y_train, y_test, target_scaler, features_scalers = load_data.prepara_dataset(building,n_steps, horizon, k,diff=diff)
    print(len(x_train.index))
    model = Smart_home.CNN_LSTM_compile(x_train, y_train, horizon)
    for i in model.get_weights():
        print(i.shape)
    fc = FederatedClient(chosen_building_id,building_class, x_train, y_train, x_test, y_test, model, target_scaler, features_scalers)
    fc.run()
    file = open('no_cluster.txt','a')
    file.write(str(fc.building_class) + '\n')
