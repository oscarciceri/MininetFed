import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import sys
import tensorflow as tf
import numpy as np
from numpy import dot
from numpy.linalg import norm
import math
from functools import reduce
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from paho.mqtt import client as mqtt_client
import paho.mqtt.subscribe as subscribe
import paho.mqtt.publish as publish
import time
import json
import logging
import sqlite3
import datetime
import traceback
from collections import defaultdict

#import scipy.spatial.distance as distance
logger = logging.getLogger()
logger.setLevel(logging.INFO) 

broker = 'localhost'
port = 1883
#port = 8083
topic_introduction = "introduction"
topic_choice= "choice"
topic_agg = "agg"
client_paho_id = f'python-mqtt-{random.randint(0, 1000)}'

    # username = 'emqx'
    # password = 'public'
def define_model():
 model = Sequential()
 model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 model.add(MaxPooling2D((2, 2)))
 model.add(Flatten())
 model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
 model.add(Dense(10, activation='softmax'))
 opt = SGD(learning_rate=0.01, momentum=0.9)
 model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
 return model




def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    #get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = 0
    for i in results:
        num_examples_total += i[1]

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    
    return weights_prime,num_examples_total
def aggregate_clustered(clusters):
    agg_weights = {}
    cluster_sizes = []
    for c, v in clusters.items():
        cluster_sizes.append((c,len(v)))
        weights, total_examples = aggregate(v)
        agg_weights[c] = [weights,total_examples]
    return agg_weights,cluster_sizes


class FederatedServer(object):

    def __init__(self, min_clients, number_choice_clients, max_rounds, use_number_choice=True ,chance=1, aggregate_cluster = 0):
        self.round = 0
        self.min_clients = min_clients
        self.number_choice_clients = number_choice_clients
        self.max_rounds = max_rounds
        self.global_model = define_model()
        self.clients_separate_by_classes = defaultdict(list)
        self.agg_clients = []
        self.choice_clients = []
        self.use_number_choice= use_number_choice
        self.chance = chance
        self.aggregate_cluster = aggregate_cluster
        self.connection_address = 'agg.db'
        self.connection_db = sqlite3.connect(self.connection_address, timeout=1200)
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        MODEL TEXT NOT NULL,
                        ROUND INTEGER NOT NULL,
                        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
                        """)
        self.cursor_db.execute("""
                        CREATE TABLE IF NOT EXISTS aggregate_models (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        MODEL TEXT NOT NULL,
                        ROUND INTEGER NOT NULL,
                        CLUSTER TEXT,
                        Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        );
                        """)
        self.cursor_db.execute("""SELECT name FROM sqlite_master WHERE type='table'""")
        for linha in self.cursor_db.fetchall():
           print(linha)
        self.cursor_db.close()
           
        
    #def on_log(self, client, userdata, level, buf):
    #    print("log: ",buf)
    def on_message_introduction(self, client, userdata, msg):
        client_msg = json.loads(msg.payload)
        time.sleep(0.5)
        topic = client_msg['resp_topic']
        building_class = client_msg['building_class']
        self.clients_separate_by_classes[building_class].append(topic)
        msg = {
                'total_round' : self.max_rounds,
                'connection_address' : self.connection_address
            }
        global_w = [i.tolist() if type(i) != list else i for i in self.global_model.get_weights()] 
                
        global_data = {
                    'Weights' : global_w
        }
        ct = datetime.datetime.now()
        self.cursor_db = self.connection_db.cursor()
        self.cursor_db.execute("""
                            INSERT INTO aggregate_models (MODEL, ROUND, Timestamp)
                            VALUES (?, ?, ?)
                            """, (json.dumps(global_data), self.round, ct))


        self.connection_db.commit()
        self.cursor_db.close()

        self.round = 1
        publish.single( topic, json.dumps(msg), hostname=broker)
        
    def on_message_choice(self, client, userdata, msg):
            print("esperando clientes choice")
            self.choice_clients.append(json.loads(msg.payload))
            print("Choice")
            print(len(self.choice_clients) >= self.min_clients)
            print(len(self.choice_clients))
            print(datetime.datetime.now())
            if len(self.choice_clients) >= self.min_clients:
                time.sleep(0.5)

                if self.use_number_choice:
                    chosen_clients = random.sample(self.choice_clients, self.number_choice_clients)
                    for c in self.choice_clients:

                        topic = c['resp_topic']
                        print(topic)
                        if c in chosen_clients :
                            chosen = 1
                        else:
                            chosen = 0
                        msg = {
                            'chosen' : chosen, 
                            'round' : self.round
                        }
                        publish.single( topic, json.dumps(msg), hostname=broker)

                else:
                    for c in self.choice_clients:
                        topic = c['resp_topic']
                        if float(c['random_number']) <= self.chance:
                            chosen = 1
                        else:
                            chosen = 0
                        msg = {
                            'chosen' : chosen, 
                            'round' : self.round
                        }
                        publish.single( topic, json.dumps(msg), hostname=broker)

                if self.round > self.max_rounds:
                    self.round = 1
                self.choice_clients = []
                


    def on_message_agg(self, client, userdata, msg):
            
            self.agg_clients.append(json.loads(msg.payload))
            print("agg")
            print(len(self.agg_clients) >= self.min_clients)
            print(len(self.agg_clients))
            print(datetime.datetime.now())
            if len(self.agg_clients) >= self.min_clients:
                
                time.sleep(0.5)
                self.cursor_db = self.connection_db.cursor()
                self.cursor_db.execute("""
                                    SELECT * FROM models WHERE ROUND=? ORDER BY Timestamp DESC;
                                   """, [self.round])
                if self.aggregate_cluster == 3:
                    data = []
                    classes = defaultdict(int) 
                    
                    for linha in self.cursor_db.fetchmany(self.number_choice_clients):
                        model_data = json.loads(linha[1])
                        classes[str(model_data['Cluster'])]+=1
                        data.append([[np.asarray(i) for i in model_data['Weights']],int(model_data['Size'])])
                    self.cursor_db.close()
                    agg_w,_ = aggregate(data)
                    agg_w = [i.tolist() if type(i) != list else i for i in agg_w] 
                    agg_data = {
                            'Weights' : agg_w
                        }
                    ct = datetime.datetime.now()
                    cluster_sizes = []
                    for c, v in classes.items():
                        cluster_sizes.append((c,v))
                        self.cursor_db = self.connection_db.cursor()
                        self.cursor_db.execute("""
                                INSERT INTO aggregate_models (MODEL, ROUND, CLUSTER, Timestamp)
                                VALUES (?, ?, ?, ?)
                                """, (json.dumps(agg_data), self.round, c, ct))
                        self.connection_db.commit()
                        self.cursor_db.close()
                    self.cluster_sizes = cluster_sizes
                else: 
                    data = defaultdict(list)   
                    for linha in self.cursor_db.fetchmany(self.number_choice_clients):
                        model_data = json.loads(linha[1])
                        data[str(model_data['Cluster'])].append([[np.asarray(i) for i in model_data['Weights']],int(model_data['Size'])])
                    self.cursor_db.close()
                    agg_weights,cluster_sizes = aggregate_clustered(data)
                    self.cluster_sizes = cluster_sizes
                    #self.global_model.set_weights(agg_w)
                    for c,w in agg_weights.items():
                        print(c)
                        agg_w = [i.tolist() if type(i) != list else i for i in w[0]] 
                        if self.aggregate_cluster == 2:
                            agg_ws = []
                            agg_ws.append(w)
                        elif self.round == self.max_rounds and self.aggregate_cluster == 1:
                            agg_ws = []
                            agg_ws.append(w)
                        else:
                            agg_data = {
                                'Weights' : agg_w
                            }
                            ct = datetime.datetime.now()
                            self.cursor_db = self.connection_db.cursor()
                            self.cursor_db.execute("""
                                INSERT INTO aggregate_models (MODEL, ROUND, CLUSTER, Timestamp)
                                VALUES (?, ?, ?, ?)
                                """, (json.dumps(agg_data), self.round, c, ct))
                            self.connection_db.commit()
                            self.cursor_db.close()
                    if (self.aggregate_cluster == 1 and self.round == self.max_rounds ) or self.aggregate_cluster == 2:

                        global_model_clustred,_ = aggregate(agg_ws)
                        agg_w = [i.tolist() if type(i) != list else i for i in global_model_clustred]
                        agg_data = {
                                'Weights' : agg_w
                            }
                        ct = datetime.datetime.now()
                        self.cursor_db = self.connection_db.cursor()
                        self.cursor_db.execute("""
                                INSERT INTO aggregate_models (MODEL, ROUND, CLUSTER, Timestamp)
                                VALUES (?, ?, ?, ?)
                                """, (json.dumps(agg_data), self.round, c, ct))
                        self.connection_db.commit()
                        self.cursor_db.close()
                for c in self.agg_clients:
                    topic = c['resp_topic']
                    print(topic)
                    
                    msg = {
                        'resp_topic' : topic
                    }
                    publish.single( topic, json.dumps(msg), hostname=broker)
                
                self.agg_clients = []
                self.round += 1
                if self.round > self.max_rounds:
                    self.client.disconnect()


    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)
        # Set Connecting Client ID
        client = mqtt_client.Client(client_paho_id)
        #client = mqtt_client.Client(client_paho_id,transport='websockets')
        #client.username_pw_set(username, password)
        client.on_connect = on_connect
        #client.on_log= self.on_log
        client.connect(broker, port, keepalive=3600)
        client.subscribe(topic_introduction)
        client.message_callback_add(topic_introduction, self.on_message_introduction)
        client.subscribe(topic_choice)
        client.message_callback_add(topic_choice, self.on_message_choice)
        client.subscribe(topic_agg)
        client.message_callback_add(topic_agg, self.on_message_agg)

        return client


    def run(self):
        self.client = self.connect_mqtt()
        try:
            self.client.loop_forever()
        except Exception as e:
            os.remove(fs.connection_address)
            self.client.disconnect()
            print(traceback.format_exc())
            print(e)

        finally:
            self.connection_db.close()


if __name__ == "__main__":
    #np.random.seed(0)
    #tf.random.set_seed(0)
    #random.seed(0)
    min_clients = int(sys.argv[1])
    number_choice_clients = int(sys.argv[2])
    rounds = int(sys.argv[3])
    #0: no aggregation
    #1: aggregate last round
    #2: aggregate every round
    #3: no clustering
    fs = FederatedServer(min_clients, number_choice_clients, rounds,True,1,0)
    fs.run()
    file = open('clusters.txt','a')

    file.write('balanced aggregate every round 1 step:\n')
    for i in fs.cluster_sizes:
       file.write('cluster: ' + str(i[0]) + ' size: ' + str(i) + '\n') 
    try:
        os.remove(fs.connection_address)
    except OSError:
        pass


