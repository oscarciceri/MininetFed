import glob
import warnings
from certifi import where
warnings.filterwarnings("ignore") 
import pandas as pd
from datetime import datetime
import math
import numpy as np
from numpy import loadtxt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K    
from tensorflow.keras.models import Sequential   # to flatten the input data
from tensorflow.keras.layers import Dense,Dropout,LSTM ,Conv1D,MaxPooling1D,Activation,TimeDistributed, Input,RepeatVector,GlobalAveragePooling1D,BatchNormalization
from matplotlib import pyplot as plt
import os
import flwr as fl
import tensorflow_addons as tfa
import csv
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
from flwr.common import NDArrays, Scalar
from typing import Dict
import re
from collections import Counter
import seaborn as sns

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def adiciona_lag(dataframe, n, horizon,diff):
  for column in dataframe.columns:
        for i in range(n):
                
                  dataframe['lag_'+ column + '_' + str(i+1)] = dataframe[column].shift(i+1)
                  dataframe.dropna(inplace=True)
                
                  if i != 0:
                    dataframe['lag_difference_' + column + '_' + str(i+1)] = dataframe['lag_' + column + '_' + str(i)] - dataframe['lag_'+ column + '_' + str(i+1)]
                  else:
                    dataframe['lag_difference_' + column + '_' + str(i+1)] = dataframe[column] - dataframe['lag_'+ column + '_' + str(i+1)]
                
        for i in range(horizon-1):
                  dataframe['horizon_' + column + '_' + str(i+1)] = dataframe[column].shift(-(i+1))
                  dataframe.dropna(inplace=True)

def prepara_dataset(dataframe, n_steps, horizon, k_features,diff=True,select=False):
    dataframe.columns = pd.io.parsers.base_parser.ParserBase({'names':dataframe.columns, 'usecols':None})._maybe_dedup_names(dataframe.columns)
    if k_features != 0:
      adiciona_lag(dataframe,n_steps,horizon,diff)
      if diff:
        columns = ['lag_difference_meter_reading_1']+dataframe.columns[dataframe.columns.str.contains('lag_difference_horizon_')].to_list()
      else:
        columns = ['meter_reading']+dataframe.columns[dataframe.columns.str.contains('horizon_meter_reading_')].to_list()
      Y = dataframe[columns]
      X = dataframe.drop(columns, 1)  

      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False, random_state=0)

    else:

      aux = dataframe.index
      df = dataframe['meter_reading']
      df = df.to_frame()
      df.index = aux
      
      adiciona_lag(df,n_steps,horizon,diff)

      if diff:
        columns = ['lag_difference_meter_reading_1']+dataframe.columns[dataframe.columns.str.contains('lag_difference_horizon_')].to_list()
      else:
        columns = ['meter_reading'] + df.columns[df.columns.str.contains('horizon_meter_reading_')].to_list()
      Y = df[columns]
      X = df.drop(columns, 1)
      x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle=False, random_state=0)

     

    x_train = x_train.fillna(x_train.mean().fillna(0))
    x_test = x_test.fillna(x_test.mean().fillna(0))
    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())


    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    scalers={}
    for column in x_train.columns:
      scaler = MinMaxScaler(feature_range=(-1,1))
      s_s = scaler.fit_transform(x_train[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      scalers['scaler_'+ column] = scaler
      x_train[column]=s_s

    for column in x_test.columns:
      scaler = scalers['scaler_'+column] 
      s_s = scaler.transform(x_test[column].values.reshape(-1, 1))
      s_s=np.reshape(s_s,len(s_s))
      x_test[column]=s_s
    
    y_train = target_scaler.fit_transform(y_train)
    y_test = target_scaler.transform(y_test)
    #print(y_test)
    return x_train, x_test, y_train, y_test, target_scaler, scalers

def load_data_genome(clean_path,n_steps, horizon, k,diff):
    files = glob.glob(clean_path + "*.csv")

    dfs = [] # empty list of the dataframes to create
    for file in files: # for each file in directory
        if file==clean_path +"weather.csv" or file==clean_path +'metadata.csv':
            continue
        meter_type = file.split("/")[6].split(".")[0].split("_")[0]# meter_type to rename the value feature
        meter = pd.read_csv(file) # load the dataset
        meter = pd.melt(meter, id_vars = "timestamp", var_name = "building_id", value_name = "meter_reading") # melt dataset
        meter["meter"] = str(meter_type) # adds column with the meter type
        dfs.append(meter) # append to list
    complete_data = pd.concat(dfs, axis=0, ignore_index=True) # concat#enate all meter
    
    data = reduce_mem_usage(complete_data)
    metadata = pd.read_csv(clean_path +'metadata.csv')
    metadata = metadata[["building_id", "site_id", "primaryspaceusage", "sqm"]]
    metadata = reduce_mem_usage(metadata)
    data = pd.merge(data, metadata, how="left", on="building_id")
    data.dropna(inplace=True)
    unique_buildings = data['building_id'].unique().tolist()
    
    chosen_building_id = random.choice(unique_buildings)
    building_class = chosen_building_id.split('_')[1]
    data_aux = data.loc[data['building_id'] == chosen_building_id]
    while len(data_aux) < 10000:
        chosen_building_id = random.choice(unique_buildings)
        building_class = chosen_building_id.split('_')[1]
        data_aux = data.loc[data['building_id'] == chosen_building_id]
    data = data_aux
    data.index = data['timestamp']
    
    data["timestamp"] = pd.to_datetime(data["timestamp"], format='%Y-%m-%d %H:%M:%S')
    data = data.loc[data['meter'] == 'electricity']
    data = data[['timestamp','building_id','meter_reading']]
    data["meter_reading"] = data['meter_reading'].astype(float).round(4)
    data.index = data['timestamp']
    building = data.drop('building_id', 1)  
    x_train,x_test, y_train, y_test, target_scaler, features_scalers = prepara_dataset(building,n_steps, horizon, k,diff=diff)
    return x_train,x_test, y_train, y_test, target_scaler, features_scalers, chosen_building_id,building_class
    
