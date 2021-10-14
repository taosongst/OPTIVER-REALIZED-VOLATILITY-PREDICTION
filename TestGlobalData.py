# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 02:01:24 2021

@author: taoso
"""
import numpy as np
import DataProcessing as DP
import pandas as pd
import multiprocessing
from functools import reduce

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import random
import shutil
import datetime

import ML as ML
from absl import app
from absl import flags
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from sklearn.utils import shuffle

from  DataProcessing import realized_volatility, trainValidSplitNp, CutAndLinear, bookDataPreprocess, combineForTraining, npToDataset
from DataProcessing import RawDataInitialization, NpRepeatByCondition, NpSplitAndNormalized

from tensorboard.plugins.hparams import api as hp

from ML import *
import ML as ML
from  DataProcessing import *
import DataProcessing as DP
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import time
import math 

from tensorboard.plugins.hparams import api as hp

"""
This file aimds to define a function that can be used to process global data.
"""

"""
First process method: for each stock, time_id, split into 10*60s, compute realized
vol, then combine them together. Maybe find a way to keep some indexes when turned into 
a np array?
"""

"""
3830 time ids * 100 stocks * 6 intervals * features (at least 2)

"""
train_book, train_file, train_trade = DP.RawDataInitialization(range(0,130))
list_of_stocks = train_book.keys()
         

data_frames = [pd.read_csv('globalData/'+stock_id+'.csv') for stock_id in list_of_stocks]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time_id', 'split_60s'],
                                             how='outer'), data_frames)

df_merged_reindexed = df_merged.set_index(['time_id', 'split_60s']).fillna(0)

"""
This np_test is the main object we will work with
"""
np_test = np.transpose(np.array(df_merged_reindexed.to_xarray().to_array()), [1,2,0]).reshape(3830, 6, 112,2)


# np.save('globaldata', np_test)
linearResults = pd.read_csv('LinearRegressionResults.csv')
linearResults = linearResults[linearResults['interval'] == 60]

linearResults.plot.scatter(x = 'score', y = 'std')
"""Now we build a 2 inputs 1 output model"""

"""
Prepare the dataset for training, stock_id=115
"""
model_mse = {}


for stock in list(list_of_stocks):
    stock_id_as_int = stock.split('=')[1]
    data, labels = DP.RawDataToBookAndTradeNp(stock_id_as_int)
    train_data, train_labels, valid_data, valid_labels = DP.trainValidSplitNp(data, labels, 0.85)
    mean = train_labels.mean()
    std = train_labels.std()
    train_labelsNormalized, valid_labelsNormalized = (train_labels-mean)/std, (valid_labels-mean)/std   
    
    cut = train_data.shape[0]
    if (data.shape[0] != 3830):
        with open('MissingTimeId.txt', 'w') as f:
            f.writelines("{stock} has missing time_id".format(stock = stock))
        continue
    
    globalTrain = np_test[:cut]
    globalValid = np_test[cut:]
    
    check_nan = np.isnan(np.sum(globalTrain))
    
    trainDataset = tf.data.Dataset.from_tensor_slices(({"input_left": train_data,
                                                        "input_right": globalTrain}, train_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True)
    
    validDataset = tf.data.Dataset.from_tensor_slices(({"input_left": valid_data, 
                                                        "input_right": globalValid}, valid_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True)
    
    """ 
    Build a simple two inputs model, one with conv1D and lstm, another with conv2D and dense
    """
    
    left_inputs = Input(shape=(600, 2), name='input_left')
    x = left_inputs
    filters = 32
    dropout_rate = 0.42
    weight_decay = 0.005
    x = Normalization()(x)
    for i in range(2):
        x = Conv1D(filters = filters,
                   kernel_size= 3,
                   padding = 'same',
                   activation = 'relu',
                   kernel_regularizer=regularizers.l2(weight_decay)
                   )(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPool1D()(x)
        filters *= 2
    x = Flatten()(x)
    x = Dense(units = 64, 
              activation='relu'
              )(x)
    
    right_inputs = Input(shape = (6, 112, 2), name='input_right')
    y = right_inputs
    y = Normalization()(y)
    # filters = 64
    for i in range(1):
        y = Conv2D(filters = filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=regularizers.l2(weight_decay)
                   )(y)
        y = Dropout(dropout_rate)(y)
        y = MaxPooling2D()(y)
        filters *= 2
    y = Flatten()(y)
    y = Dense(units = 48,
              activation='relu',
              )(y)
    z = Concatenate()([x,y])
    z = Flatten()(z)
    outputs = Dense(1)(z)
    
    model = Model([left_inputs, right_inputs], outputs)
    model.compile(loss =  'mse',
                  optimizer= 'adam',
                  metrics = [tf.keras.metrics.MeanSquaredError()])
    model.summary()
    
    check_point, early_stopping, learning_rate_reduction = ML.get_callbacks('test', 
                                                                            monitor='val_mean_squared_error', mode='min')
    model.fit(trainDataset, validation_data=validDataset, epochs=500, callbacks=[early_stopping])
    mse = model.evaluate(validDataset)
    model_mse[stock] = mse
    
    
df = pd.DataFrame(model_mse)  
df.to_csv('globalModelFirstTest.csv')  
    
score_improved = {}   
for stock in model_mse.keys():
    value = linearResults[linearResults['stock_id'] == stock]['score']
    print(type(value))
    score_improved[stock] = np.array((1-model_mse[stock][1] -value )/value)[0]
    
    












