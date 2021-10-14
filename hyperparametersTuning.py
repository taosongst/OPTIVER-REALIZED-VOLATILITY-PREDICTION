# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 20:25:58 2021

@author: taoso
"""

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
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
import time
from DataProcessing import CutAndLinear, dataToPd, realized_volatility, trainValidSplitNp
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import sklearn 
import multiprocessing
from tensorboard.plugins.hparams import api as hp
from ML import BATCH_SIZE
import tensorflow_datasets as tfds


"""
This file should not be imported!
"""
if __name__ != "__main__":
    raise Exception("This module should not be imported")
    

trainDataset, validDataset= DP.datasetInitialization([0,1])
iterator = trainDataset.make_one_shot_iterator()
    
 
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.3, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'mse'
# =============================================================================
# with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
#   hp.hparams_config(
#     hparams=[HP_DROPOUT, HP_OPTIMIZER],
#     metrics=[hp.Metric(METRIC_ACCURACY, display_name='mse')],
#   )
# =============================================================================
 
 
def LSTM_modelTuning(input_shape, hparams, weight_decay, trainData, validData, batch_size=BATCH_SIZE):
    lstm_model = Sequential([        
        Conv1D(64,(3),padding = 'SAME', activation = 'relu', 
               kernel_regularizer = regularizers.l2(weight_decay), batch_input_shape = (batch_size, 600, 1)),
        Dropout(hparams[HP_DROPOUT]),
        MaxPool1D(3, padding = 'same'),
        BatchNormalization(),   
        tf.keras.layers.LSTM(64, stateful = True, return_sequences = True, kernel_regularizer = regularizers.l1(weight_decay)),
        Dropout(hparams[HP_DROPOUT]),
        Flatten(),
        Dense(units = 64, activation = 'relu',kernel_regularizer = regularizers.l1(weight_decay)),
        Dropout(hparams[HP_DROPOUT]),
        Dense(1)
    ])
    lstm_model.compile(optimizer = hparams[HP_OPTIMIZER], loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    history = lstm_model.fit(trainData,  epochs = 3, 
                   verbose = 2, callbacks=[tf.keras.callbacks.TensorBoard('logs'), 
                                           hp.KerasCallback('logs', hparams)], validation_data = validData)
    _, metric = lstm_model.evaluate(validData)
   
    return metric

def run(run_dir, hparams, trainData, validData):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    metric = LSTM_modelTuning((600,1), hparams, 0.001, trainData, validData)
    tf.summary.scalar(METRIC_ACCURACY, metric, step=10)
    return metric



session_num = 0
for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
  for optimizer in HP_OPTIMIZER.domain.values:
    hparams = {
        HP_DROPOUT: dropout_rate,
        HP_OPTIMIZER: optimizer,
    }
    run_name = "run-%d" % session_num
# =============================================================================
#     print('--- Starting trial: %s' % run_name)
#     print({h.name: hparams[h] for h in hparams})
# =============================================================================
    run('logs/hparam_tuning/' + run_name, hparams, trainDataset, validDataset)
    print("here finished one trial")
    print(hparams)
    print("sessions number = ", session_num)
    session_num += 1
    
"""
The following function tune some basic hyperparameters, including some units, weight_decay, learning_rate
"""   
def HyperparametersTuning(trainDataset, validDataset):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    