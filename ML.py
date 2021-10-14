# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:38:03 2021

@author: taoso
"""

"""
This file is used to define various machine learning models 
"""
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import sklearn
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate, SimpleRNN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
import time
from sklearn.linear_model import LinearRegression
import keras
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers.experimental.preprocessing import Normalization


BATCH_SIZE = 512

def get_callbacks(savePath, 
                  earlingStopingPatience = 50, 
                  lr_reduction_factor = 0.90, 
                  lr_reduction_patience = 30,
                  monitor = 'val_accuracy', 
                  mode = 'max'
                  ):   
    check_point = ModelCheckpoint(savePath, save_weights_only = False, save_best_only = True, monitor = monitor, verbose=1)
    early_stopping = EarlyStopping(monitor = monitor, mode = mode, patience = earlingStopingPatience)
    learning_rate_reduction = ReduceLROnPlateau(monitor = monitor, 
                                                patience = lr_reduction_patience, factor =lr_reduction_factor)
    return check_point, early_stopping, learning_rate_reduction

def AdamOPtimizer(initial_learning_rate = 0.001):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)
    adamOptimizer = tf.keras.optimizers.Adam(
    learning_rate= lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    return adamOptimizer
    

def RNN_model(input_shape, dropout_rate, weight_decay, batch_size= BATCH_SIZE):
    """
    Parameters
    ----------
    input_shape : list of integers
        DESCRIPTION: input shape
    dropout_rate : float
        DESCRIPTION: dropout rate
    weight_decay : float
        DESCRIPTION: weight_decay
    batch_size : integer, optional
        DESCRIPTION: batch size. The default is BATCH_SIZE.

    Returns
    -------
    rnn_model : model 
        DESCRIPTION: a compiled rnn model

    """
    rnn_model = Sequential([        
        Conv1D(128,(3),padding = 'SAME', activation = 'relu', 
               kernel_regularizer = regularizers.l2(weight_decay), batch_input_shape = (batch_size, 600, 1)),
        Dropout(dropout_rate),
        MaxPool1D(2),
        BatchNormalization(),
        Conv1D(128,(3),activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        MaxPool1D(2),
        Conv1D(256,(3),activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        MaxPool1D(2),
        BatchNormalization(),
        tf.keras.layers.GRU(600, stateful = True, return_sequences = True, kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        BatchNormalization(),
        tf.keras.layers.GRU(600, stateful = True,  return_sequences = False, kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Flatten(),
        BatchNormalization(),
        Dense(units = 64, activation = 'relu',kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        BatchNormalization(),
        Dense(units = 64, activation = 'relu',kernel_regularizer = regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    adamOptimizer = AdamOPtimizer()
    rnn_model.compile(optimizer = adamOptimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    return rnn_model

def LSTM_model(input_shape, dropout_rate, weight_decay, batch_size=BATCH_SIZE):
    lstm_model = Sequential([        
        Conv1D(64,(3),padding = 'SAME', activation = 'relu', 
               kernel_regularizer = regularizers.l2(weight_decay), batch_input_shape = (batch_size, 600, 1)),
        Dropout(dropout_rate),
        MaxPool1D(3, padding = 'same'),
        BatchNormalization(),   
# =============================================================================
#         Conv1D(128,(3),activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
#         Dropout(dropout_rate),
#         MaxPool1D(3, padding = 'same'),
#         BatchNormalization(),
# =============================================================================
# =============================================================================
#         Conv1D(128,(3),activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
#         Dropout(dropout_rate),
#         MaxPool1D(3, padding = 'same'),
#         BatchNormalization(),
# =============================================================================
#         Conv1D(32,(3),activation = 'relu', kernel_regularizer = regularizers.l2(weight_decay)),
#         Dropout(dropout_rate),
#         MaxPool1D(2),
        tf.keras.layers.LSTM(64, stateful = True, return_sequences = True, kernel_regularizer = regularizers.l1(weight_decay)),
        Dropout(dropout_rate),
# =============================================================================
#         tf.keras.layers.LSTM(128, stateful = True,  return_sequences = True, kernel_regularizer = regularizers.l2(weight_decay)),
#         Dropout(dropout_rate),
# =============================================================================
        Flatten(),
        Dense(units = 64, activation = 'relu',kernel_regularizer = regularizers.l1(weight_decay)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    adamOptimizer = AdamOPtimizer(0.0003)
    lstm_model.compile(optimizer = adamOptimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    return lstm_model

def trainModel(modelGenerator, trainDataset, validDataset, savePath, drop_rate = 0.5, weight_decay = 0.01):
    """

    Parameters
    ----------
    modelGenerator : a function that returns a model
        DESCRIPTION.
    trainDataset : tf dataset
        DESCRIPTION.
    validDataset : TYPE
        DESCRIPTION.
    savePath : TYPE
        DESCRIPTION.
    drop_rate : TYPE, optional
        DESCRIPTION. The default is 0.5.
    weight_decay : TYPE, optional
        DESCRIPTION. The default is 0.01.

    Returns
    -------
    history : TYPE
        DESCRIPTION.

    """
    model = modelGenerator((600,1), drop_rate, weight_decay)
    print("model summary:" ,model.summary())
    #add tensorboard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir = "tb_callback_dir", histogram_freq=1)
    check_point, early_stopping, learning_rate_reduction = get_callbacks(savePath)
    def trainModel(model, dataSet, validation_data, epochs = 500):
        history = model.fit(dataSet, epochs = epochs, validation_data = validation_data,
                            callbacks = [tensorboard_callback, check_point, early_stopping], verbose=2) #without callback
        return history
    history = trainModel(model, trainDataset, validDataset)
    frame = pd.DataFrame(history.history)
    acc_plot = frame.plot(y = "val_mean_squared_error", title="mse vs Epochs", legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="val_mse")
    return history

def trainFromHistory(history, trainDataset, validDataset, initial_rate = 0.0001):
    model = history.model
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir = "tb_callback_dir", histogram_freq=1)
    check_point, early_stopping, learning_rate_reduction = get_callbacks('recentModel')
    def trainModel(model, dataSet, validation_data, epochs = 500):
        history = model.fit(dataSet, epochs = epochs, validation_data = validation_data,
                            callbacks = [tensorboard_callback, learning_rate_reduction], verbose=2) #without callback
        return history
    newHistory = trainModel(model, trainDataset, validDataset)
    frame = pd.DataFrame(history.history)
    acc_plot = frame.plot(y = "val_mean_squared_error", title="mse vs Epochs", legend=False)
    acc_plot.set(xlabel="Epochs", ylabel="val_mse")
    return newHistory



def train_saved_model(path, trainSet, validationSet, savedPath):
    thisModel = tf.keras.models.load_model(path)
    initial_learning_rate = 0.00001 #instead of E-3 use E-4 pr e-5
    adamOptimizer = AdamOPtimizer(initial_learning_rate)
    thisModel.compile(optimizer = adamOptimizer, loss = 'mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    check_point, early_stopping, learning_rate_reduction = get_callbacks(savedPath)
    def trainModel(model, dataSet, validation_data, epochs = 5):
        history = model.fit(dataSet, epochs = epochs, validation_data = validation_data,
                        callbacks = [check_point, early_stopping]) #without callback
        return history
    history = trainModel(thisModel, trainSet, validationSet)
    return history


"""
This is a model aimed to classify labels defined using linear regression
"""
def ClassifyLinearRegressionLabels(trainDataset, validDataset, weight_decay = 0.0008, dropout_rate = 0.51):
    lstm_model = Sequential([        
    Conv1D(64,(3),padding = 'SAME', activation = 'relu', 
           batch_input_shape = (BATCH_SIZE, 600, 1), kernel_regularizer = regularizers.l1(weight_decay)),
    MaxPool1D(3, padding = 'same'),
    BatchNormalization(),
    Conv1D(64,(3),padding = 'SAME', activation = 'relu', kernel_regularizer = regularizers.l1(weight_decay)),
    MaxPool1D(3, padding = 'same'),
    BatchNormalization(),
    tf.keras.layers.LSTM(128, stateful = True, return_sequences = True, kernel_regularizer = regularizers.l1(weight_decay)),
    Dropout(dropout_rate),
    Flatten(),
    Dense(units = 64, activation = 'relu',kernel_regularizer = regularizers.l1(weight_decay)),
    Dropout(dropout_rate),
    Dense(2, activation='softmax')
])
    adamOptimizer = AdamOPtimizer(0.0001)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.005, momentum = 0.9)
    check_point, early_stopping, learning_rate_reduction = get_callbacks('RecentModel')
    lstm_model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    history = lstm_model.fit( trainDataset , validation_data=validDataset , epochs=20, callbacks=[learning_rate_reduction])
    return history
    
    


def ModelWithoutGlobal(isClassifyModel = False, useLinearOnly = False):  
    interval = 60
    left_inputs = Input(shape=(600//interval, 4), name='input_10s')
    x = left_inputs
    filters = 32
    dropout_rate = 0.18
    weight_decay = 0.005
    x = Normalization()(x)
    for i in range(2):
        x = Conv1D(filters = filters,
                   kernel_size= 3,
                   padding = 'same',
                   activation = 'relu',
                   kernel_regularizer=regularizers.l1(weight_decay)
                   )(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPool1D(2)(x)
        filters *= 2
    x = Flatten()(x)
    x = Dense(units = 32, 
              activation='relu'
              )(x)
    x = Dense(2)(x)
    
    # x3 = left_inputs
    # x3 = Normalization()(x3)
    # x3 = Conv1D(32, 
    #             kernel_size = 3, 
    #             padding = 'same', 
    #             activation = 'relu', 
    #             kernel_regularizer=regularizers.l1(weight_decay)
    #             )(x3)
    # x3 = SimpleRNN(32, return_sequences=True, return_state=False)(x3)
    # x3 = Dropout(dropout_rate)(x3)
    # # x3 = SimpleRNN(32, return_sequences= False, return_state=False)(x3)
    # # x3 = Dropout(dropout_rate)(x3)
    # x3 = Flatten()(x3)
    # x3 = Dense(32,
    #             activation='relu'
    #             )(x3)
    # x3 = Dense(1)(x3)
    
    x3 = left_inputs
    x3 = Normalization()(x3)
    # filters = 64
    for i in range(1):
        x3 = Conv1D(filters = filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=regularizers.l1(weight_decay)
                   )(x3)
        x3 = Dropout(dropout_rate)(x3)
        x3 = MaxPool1D()(x3)
        filters *= 2
    x3 = Flatten()(x3)
    x3 = Dense(units = 48,
              activation='relu',
              )(x3)
    x3 = Dense(2)(x3)
    # =============================================================================
    # second input: 10s per interval data
    # =============================================================================
    inputs_smallIntervals = Input(shape = (20,4), name = 'input_smallIntervals')
    x2 = inputs_smallIntervals
    x2 = Normalization()(x2)
    x2 = Conv1D(filters = 16,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l1(weight_decay))(x2)
    x2 = MaxPool1D(2)(x2)
    x2 = SimpleRNN(32, 
              return_sequences=False, 
              return_state=False)(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(32, activation='relu')(x2)
    x2 = Dense(2)(x2)
    
    
    # =============================================================================
    # third input: global data
    # =============================================================================
    # right_inputs = Input(shape = (6, 112, 2), name='input_global')
    # y = right_inputs
    # y = Normalization()(y)
    # # filters = 64
    # for i in range(1):
    #     y = Conv2D(filters = filters,
    #                kernel_size=3,
    #                padding='same',
    #                activation='relu',
    #                kernel_regularizer=regularizers.l1(weight_decay)
    #                )(y)
    #     y = Dropout(dropout_rate)(y)
    #     y = MaxPooling2D()(y)
    #     filters *= 2
    # y = Flatten()(y)
    # y = Dense(units = 48,
    #           activation='relu',
    #           )(y)
    # y = Dense(1)(y)
    # =============================================================================
    # 4th input: 120s per interval data
    # =============================================================================
    inputs_intervals = Input(shape = (5,), name = 'input_intervals')
    z = inputs_intervals
    # z = Normalization()(z)
    z = Flatten()(z)
    z = Dense(32,activation='relu')(z)
    z = Dense(1)(z)
    
    if (useLinearOnly == False):
        w = Concatenate()([x, x2, z, x3])
    else:
        w = Concatenate()([x2, z])
    # w = Concatenate()([x,x2,y])
    # w = BatchNormalization()(w)
    w = Flatten()(w)
    # w = Dense(32,
    #           activation='relu',
    #           kernel_regularizer=regularizers.l1(weight_decay))(w)
    # w = Concatenate()([w,z])
    # w = Dense(8, activation='relu')(w)
    if(isClassifyModel == False):
        outputs = Dense(1)(w)
        
        model = Model([left_inputs, inputs_intervals, inputs_smallIntervals], outputs)
        model.compile(loss =  'mse',
                      optimizer= 'adam',
                      metrics = [tf.keras.metrics.MeanSquaredError()])
    else:
        outputs = Dense(2, activation='softmax')(w)
        model = Model([left_inputs, inputs_intervals, inputs_smallIntervals], outputs)
        model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
        
    return model

















