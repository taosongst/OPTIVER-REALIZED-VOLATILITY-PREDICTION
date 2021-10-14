# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:49:29 2021

@author: taoso
"""

from ML import *
from ML import trainFromHistory
import ML as ML
from  DataProcessing import realized_volatility, trainValidSplitNp, CutAndLinear, bookDataPreprocess, combineForTraining, npToDataset
from DataProcessing import RawDataInitialization, NpRepeatByCondition, NpSplitAndNormalized
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
import BestLinearRegression as BL
from sklearn.utils import shuffle
import sklearn
from sklearn.utils import shuffle



# =============================================================================
# Goal: pick a stock, say stock_id=43. We want to label our data points by 
# whetehr the value of this data point can be fitted well by linear regression.
# =============================================================================

"""
Pick stock_id=43, interval = 30, fix a linear model
"""

train_book, train_file, train_trade = RawDataInitialization([0])
stock_id = 'stock_id=0'
interval = 20
dfWithPrediction = CutAndLinear(train_book, train_file, stock_id, interval)

train_target = dfWithPrediction[['stock_id','time_id','abnormal']]
train_target = train_target.rename({'abnormal':'target'}, axis = 1)
train_target['target'] = train_target['target'].astype('uint8')

# =============================================================================
# store it into newLabelsFromLinearModel_stock_id=43.csv 
# =============================================================================

train_target.to_csv("newLabelsFromLinearModel_stock_id=43.csv")
train_np, labels_np = bookDataPreprocess(train_book[stock_id], train_target, stock_id)

# =============================================================================
# Repeat those with label == 1
# =============================================================================
trainNpRepeated, labelsNpRepeated = NpRepeatByCondition(train_np, labels_np, 1, times_repeat=2)
trainNpRepeatedShuffled, labelsNpRepeatedShuffled = shuffle(trainNpRepeated, labelsNpRepeated)

# =============================================================================
# Shuffle before passing to create dataset
# =============================================================================



trainBookAndLabel, validationBookAndLabel, normalizers = NpSplitAndNormalized(trainNpRepeatedShuffled, 
                                                                              labelsNpRepeatedShuffled, 0.8, categoricalLabels=True)
trainDataset, validDataset = npToDataset(trainBookAndLabel[0], 
                                         trainBookAndLabel[1], validationBookAndLabel[0], validationBookAndLabel[1])


# =============================================================================
# trainCombined, validCombined, normalizers = combineForTraining(list_of_stocks[0:2], train_book, train_target,  0.85, categoricalLabels=True)
# trainDataset, validDataset = npToDataset(trainCombined[0], trainCombined[1], validCombined[0], validCombined[1], ML.BATCH_SIZE )
# print(type(trainCombined[1][0]))
# 
# =============================================================================
"""
Now dfWithPrediction['abnormal'] can be used as the new labels for our classification problem
But how to adjust the data type? or whatevery, to actually build the model???
"""

##weight_decay = 0.001, lr = 0.001, dropout = 0.5 accuracy = 0.99/0.85
##weight_decay = 0.0015, lr = 0.001, dropout = 0.6, accuracy = 0.98/0/83
##weight_decay = 0.0015, lr = 0.001, dropout = 0.65, lr_decay = 0.98 ||nothing
##weight_decay = 0.0010, lr = 0.0001, dropout = 0.65, lr_decay = 0.98||200 epoch 0.58
##weight_decay = 0.0010, lr = 0.0001, dropout = 0.65, lr_decay = 0.94 Slow
##weight_decay = 0.0010, lr = 0.0001, dropout = 0.55, lr_decay = 0.94 Slow
##weight_decay = 0.0010, lr = 0.0001, dropout = 0.55, lr_decay = 0.97
##weight_decay = 0.0008, epochs/0.8, lr = 0.1, dropout = 0.51, lr_decay = 50 epochs/0.8 || too slow, 2000 epochs ~0.8
##weight_decay = 0.0007, epochs/0.8, lr = 0.1, dropout = 0.51, lr_decay = 100 epochs/0.8 || slow 1000 epochs ~ 0.82
##weight_decay = 0.0007, dropout_rate = 0.51, lr = 0.005, lr_decay = 30/0.9 || reasonably fast, 500epochs~ 082, then no improvement
## add a conv layer, way faster... , less stable
history = ML.ClassifyLinearRegressionLabels(trainDataset, validDataset)


newHistory = trainFromHistory(history, trainDataset, validDataset)
history.model.summary()
