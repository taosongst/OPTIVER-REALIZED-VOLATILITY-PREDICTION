# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:52:20 2021

@author: taoso
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import random
import shutil
import datetime
import numpy as np
import DataProcessing as DP
import pandas as pd
import multiprocessing
from functools import reduce
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
import tensorflow_addons as tfa


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
from tensorflow.keras.layers import LSTM, SimpleRNN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import time
import math 
from tensorboard.plugins.hparams import api as hp

linearResults = pd.read_csv('LinearRegressionResults.csv')

train_book, train_file, train_trade = DP.RawDataInitialization(range(0,130))
list_of_stocks = train_book.keys()
         
data_frames = [pd.read_csv('globalData/'+stock_id+'.csv') for stock_id in list_of_stocks]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time_id', 'split_60s'],
                                             how='outer'), data_frames)
df_merged_reindexed = df_merged.set_index(['time_id', 'split_60s']).fillna(0)

df = pd.read_csv('4_inputs_results.csv', index_col=0)


# df['labels'] = df['criterion'].map(reverse_label)

# =============================================================================
# This part is the model using global data
# =============================================================================
list_of_stocks_skip = df.index
mse_4inputs = {}
for stock_id in list(train_book.keys()):
    if stock_id  in list_of_stocks_skip:
        continue
# for stock_id in list(train_):
    print("Now starts {stock_id}".format(stock_id = stock_id))
    # stock_id = 'stock_id=52'
    stock_id_as_int = stock_id.split('=')[1]
    data, labels = DP.RawDataToBookAndTradeNp(stock_id_as_int)
    train_data, train_labels, valid_data, valid_labels = DP.trainValidSplitNp(data, labels, 0.85)
    mean = train_labels.mean()
    std = train_labels.std()
    train_labelsNormalized, valid_labelsNormalized = (train_labels-mean)/std, (valid_labels-mean)/std   
    
    
    np_test = np.transpose(np.array(df_merged_reindexed.to_xarray().to_array()), [1,2,0]).reshape(3830, 6, 112,2)
    cut = train_data.shape[0]
    globalTrain = np_test[:cut]
    globalValid = np_test[cut:]
    if (data.shape[0] != 3830):
        with open('MissingTimeId.txt', 'w') as f:
            f.writelines("{stock} has missing time_id".format(stock = stock_id))
        continue
    
    interval = 120
    df2 = DP.RawDataToPd(train_book[stock_id], train_file, stock_id)
    df2['split_60s'] = df2['seconds_in_bucket']//interval
    df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
    
    interval = 30
    # df4 = DP.RawDataToPd(train_book[stock_id], train_file, stock_id)
    df4 = DP.RawDataToBookAndTradePd(60)
    df4['split_{interval}s'.format(interval = interval)] = df4['seconds_in_bucket']//interval
    df5 =  pd.DataFrame(df4.groupby(['stock_id',
                                     'time_id',
                                     'split_{interval}s'.format(interval = interval)])[['log_return', 'log_return_trade']].apply(realized_volatility))
    
    dataFromIntervals = np.array(df3.unstack())
    dataFromSmallIntervals =np.array(df5.unstack())
    dataFromSmallIntervals = np.transpose(dataFromSmallIntervals.reshape(3830, 2, 600//interval),[0,2,1]) #2 = features = length of ['log_return', 'log_return_trade']
    
    trainFromInterval = dataFromIntervals[:cut]
    validFromInterval = dataFromIntervals[cut:]
    
    trainFromSmallInterval = dataFromSmallIntervals[:cut]
    validFromSmallInterval = dataFromSmallIntervals[cut:]
    
    trainDataset = tf.data.Dataset.from_tensor_slices(({"input_600s": train_data,
                                                        "input_global": globalTrain,
                                                        'input_intervals':trainFromInterval,
                                                        'input_smallIntervals': trainFromSmallInterval }, 
                                                       train_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True).repeat(3)
    
    validDataset = tf.data.Dataset.from_tensor_slices(({"input_600s": valid_data, 
                                                        "input_global": globalValid, 
                                                        'input_intervals': validFromInterval,
                                                        'input_smallIntervals': validFromSmallInterval}, 
                                                       valid_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True)
    
    """
    We add some recurrent input
    """
    
    # =============================================================================
    # first input: 600s data
    # =============================================================================
    left_inputs = Input(shape=(600, 2), name='input_600s')
    x = left_inputs
    filters = 32
    dropout_rate = 0.35
    weight_decay = 0.012
    x = Normalization()(x)
    for i in range(2):
        x = Conv1D(filters = filters,
                   kernel_size= 3,
                   padding = 'same',
                   activation = 'relu',
                   kernel_regularizer=regularizers.l1(weight_decay)
                   )(x)
        x = Dropout(dropout_rate)(x)
        x = MaxPool1D()(x)
        filters *= 2
    x = Flatten()(x)
    x = Dense(units = 32, 
              activation='relu'
              )(x)
    x = Dense(1)(x)
    # =============================================================================
    # second input: 10s per interval data
    # =============================================================================
    inputs_smallIntervals = Input(shape = (20,1), name = 'input_smallIntervals')
    x2 = inputs_smallIntervals
    # x2 = Normalization()(x2)
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
    x2 = Dense(1)(x2)
    # =============================================================================
    # third input: global data
    # =============================================================================
    right_inputs = Input(shape = (6, 112, 2), name='input_global')
    y = right_inputs
    y = Normalization()(y)
    # filters = 64
    for i in range(1):
        y = Conv2D(filters = filters,
                   kernel_size=3,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=regularizers.l1(weight_decay)
                   )(y)
        y = Dropout(dropout_rate)(y)
        y = MaxPooling2D()(y)
        filters *= 2
    y = Flatten()(y)
    y = Dense(units = 48,
              activation='relu',
              )(y)
    y = Dense(1)(y)
    # =============================================================================
    # 4th input: 120s per interval data
    # =============================================================================
    inputs_intervals = Input(shape = (5,), name = 'input_intervals')
    z = inputs_intervals
    z = Normalization()(z)
    z = Flatten()(z)
    z = Dense(32,activation='relu')(z)
    z = Dense(1)(z)
    
    w = Concatenate()([x, x2, y , z])
    # w = BatchNormalization()(w)
    w = Flatten()(w)
    # w = Dense(32,
    #           activation='relu',
    #           kernel_regularizer=regularizers.l1(weight_decay))(w)
    # w = Concatenate()([w,z])
    # w = Dense(8, activation='relu')(w)
    outputs = Dense(1)(w)
    
    model = Model([left_inputs, right_inputs, inputs_intervals, inputs_smallIntervals], outputs)
    model.compile(loss =  'mse',
                  optimizer= 'adam',
                  metrics = [tf.keras.metrics.MeanSquaredError()])
    model.summary()
    
    check_point, early_stopping, learning_rate_reduction = ML.get_callbacks('test', 
                                                                            monitor='val_mean_squared_error', mode='min')
    model.fit(trainDataset, validation_data=validDataset, epochs=2000, callbacks=[early_stopping])
    # model.save('goodResults/{stock_id}'.format(stock_id = stock_id))   
    # load_model = tf.keras.models.load_model('goodResults/4inputs_stock=52_score=0.82_mse=0.13')
    # load_model.summary()
    val_mse = model.evaluate(validDataset)
    mse = model.evaluate(trainDataset)

    # mse_4inputs[stock_id] = {'val_mse':val_mse, 'mse':mse}
    
    
# dfNew = pd.DataFrame.from_dict(mse_4inputs, orient='index')   
# df = df.append(dfNew)
# df.to_csv('4_inputs_results.csv')
    
    

# =============================================================================
#
# below is a testing of replacing 600 seconds with a 10s intervals version
#
# =============================================================================
for stock in list_of_stocks:
    try: 
        stock_id = stock
        print("Now starts {stock_id}".format(stock_id = stock_id))
        # stock_id = 'stock_id=52'
        stock_id_as_int = stock_id.split('=')[1]
        data, labels = DP.RawDataToBookAndTradeNp(stock_id_as_int)
        train_data, train_labels, valid_data, valid_labels = DP.trainValidSplitNp(data, labels, 0.85)
        mean = train_labels.mean()
        std = train_labels.std()
        train_labelsNormalized, valid_labelsNormalized = (train_labels-mean)/std, (valid_labels-mean)/std   
        labelsNormalized = (labels - mean)/std
        
        np_test = np.transpose(np.array(df_merged_reindexed.to_xarray().to_array()), [1,2,0]).reshape(3830, 6, 112,2)
        cut = train_data.shape[0]
        globalTrain = np_test[:cut]
        globalValid = np_test[cut:]
        # if (data.shape[0] != 3830):
        #     with open('MissingTimeId.txt', 'w') as f:
        #         f.writelines("{stock} has missing time_id".format(stock = stock_id))
        #     continue
        
        interval = 120
        df2 = DP.RawDataToPd(train_book[stock_id], train_file, stock_id)
        df2['split_60s'] = df2['seconds_in_bucket']//interval
        df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
        
        interval = 30
        df4 = DP.RawDataToBookAndTradePd(stock_id.split('=')[1])
        df4['split_{interval}s'.format(interval = interval)] = df4['seconds_in_bucket']//interval
        df5 =  pd.DataFrame(df4.groupby(['stock_id',
                                          'time_id',
                                          'split_{interval}s'.format(interval = interval)])[[#'log_return',
                                                                                            'log_return_trade',
                                                                                            #'log_return2',
                                                                                            # 'weighted_log_return_book',                                                                                                                  
                                                                                            # 'weighted_log_return_book2',
                                                                                            # 'weighted_log_return_trade',
                                                                                            'logWeighted_log_return_trade',
                                                                                            'logWeighted_log_return_book',
                                                                                            'logWeighted_log_return_book2'
                                                                                            ]].apply(realized_volatility))
        
        # interval = 10
        # df6 = DP.RawDataToPd(train_book[stock_id], train_file, stock_id)
        # df6['split_{interval}s'.format(interval = interval)] = df6['seconds_in_bucket']//interval
        # df7 =  pd.DataFrame(df6.groupby(['stock_id',
        #                                  'time_id',
        #                                  'split_{interval}s'.format(interval = interval)])['log_return'].apply(realized_volatility))
        interval = 60
        # df4 = DP.RawDataToPd(train_book[stock_id], train_file, stock_id)
        df6 = DP.RawDataToBookAndTradePd(stock_id.split('=')[1])
        df6['split_{interval}s'.format(interval = interval)] = df6['seconds_in_bucket']//interval
        df7 =  pd.DataFrame(df6.groupby(['stock_id',
                                          'time_id',
                                          'split_{interval}s'.format(interval = interval)])[[#'log_return',
                                                                                            # 'log_return2',                                                                                                                                       
                                                                                            'log_return_trade',  
                                                                                            'logWeighted_log_return_trade', 
                                                                                            'logWeighted_log_return_book',                                                                                                                                                                   
                                                                                            'logWeighted_log_return_book2'
                                                                                            ]].apply(realized_volatility))
           
        
        
        dataFromIntervals = np.array(df3.unstack())
        dataFromSmallIntervals = np.transpose(np.array(df5.unstack()).reshape(3830,4,20),[0,2,1])
        dataFrom60intervals = np.transpose(np.array(df7.unstack()).reshape(3830,4,600//interval),[0,2,1])
        
        trainFromInterval = dataFromIntervals[:cut]
        validFromInterval = dataFromIntervals[cut:]
        
        trainFromSmallInterval = dataFromSmallIntervals[:cut]
        validFromSmallInterval = dataFromSmallIntervals[cut:]
        
        trainFrom10sIntervals = dataFrom60intervals[:cut]
        validFrom10sIntervals = dataFrom60intervals[cut:]
        
        
        trainDataset = tf.data.Dataset.from_tensor_slices(({"input_10s": trainFrom10sIntervals,
                                                            "input_global": globalTrain,
                                                            'input_intervals':trainFromInterval,
                                                            'input_smallIntervals': trainFromSmallInterval }, 
                                                            train_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True).repeat(3)
        
        validDataset = tf.data.Dataset.from_tensor_slices(({"input_10s": validFrom10sIntervals, 
                                                            "input_global": globalValid, 
                                                            'input_intervals': validFromInterval,
                                                            'input_smallIntervals': validFromSmallInterval}, 
                                                            valid_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True)
        
        classifyDataset = tf.data.Dataset.from_tensor_slices(({"input_10s": trainFrom10sIntervals,
                                                            "input_global": globalTrain,
                                                            'input_intervals':trainFromInterval,
                                                            'input_smallIntervals': trainFromSmallInterval }, 
                                                            train_labelsNormalized)).batch(1)
        
        
        
        valid_labelsNormalized.std()
        train_labelsNormalized.std()
        """
        We add some recurrent input
        """
        # overallDataAsIterator = overallDataset.as_numpy_iterator()
        # test = list(overallDataAsIterator)
        
        
        # =============================================================================
        # first input: 600s data
        # =============================================================================
        
        model = ML.ModelWithoutGlobal()
        model.summary()
        
        check_point, early_stopping, learning_rate_reduction = ML.get_callbacks('allStocks21Sep/'+stock_id, 
                                                                                monitor='val_mean_squared_error', mode='min')
        model.fit(trainDataset, validation_data=validDataset, epochs=500, callbacks=[early_stopping, check_point])
        # model.save('goodResults/{stock_id}'.format(stock_id = stock_id))   
        # load_model = tf.keras.models.load_model('goodResults/4inputs_stock=52_score=0.82_mse=0.13')
        # load_model.summary()
        val_mse = model.evaluate(validDataset)
        mse = model.evaluate(trainDataset)
        # model.save('allStocks21Sep/'+stock_id)
        with open('allStocks21Sep/test.txt', 'a') as f:
            f.write(stock_id +  ": mse = {}".format(mse) )
            # except:
            #     with open('allStocks21Sep/test.txt', 'w') as f:
            #         f.write("something is wrong with {}".format(stock_id))
    except:
        None
            


# model.save('test_with_abnormal')
# model.weights
new_model = ML.ModelWithoutGlobal()

old_weights = model.get_weights()

maxLabel = labelsNormalized.max()
minLabel = labelsNormalized.min()

new_model.set_weights(old_weights)
# compile model
# online forecast
# X = test
# oneDataPoint = X[0][0]
# for i in range(len(X)):
# 	testX, testy = X[i][0], X[i][1]
# #	testX = testX.reshape(1, 1, 1)
# 	yhat = new_model.predict(testX)
# 	print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))
    
def ReverseNormalization(x):
    return (x*std) + mean
prediction = (model.predict(classifyDataset))
errors = np.array(abs((prediction - train_labelsNormalized.reshape(3255,1))))
percentile = np.percentile(errors, 95)
def abnormal(x):
    if (abs(x) > percentile):
        return 1
    else:   
        return 0
    
classifyLabels = np.array([abnormal(x) for x in errors]).astype('uint8')
abnormalLabels = [x for x in range(len(classifyLabels)) if classifyLabels[x] == 1] 
trainLabelsUsual = [x for x in range(len(classifyLabels)) if classifyLabels[x] == 0] 


checkCount = classifyLabels.sum()

count = 0
with open('allStocks21Sep/test.txt', 'r') as f:
    for line in f:
        count += len(line.split("stock_id="))
    

# =============================================================================
# build new dataset, replace train labels with these classifyLabels
# =============================================================================

oldPrediction = model.predict(trainDataset)
classifyModel = ML.ModelWithoutGlobal(isClassifyModel=True)
classifyModel.summary()

# =============================================================================
# Now we want to train a classifier based on this model
# =============================================================================

#first we output model prediction for each data point


# =============================================================================
# Test Repeat Dataset function
# =============================================================================

testRepeated = DP.tupleDictionaryRepeatByCondition( ({"input_10s": trainFrom10sIntervals,
                                                    "input_global": globalTrain,
                                                    'input_intervals':trainFromInterval,
                                                    'input_smallIntervals': trainFromSmallInterval }, 
                                                   classifyLabels) , 1, 19)

testDatasetRepeated = tf.data.Dataset.from_tensor_slices(testRepeated).batch(ML.BATCH_SIZE, 
                                                                             drop_remainder=True).repeat(3).shuffle(10000)

classifyValidDataset = tf.data.Dataset.from_tensor_slices(({"input_10s": validFrom10sIntervals, 
                                                    "input_global": globalValid, 
                                                    'input_intervals': validFromInterval,
                                                    'input_smallIntervals': validFromSmallInterval}, 
                                                   valid_labelsNormalized)).batch(1)

aTest = list(testDatasetRepeated.as_numpy_iterator())

classifyModel.fit(testDatasetRepeated, epochs=30, callbacks=[early_stopping])

new_classifyModel = ML.ModelWithoutGlobal(isClassifyModel=True)

new_classifyModel.set_weights(classifyModel.get_weights())

classifyValidLabels = new_classifyModel.predict(classifyValidDataset)

classifyAbnormalityLabels = [ 1*(x[1]>0.5) for x in classifyValidLabels]

classifyIndexes = [ x for x in range(len(classifyAbnormalityLabels)) if classifyAbnormalityLabels[x] == 0]
classifyIndexesAbnormal = [ x for x in range(len(classifyAbnormalityLabels)) if classifyAbnormalityLabels[x] == 1]



classifyValidDatasetWithLabels = tf.data.Dataset.from_tensor_slices(({"input_10s": validFrom10sIntervals[classifyIndexes], 
                                                    "input_global": globalValid[classifyIndexes], 
                                                    'input_intervals': validFromInterval[classifyIndexes],
                                                    'input_smallIntervals': validFromSmallInterval[classifyIndexes]}, 
                                                   valid_labelsNormalized[classifyIndexes])).batch(1)

classifyValidDatasetWithLabelsAbnormal = tf.data.Dataset.from_tensor_slices(({"input_10s": validFrom10sIntervals[classifyIndexesAbnormal], 
                                                    "input_global": globalValid[classifyIndexesAbnormal], 
                                                    'input_intervals': validFromInterval[classifyIndexesAbnormal],
                                                    'input_smallIntervals': validFromSmallInterval[classifyIndexesAbnormal]}, 
                                                   valid_labelsNormalized[classifyIndexesAbnormal])).batch(1)

new_model.evaluate(classifyValidDatasetWithLabels)

# =============================================================================
# Result looks amazing! We achieve a 0.1 mse on these normal data points!
# =============================================================================

# =============================================================================
# Now we use the same new_classifyModel to isolate other dataset from train data, and train another model 
# to predict abnormality
# =============================================================================

trainDatasetAbnormal = tf.data.Dataset.from_tensor_slices(({"input_10s": trainFrom10sIntervals[abnormalLabels],
                                                    "input_global": globalTrain[abnormalLabels],
                                                    'input_intervals':trainFromInterval[abnormalLabels],
                                                    'input_smallIntervals': trainFromSmallInterval[abnormalLabels] }, 
                                                   train_labelsNormalized[abnormalLabels])).batch(32, drop_remainder=True).repeat(10).shuffle(1000)
trainDatasetUsual = tf.data.Dataset.from_tensor_slices(({"input_10s": trainFrom10sIntervals[trainLabelsUsual],
                                                    "input_global": globalTrain[trainLabelsUsual],
                                                    'input_intervals':trainFromInterval[trainLabelsUsual],
                                                    'input_smallIntervals': trainFromSmallInterval[trainLabelsUsual] }, 
                                                   train_labelsNormalized[trainLabelsUsual])).batch(256, drop_remainder=True).repeat(3).shuffle(1000)



# =============================================================================
# Use transfer learning
# =============================================================================

modelAbnormal = ML.ModelWithoutGlobal()
modelAbnormal.set_weights(model.get_weights())
finetuneLayers = len(modelAbnormal.layers)-1
for layer in modelAbnormal.layers[:finetuneLayers-2]:
    layer.trainable = False
modelAbnormal.fit(trainDatasetAbnormal, 
              validation_data= classifyValidDatasetWithLabelsAbnormal,
              epochs=3,
              callbacks=[early_stopping])

# Compare with original model
model.evaluate(trainDatasetAbnormal)
model.evaluate(classifyValidDatasetWithLabelsAbnormal)
    
modelUsual = ML.ModelWithoutGlobal()
modelUsual.set_weights(model.get_weights())
for layer in modelUsual.layers[:finetuneLayers-15]:
    layer.trainable = False
modelUsual.fit(trainDatasetUsual, 
                  validation_data= classifyValidDatasetWithLabels,
                  epochs=10,
                  callbacks=[early_stopping])
model.evaluate(trainDatasetUsual)
model.evaluate(classifyValidDatasetWithLabels)
modelAbnormal.summary()

modelAbnormal.evaluate(classifyValidDatasetWithLabelsAbnormal)

new_model.evaluate(classifyValidDatasetWithLabelsAbnormal)

valid_labelsNormalized[classifyIndexesAbnormal].std()
# =============================================================================
# Then we use classifyModel to tag validation dataset, and see if we can improve something 
# =============================================================================











