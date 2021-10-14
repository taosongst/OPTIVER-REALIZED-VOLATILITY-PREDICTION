# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:14:37 2021

@author: taoso
"""

"""
This file contains all the functions we use to preprocess data
"""

import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import sklearn
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
import time
from sklearn.linear_model import LinearRegression
import ML as ML


    

def RawDataInitialization(rangeOfStockId):
    train_file = pd.read_csv("../train.csv")
    train_file['stock_id'] = train_file['stock_id'].apply(lambda x: 'stock_id='+str(x))
    list_of_stocks = ["stock_id={num}".replace("{num}",str(i)) for i in rangeOfStockId] #the number of stocks loaded
    train_book = {}
    train_trade = {}
    for stock_id in list_of_stocks:
        try:
            path = glob.glob("../book_train.parquet/{id}/*".replace("{id}",stock_id))[0]
            path_trade = glob.glob("../trade_train.parquet/{id}/*".replace("{id}",stock_id))[0]
            train_book[stock_id] = pd.read_parquet(path)
            train_trade[stock_id] = pd.read_parquet(path_trade)
        except Exception:
            None
    return train_book, train_file, train_trade      
 
"""
This function have not been tested, use with caution. 
"""    
def datasetInitialization(rangeOfStockId, categoricalLabels = False, returnNormalizers = False):
    """

    Parameters
    ----------
    rangeOfStockId : 
        list of stock ids, e.g. [0, 5, 100]
    categoricalLabels : 
        DESCRIPTION: whether the label is categorical. The default is False.
    returnNormalizers : bool
        DESCRIPTION: whether to return normalizers. The default is False.

    Returns
    -------
        DESCRIPTION: returns trainDataset and validDataset (and probably normalizers)

    """
    train_book, train_file, train_trade = RawDataInitialization(rangeOfStockId)
    list_of_stocks = train_book.keys()
    trainCombined, validCombined, normalizers = combineForTraining(list_of_stocks, train_book, train_file,  0.85)
    trainDataset, validDataset = npToDataset(trainCombined[0], trainCombined[1], validCombined[0], validCombined[1], ML.BATCH_SIZE )
    if not returnNormalizers:
        return trainDataset, validDataset 
    else:
        return trainDataset, validDataset, normalizers

def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def bookDataPreprocess(_dataFrame , _train_target, _stock_id):
    '''
    Parameters
    ----------
    _dataFrame : dataframe
        DESCRIPTION: raw dataframe, containing information of a single stock
    _train_target : dataframe
        DESCRIPTION: all training labels. Could contain labels of other stocks.
    _stock_id : string
        DESCRIPTION: stock id

    Returns
    -------
    _train_np : np array
        DESCRIPTION: train np array, with missing seconds filled, wap and log_return of wap computed
    _train_label_np : np array
        DESCRIPTION: labels of stock_id

    '''
    _dataFrame['wap'] = (_dataFrame['bid_price1'] * _dataFrame['ask_size1'] +
                       _dataFrame['ask_price1'] * _dataFrame['bid_size1']) / (
                        _dataFrame['bid_size1']+ _dataFrame['ask_size1'])
    _lst1 = _dataFrame['time_id'].unique().tolist()
    _col1 = np.repeat(_lst1,600)
    _col2 = [[x for x in range(600)] * len(_lst1)][0]
    _df = pd.DataFrame(list(zip(_col1,_col2)),
                      columns = ['time_id', 'seconds_in_bucket'])
    _df2 = pd.merge(_df,_dataFrame, on = ['time_id', 'seconds_in_bucket'], how = 'left')
    _df2 = _df2.fillna(method = 'ffill')
    _df2['log_return'] = _df2.groupby(['time_id'])['wap'].apply(log_return)
    _df2 = _df2.pivot(index = 'time_id', columns = 'seconds_in_bucket', values = ['log_return']) #change the features of each row here
    _df2 = _df2.fillna(0)
    _df2 = pd.merge(_df2, _train_target[_train_target['stock_id'] == _stock_id][['time_id','target']], on = ['time_id'], how = 'left')
    _features_list = [x for x in _df2.columns if 'log_return' in x]
    _train_np = _df2[_features_list].to_numpy()
    _train_label_np = _df2['target'].to_numpy()
    _train_np = np.expand_dims(_train_np, axis = 2)
    return _train_np, _train_label_np


def bookDataPreprocessToPd(_dataFrame , _train_target, _stock_id):
    '''
    Parameters
    ----------
    _dataFrame : dataframe
        DESCRIPTION: raw dataframe, containing information of a single stock
    _train_target : dataframe
        DESCRIPTION: all training labels. Could contain labels of other stocks.
    _stock_id : string
        DESCRIPTION: stock id

    Returns
    -------
    _train_np : np array
        DESCRIPTION: train np array, with missing seconds filled, wap and log_return of wap computed
    _train_label_np : np array
        DESCRIPTION: labels of stock_id

    '''
    train_book, train_file, train_trade = RawDataInitialization([0])
    _lst1 = train_book['stock_id=0']['time_id'].unique().tolist()
    _dataFrame['wap'] = (_dataFrame['bid_price1'] * _dataFrame['ask_size1'] +
                       _dataFrame['ask_price1'] * _dataFrame['bid_size1']) / (
                        _dataFrame['bid_size1']+ _dataFrame['ask_size1'])
    # _lst1 = _dataFrame['time_id'].unique().tolist()   ##this was changed to use stock_id=0 as reference frame
    _col1 = np.repeat(_lst1,600)
    _col2 = [[x for x in range(600)] * len(_lst1)][0]
    _df = pd.DataFrame(list(zip(_col1,_col2)),
                      columns = ['time_id', 'seconds_in_bucket'])
    _df2 = pd.merge(_df,_dataFrame, on = ['time_id', 'seconds_in_bucket'], how = 'left')
    _df2 = _df2.fillna(method = 'ffill')
    _df2['log_return'] = _df2.groupby(['time_id'])['wap'].apply(log_return)
    _df2 = _df2.pivot(index = 'time_id', columns = 'seconds_in_bucket', values = ['log_return']) #change the features of each row here
    _df2 = _df2.fillna(0)
    _df2 = pd.merge(_df2, _train_target[_train_target['stock_id'] == _stock_id][['time_id','target']], on = ['time_id'], how = 'left')
    return _df2

def RawDataToPd(_dataFrame , _train_target, _stock_id):
    """

    Parameters
    ----------
    _dataFrame : pd
        DESCRIPTION: train_book[stock_id]
    _train_target : pd
        DESCRIPTION: useless here
    _stock_id : TYPE
        DESCRIPTION.

    Returns
    -------
    _df2 : TYPE
        DESCRIPTION.

    """
    _dataFrame['wap'] = (_dataFrame['bid_price1'] * _dataFrame['ask_size1'] +
                       _dataFrame['ask_price1'] * _dataFrame['bid_size1']) / (
                        _dataFrame['bid_size1']+ _dataFrame['ask_size1'])
    _dataFrame['wap2'] = (_dataFrame['bid_price2'] * _dataFrame['ask_size2'] +
                       _dataFrame['ask_price2'] * _dataFrame['bid_size2']) / (
                        _dataFrame['bid_size2']+ _dataFrame['ask_size2'])
    # _lst1 = _dataFrame['time_id'].unique().tolist()
    train_book, train_file, train_trade = RawDataInitialization([0])
    _lst1 = train_book['stock_id=0']['time_id'].unique().tolist()
    
    _col1 = np.repeat(_lst1,600)
    _col2 = [[x for x in range(600)] * len(_lst1)][0]
    _df = pd.DataFrame(list(zip(_col1,_col2)),
                      columns = ['time_id', 'seconds_in_bucket'])
    _df2 = pd.merge(_df,_dataFrame, on = ['time_id', 'seconds_in_bucket'], how = 'left')
    _df2 = _df2.fillna(method = 'ffill')
    #now we have a big dataframe with missing data filled, can compute log_return 
    _df2['log_return'] = _df2.groupby(['time_id'])['wap'].apply(log_return)
    _df2['log_return2'] = _df2.groupby(['time_id'])['wap2'].apply(log_return)
    _df2 = _df2.fillna(0)
    _df2['stock_id'] = _stock_id
    return _df2

def RawDataToPdTrade(_dataFrame , _train_target, _stock_id):
    '''
    Parameters
    ----------
    _dataFrame : dataframe
        DESCRIPTION: raw dataframe, containing information of a single stock
    _train_target : dataframe
        DESCRIPTION: all training labels. Could contain labels of other stocks.
    _stock_id : string
        DESCRIPTION: stock id

    Returns
    -------
    _train_np : np dataframe
        DESCRIPTION: train np array, with missing seconds filled, wap and log_return of wap computed
    _train_label_np : np array
        DESCRIPTION: labels of stock_id

    '''
    
    # _lst1 = _dataFrame['time_id'].unique().tolist()
    train_book, train_file, train_trade = RawDataInitialization([0])
    _lst1 = train_book['stock_id=0']['time_id'].unique().tolist()
    _col1 = np.repeat(_lst1,600)
    _col2 = [[x for x in range(600)] * len(_lst1)][0]
    _df = pd.DataFrame(list(zip(_col1,_col2)),
                      columns = ['time_id', 'seconds_in_bucket'])
    _df2 = pd.merge(_df,_dataFrame, on = ['time_id', 'seconds_in_bucket'], how = 'left')
    _df2['lastPrice'] = _df2.groupby(['time_id'])['price'].ffill().shift(1)
    _df2['log_return'] = _df2['price'].apply(lambda x: np.log(x)) - _df2['lastPrice'].apply(lambda x: np.log(x))
    _df2.fillna(0, inplace=True) 
    _df2['stock_id'] = _stock_id
    _df2['log_return'] = _df2['log_return'].replace(np.double('inf'), 0)
    _df2 = _df2.rename({'log_return':'log_return_trade'}, axis=1)                                                                                      
    return _df2

# =============================================================================
# def RawDataToPdTrade(_dataFrame , _train_target, _stock_id):
#     _lst1 = _dataFrame['time_id'].unique().tolist()
#     _col1 = np.repeat(_lst1,600)
#     _col2 = [[x for x in range(600)] * len(_lst1)][0]
#     _df = pd.DataFrame(list(zip(_col1,_col2)),
#                       columns = ['time_id', 'seconds_in_bucket'])
#     _df2 = pd.merge(_df,_dataFrame, on = ['time_id', 'seconds_in_bucket'], how = 'left')
#     _df2 = _df2.fillna(method = 'ffill')
#     #now we have a big dataframe with missing data filled, can compute log_return 
#     _df2['log_return'] = _df2.groupby(['time_id'])['price'].apply(log_return)
#     _df2 = _df2.fillna(0)
#     _df2['stock_id'] = _stock_id
#     return _df2
# =============================================================================

def RawDataToBookAndTradeNp(stock_id_as_int, features = ['log_return',
                                                         # 'log_return2',
                                                         # 'weighted_log_return_book',                                                                                                                  
                                                         # 'weighted_log_return_book2',
                                                         'log_return_trade',
                                                         # 'weighted_log_return_trade',
                                                         # 'logWeighted_log_return_trade',
                                                         # 'logWeighted_log_return_book',
                                                         # 'logWeighted_log_return_book2'
                                                         ]):
    num_features = len(features)
    stock_id = 'stock_id=' + str(stock_id_as_int)
    train_book, train_file, train_trade = RawDataInitialization([stock_id_as_int])
    trade = RawDataToPdTrade(train_trade[stock_id], train_file, stock_id)
    book = RawDataToPd(train_book[stock_id], train_file, stock_id)
    combined = trade.merge(book, on = ['stock_id', 'time_id', 'seconds_in_bucket'], how = 'left')
    combined['weighted_log_return_trade'] = combined['log_return_trade']*combined['size']
    combined['weighted_log_return_book'] = combined['log_return']*(combined['bid_size1'] + combined['ask_size1'])
    combined['weighted_log_return_book2'] = combined['log_return2']*(combined['bid_size2'] + combined['ask_size2'])
    combined['logWeighted_log_return_trade'] = combined['log_return_trade']*combined['size'].apply(np.log)
    combined['logWeighted_log_return_book'] = combined['log_return']*(combined['bid_size1'] + combined['ask_size1']).apply(np.log)
    combined['logWeighted_log_return_book2'] = combined['log_return2']*(combined['bid_size2'] + combined['ask_size2']).apply(np.log)
    combined = combined.fillna(0)
    combinedPivoted = combined.pivot(index = 'time_id', 
                                 columns = 'seconds_in_bucket', 
                                 values=features)
    combinedNp = np.array(combinedPivoted)
    combinedNpReshaped = combinedNp.reshape((combinedNp.shape[0],num_features, 600)) # This is crucial. Can not do reshape(3830, 600, 5)
    combinedNpReshapedShifted = np.swapaxes(combinedNpReshaped, 1, 2)
    labelsDf = train_file[train_file['stock_id'] == stock_id]
    labelsNp = np.array(labelsDf['target'])
    return combinedNpReshapedShifted, labelsNp

def RawDataToBookAndTradePd(stock_id_as_int, features = ['log_return',
                                                         # 'log_return2',
                                                         # 'weighted_log_return_book',                                                                                                                  
                                                         # 'weighted_log_return_book2',
                                                         'log_return_trade',
                                                         # 'weighted_log_return_trade',
                                                         # 'logWeighted_log_return_trade',
                                                         # 'logWeighted_log_return_book',
                                                         # 'logWeighted_log_return_book2'
                                                         ]):
    num_features = len(features)
    stock_id = 'stock_id=' + str(stock_id_as_int)
    train_book, train_file, train_trade = RawDataInitialization([stock_id_as_int])
    trade = RawDataToPdTrade(train_trade[stock_id], train_file, stock_id)
    book = RawDataToPd(train_book[stock_id], train_file, stock_id)
    combined = trade.merge(book, on = ['stock_id', 'time_id', 'seconds_in_bucket'], how = 'left')
    combined['weighted_log_return_trade'] = combined['log_return_trade']*combined['size']
    combined['weighted_log_return_book'] = combined['log_return']*(combined['bid_size1'] + combined['ask_size1'])
    combined['weighted_log_return_book2'] = combined['log_return2']*(combined['bid_size2'] + combined['ask_size2'])
    combined['logWeighted_log_return_trade'] = combined['log_return_trade']*combined['size'].apply(np.log)
    combined['logWeighted_log_return_book'] = combined['log_return']*(combined['bid_size1'] + combined['ask_size1']).apply(np.log)
    combined['logWeighted_log_return_book2'] = combined['log_return2']*(combined['bid_size2'] + combined['ask_size2']).apply(np.log)
    combined = combined.fillna(0)
    # combinedPivoted = combined.pivot(index = 'time_id', 
    #                              columns = 'seconds_in_bucket', 
    #                              values=features)
    # combinedNp = np.array(combinedPivoted)
    # combinedNpReshaped = combinedNp.reshape((combinedNp.shape[0],num_features, 600)) # This is crucial. Can not do reshape(3830, 600, 5)
    # combinedNpReshapedShifted = np.swapaxes(combinedNpReshaped, 1, 2)
    # labelsDf = train_file[train_file['stock_id'] == stock_id]
    # labelsNp = np.array(labelsDf['target'])
    return combined
def NpSplitAndNormalized(trainNp, trainLabelsnp, validationRatio, categoricalLabels = False):
    '''
    Parameters
    ----------
    trainNp : np array
        DESCRIPTION: data of a single stock
    trainLabelsnp : np array
        DESCRIPTION: labels of a single stock
    validationRatio : TYPE = double
        DESCRIPTION: train/total ratio in the train/valid split
    categoricalLabels : TYPE = boolean. optional
        DESCRIPTION: False by default. If False, then labels will be normalized. Otherwise not. 

    Returns
    -------
    trainBookAndLabel : TYPE = tuple 
        DESCRIPTION: (trainData, trainLabels) both np
    validationBookAndLabel : TYPE = tuple
        DESCRIPTION: (validData, validLabels) both np
    normalizers : TYPE= tuple
        DESCRIPTION: (dataNormalzier, labelsNormalizer, labelsRenormalizer)

    '''
    X, Y = trainNp, trainLabelsnp
    trainData, trainLabels, validData, validLabels = trainValidSplitNp(X,Y,validationRatio)
    dataNormalzier = produceNormalizer(trainData)
    labelsNormalizer, labelsRenormalizer = produceLabelsNormalizer(trainLabels)
    trainData = dataNormalzier(trainData)
    validData = dataNormalzier(validData)
    if categoricalLabels == False:               
        trainLabels = labelsNormalizer(trainLabels)
        validLabels = labelsNormalizer(validLabels)
    trainBookAndLabel = (trainData, trainLabels)
    validationBookAndLabel = (validData, validLabels)
    normalizers= (dataNormalzier, labelsNormalizer, labelsRenormalizer)
    return trainBookAndLabel, validationBookAndLabel, normalizers

def NpRepeatByCondition(trainNp, labelsNp, value, times_repeat = 1):
    '''
    Repeat certain subset of both trainNp and labelsNp

    Parameters
    ----------
    trainNp : np array
        DESCRIPTION: train np array
    labelsNp : np array
        DESCRIPTION: labels np array
    value : the criterior
        DESCRIPTION: 
    times_repeat : int
        DESCRIPTION: how many times to repeat

    Returns
    -------
    repeated traiNp and labelsNp

    '''
    indexes = [i for i in range(labelsNp.shape[0]) if labelsNp[i] == value]
    trainToBeRepeated = trainNp[indexes]
    labelsToBeRepeated = labelsNp[indexes]
    for i in range(times_repeat):
        trainNp = np.concatenate((trainNp, trainToBeRepeated), axis = 0)
        labelsNp = np.concatenate((labelsNp, labelsToBeRepeated))
    return trainNp, labelsNp
def DatasetRepeatByCondition(dataset, value, times_repeat = 1):
    '''
    Repeat certain subset of both trainNp and labelsNp

    Parameters
    ----------
    trainNp : np array
        DESCRIPTION: train np array
    labelsNp : np array
        DESCRIPTION: labels np array
    value : the criterior
        DESCRIPTION: 
    times_repeat : int
        DESCRIPTION: how many times to repeat

    Returns
    -------
    repeated traiNp and labelsNp

    '''
    trainDataAsList = list(dataset.as_numpy_iterator())
    length = len(trainDataAsList)
    for i in range(length):
        if (trainDataAsList[i][1] == value):
            for _ in range(times_repeat):
                trainDataAsList.append(trainDataAsList[i])
    # indexes = [i for i in range(labelsNp.shape[0]) if labelsNp[i] == value]
    # trainToBeRepeated = trainNp[indexes]
    # labelsToBeRepeated = labelsNp[indexes]
    # for i in range(times_repeat):
    #     trainNp = np.concatenate((trainNp, trainToBeRepeated), axis = 0)
    #     labelsNp = np.concatenate((labelsNp, labelsToBeRepeated))
    return trainDataAsList

def tupleDictionaryRepeatByCondition(tupleDict, value, times_repeat = 1):
    '''
    Repeat certain subset of both trainNp and labelsNp

    Parameters
    ----------
    trainNp : np array
        DESCRIPTION: train np array
    labelsNp : np array
        DESCRIPTION: labels np array
    value : the criterior
        DESCRIPTION: 
    times_repeat : int
        DESCRIPTION: how many times to repeat

    Returns
    -------
    repeated traiNp and labelsNp

    '''
    labels = tupleDict[1]
    indexes = [i for i in range(len(labels)) if labels[i] == value]
    dataAsDict = tupleDict[0]
    for _ in range(times_repeat):
        for key in dataAsDict.keys():
            dataAsDict[key] = np.concatenate((dataAsDict[key], dataAsDict[key][indexes]), 0)                                    
        labels = np.concatenate((labels, labels[indexes]),  0)
    return (dataAsDict, labels)
      

def combineForTraining(list_of_stocks, train_book, train_file, validationRatio, categoricalLabels = False):
    """
    Parameters
    ----------
    list_of_stocks : list
        DESCRIPTION: can be taken as the keys of train_book
    train_book : dictionary
        DESCRIPTION: dictionary of raw data
    train_file : dataFrame
        DESCRIPTION: should have stock_id, time_id, target
    validationRatio : double
        DESCRIPTION: ratio of train/total split

    Returns
    -------
    trainCombined and validCombined that can be fed into npToDataset()

    """
    print("For testing: his is the new function")
    normalizers = {}
    trainBookAndLabel = {} #each member is a np array
    validationBookAndLabel = {}
    for stock_id in list_of_stocks:
        try:           
            #train_file show be the one for this stock id
            X,Y = bookDataPreprocess(train_book[stock_id], train_file, stock_id)
            trainData, trainLabels, validData, validLabels = trainValidSplitNp(X,Y,validationRatio)
            dataNormalzier = produceNormalizer(trainData)
            labelsNormalizer, labelsRenormalizer = produceLabelsNormalizer(trainLabels)
            trainData = dataNormalzier(trainData)
            validData = dataNormalzier(validData)
            if categoricalLabels == False:               
                trainLabels = labelsNormalizer(trainLabels)
                validLabels = labelsNormalizer(validLabels)
            trainBookAndLabel[stock_id] = (trainData, trainLabels)
            validationBookAndLabel[stock_id] = (validData, validLabels)
            normalizers[stock_id] = (dataNormalzier, labelsNormalizer, labelsRenormalizer)
# =============================================================================
#This crunk was rewritten using produceNormalizer, trainValidSplit
#             bookRescaleFactor[i] = processedBook[i].std()
#             labelRescaleFactor[i] = processedLabel[i].std()
#             #normalize using std
#             processedBook[i] /= bookRescaleFactor[i]
#             processedLabel[i] /= labelRescaleFactor[i]
#             #now we split into train_set and valid set. 
#             l = int(validationRatio*len(processedBook[i]))
#             trainBookAndLabel[i] = (processedBook[i][:l], processedLabel[i][:l])
#             validationBookAndLabel[i] = (processedBook[i][l:], processedLabel[i][l:])
# =============================================================================
        except:
            None
    #combine them together
    trainCombined = (np.concatenate([trainBookAndLabel[i][0] for i in trainBookAndLabel.keys()]), 
                     np.concatenate([trainBookAndLabel[i][1] for i in trainBookAndLabel.keys()]))
    validCombined = (np.concatenate([validationBookAndLabel[i][0] for i in validationBookAndLabel.keys()]), 
                     np.concatenate([validationBookAndLabel[i][1] for i in validationBookAndLabel.keys()]))
    #create
    #print out some information
    print('this set contains',len(train_book.keys()),'stocks')
    return trainCombined, validCombined, normalizers

def npToDataset(trainData, trainLabels, validData, validLabels, BATCH_SIZE = ML.BATCH_SIZE, SHUFFLE_BUFFER_SIZE = 10000):   
    train_dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((validData, validLabels))
    train_dataset = train_dataset.shuffle(trainData.shape[0]).batch(BATCH_SIZE, drop_remainder=True) #not shuffle first
    valid_dataset = valid_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return train_dataset, valid_dataset 

def pdToDataset(trainBookAndLabel, validationBookAndLabel, batch_size = ML.BATCH_SIZE, shuffle_buffer_size = 10000):
    trainData = np.concatenate([trainBookAndLabel[i][0] for i in trainBookAndLabel.keys()])
    trainLabels = np.concatenate([trainBookAndLabel[i][1] for i in trainBookAndLabel.keys()])
    validData = np.concatenate([validationBookAndLabel[i][0] for i in validationBookAndLabel.keys()])
    validLabels = np.concatenate([validationBookAndLabel[i][1] for i in validationBookAndLabel.keys()])
    train_dataset = tf.data.Dataset.from_tensor_slices((trainData, trainLabels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((validData, validLabels))
    train_dataset = train_dataset.shuffle(trainData.shape[0]).batch(batch_size, drop_remainder=True) #not shuffle first
    valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
    return train_dataset, valid_dataset 
    

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))
"""
this function compute realized vol of every time_id and compare with train labels = future vol
"""
def addVols(dataFrame, train_file):
    df = pd.DataFrame(dataFrame.groupby(['time_id','stock_id'])['log_return'].apply(realized_volatility))
    df2 = df.merge(train_file, on = ['stock_id','time_id'], how = 'left')
    df3 = df2.rename(columns = {'log_return':'realized_vol'})
    df3['vol_change'] = (df3['target'] - df3['realized_vol'])/df3['realized_vol']
    return df3


def combineAddVols(train_book, train_file):
    processedBook = {}
    for key in train_book.keys():
        processedBook[key] = RawDataToPd(train_book[key], train_file, key)
    df = pd.concat([addVols(processedBook[key], train_file) for key in train_book.keys()])
    return df    
 
   
def volsToNp(testSplitVol):
    """

    Parameters
    ----------
    testSplitVol : dataFrame
        DESCRIPTION: the dataframe to be unstacked

    Returns
    -------
    df : dataframe
        unstacked dataframe

    """
    df = testSplitVol.unstack()
    return df


def trainValidSplitNp(data, labels, ratio):
    """
    Parameters
    ----------
    data : np
        overall data.
    labels : np
        overall labels.
    ratio : double, [0,1]
        train/overall.
        
    Returns
    -------
    trainData : np
    trainLabels : np
    validData : np
    validLabels : np

    """
    n = labels.shape[0]
    cut = int(ratio*n)
    trainData = data[:cut]
    trainLabels = labels[:cut]
    validData = data[cut:]
    validLabels = labels[cut:]
    return trainData, trainLabels, validData, validLabels

def AddIntervalVolCombined(train_book, train_file, train_trade, stock_id, interval = 100):
    df_book = AddIntervalVol(train_book, train_file, stock_id)
    df_trade = AddIntervalVolTrade(train_trade, train_file, stock_id)
    dfCombined = df_book.merge(df_trade, on = ['time_id','split_60s'], how = 'left')
    return dfCombined


def AddIntervalVol(train_book, train_file, stock_id, interval = 100):
    """
    Parameters
    ----------
    train_book : dictionary
        dictionary of raw book data.
    train_file : dataframe
        dataframe of labels.
    stock_id : string
        e.g. 'stock_id=0'.train_file = pd.read_csv("train.csv")
    interval : int, optional
        Interval of splitting. The default is 120.

    Returns
    -------
    TYPE: dataFrame
        DESCRIPTION: split 600 seconds by interval, compute realized vol for each interval, use these 600//interval numbers
         and 'target' as target to produce a linear regression
    """
    
    df2 = RawDataToPd(train_book[stock_id], train_file, stock_id)
    df2['split_60s'] = df2['seconds_in_bucket']//interval
    df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
    df3.rename({'log_return':stock_id + '_' + str(interval)+'sVol'}, axis = 1, inplace=True)
    return df3


def AddIntervalVolTrade(train_trade, train_file, stock_id, interval = 100):
    """
    Parameters
    ----------
    train_book : dictionary
        dictionary of raw book data.
    train_file : dataframe
        dataframe of labels.
    stock_id : string
        e.g. 'stock_id=0'.train_file = pd.read_csv("train.csv")
    interval : int, optional
        Interval of splitting. The default is 120.

    Returns
    -------
    TYPE: dataFrame
        DESCRIPTION: split 600 seconds by interval, compute realized vol for each interval, use these 600//interval numbers
         and 'target' as target to produce a linear regression
    """
    
    df2 = RawDataToPdTrade(train_trade[stock_id], train_file, stock_id)
    df2['split_60s'] = df2['seconds_in_bucket']//interval
    df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return_trade'].apply(realized_volatility))
    df3.rename({'log_return_trade':stock_id + '_' + str(interval)+'sVol_trade'}, axis = 1, inplace=True)
    return df3      

def StoreGlobalIntervalsVol(train_book, train_file, train_trade, stock_id):
    AddIntervalVolCombined(train_book, train_file, train_trade, stock_id).to_csv('globalData/'+stock_id+'.csv')

def CutAndLinear(train_book, train_file, stock_id, interval = 120, quantile = 0.75):
    """
    Parameters
    ----------
    train_book : dictionary
        dictionary of raw book data.
    train_file : dataframe
        dataframe of labels.
    stock_id : string
        e.g. 'stock_id=0'.train_file = pd.read_csv("train.csv")
    interval : int, optional
        Interval of splitting. The default is 120.

    Returns
    -------
    TYPE: dataFrame
        DESCRIPTION: split 600 seconds by interval, compute realized vol for each interval, use these 600//interval numbers
         and 'target' as target to produce a linear regression
    """
    
    df2 = RawDataToPd(train_book[stock_id], train_file, stock_id)
    df2['split_60s'] = df2['seconds_in_bucket']//interval
    df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
    # Now build np dataFrames to use linear regression
    dfUnstack = df3.unstack()
    dfLabels=  train_file[train_file['stock_id'] == stock_id]
    x = np.array(dfUnstack)
    y = dfLabels['target']
    x_train, y_train, x_val, y_val = trainValidSplitNp(x,y,0.6)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    regModel = sklearn.linear_model.Ridge(alpha = 0.0001)
    regModel.fit(x_train, y_train)
    print("the R2 score on valid set is {validScore}".format(validScore = regModel.score(x_val,y_val)))
    
    # With this model 'reg', we plug in back to our vols
    def UsePredictionModel(oneDimPd):
        return regModel.predict(np.array(oneDimPd).reshape([1,(600//interval)]))[0]
    
    dfPrediction = pd.DataFrame(df3.groupby(['stock_id','time_id'])['log_return'].agg(UsePredictionModel))
    dfPrediction = dfPrediction.rename({'log_return':'predictVol'}, axis = 1)
    dfReturn = dfLabels.merge(dfPrediction, on = ['stock_id','time_id'], how = 'left')
    dfReturn['vol-volPredict'] = dfReturn['target'] - dfReturn['predictVol']
    dfReturn['residual/predict'] = dfReturn['vol-volPredict']/dfReturn['predictVol']
    dfReturn['residual/target'] = abs(dfReturn['vol-volPredict']/dfReturn['target'])
    quantileValue = np.quantile(dfReturn['residual/target'], quantile)
    dfReturn['abnormal'] = (dfReturn['residual/target'] > quantileValue)
    return dfReturn

# =============================================================================
# To be debugged
# def CutAndLinear_trade(train_trade, train_file, stock_id, interval = 120, quantile = 0.75):
#     """
#     Parameters
#     ----------
#     train_book : dictionary
#         dictionary of raw book data.
#     train_file : dataframe
#         dataframe of labels.
#     stock_id : string
#         e.g. 'stock_id=0'.train_file = pd.read_csv("train.csv")
#     interval : int, optional
#         Interval of splitting. The default is 120.
# 
#     Returns
#     -------
#     TYPE: dataFrame
#         DESCRIPTION: split 600 seconds by interval, compute realized vol for each interval, use these 600//interval numbers
#          and 'target' as target to produce a linear regression
#     """
# 
#     
#     df2 = RawDataToPdTrade(train_trade[stock_id], train_file, stock_id)
#     df2['split_60s'] = df2['seconds_in_bucket']//interval
#     df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
#     # Now build np dataFrames to use linear regression
#     dfUnstack = df3.unstack()
#     dfLabels=  train_file[train_file['stock_id'] == stock_id]
#     x = np.array(dfUnstack)
#     y = dfLabels['target']
#     x_train, y_train, x_val, y_val = trainValidSplitNp(x,y,0.6)
#     print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
#     regModel = sklearn.linear_model.Ridge(alpha = 0.0001)
#     regModel.fit(x_train, y_train)
#     print("the R2 score on valid set is {validScore}".format(validScore = regModel.score(x_val,y_val)))
#     
#     # With this model 'reg', we plug in back to our vols
#     def UsePredictionModel(oneDimPd):
#         return regModel.predict(np.array(oneDimPd).reshape([1,(600//interval)]))[0]
#     
#     dfPrediction = pd.DataFrame(df3.groupby(['stock_id','time_id'])['log_return'].agg(UsePredictionModel))
#     dfPrediction = dfPrediction.rename({'log_return':'predictVol'}, axis = 1)
#     dfReturn = dfLabels.merge(dfPrediction, on = ['stock_id','time_id'], how = 'left')
#     dfReturn['vol-volPredict'] = dfReturn['target'] - dfReturn['predictVol']
#     dfReturn['residual/predict'] = dfReturn['vol-volPredict']/dfReturn['predictVol']
#     dfReturn['residual/target'] = abs(dfReturn['vol-volPredict']/dfReturn['target'])
#     quantileValue = np.quantile(dfReturn['residual/target'], quantile)
#     dfReturn['abnormal'] = (dfReturn['residual/target'] > quantileValue)
#     return dfReturn
# =============================================================================

def produceNormalizer(trainData):
    """
    Parameters
    ----------
    trainData : np array
        Description: np array of training data (not intersecting with val/test data)

    Returns
    -------
    Type: function
        DESCRIPTION: normalizer function.

    """
    mean = trainData.mean()
    std = trainData.std()
    def dataNormalzier(x):
        return (x-mean)/std
    return dataNormalzier   

"""
This function takes in training labels as np array, produce normalizer and denormalizer of training labels
"""   
def produceLabelsNormalizer(trainLabels):
    """
    Parameters
    ----------
    trainLabels : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    mean = trainLabels.mean()
    std = trainLabels.std()
    def labelsNormalizer(x):
        return (x-mean)/std
    def labelsRenormalizer(x):
        return x*std+mean
    return labelsNormalizer, labelsRenormalizer

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_book, train_file, train_trade = RawDataInitialization([0])
