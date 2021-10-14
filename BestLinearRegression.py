# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 02:18:12 2021

@author: taoso
"""
from ML import *
import ML as ML
from  DataProcessing import *
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
from DataProcessing import CutAndLinear, realized_volatility, trainValidSplitNp
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import sklearn 
import multiprocessing


def BestLinearModel(train_book, train_file, stock_id, file, listOfIntervalNumbers = [3,5,10,12,15,20]):
    """
    This is a one-stock version

    Parameters
    ----------
    train_book : dictionary
        DESCRIPTION: raw book data
    train_file : dataFrame
        DESCRIPTION: raw labels data
    stock_id : string
        DESCRIPTION: stock_id of this stock
    listOfIntervalNumbers : array, optional
        DESCRIPTION. The default is [3,5,10,12,15,20].

    Returns
    -------
    Returns the best linear model that gives the smallest MSE, evaluated using k-cross validation.
The main parameter is the cut interval.

    """
# =============================================================================
#     with open(file, 'a') as f:
#         f.writelines('\n' + stock_id)
# =============================================================================
    modelsReturn = {}
    for intervalNumber in listOfIntervalNumbers:
        modelsReturn[intervalNumber] = []
        interval = 600//intervalNumber
        df2 = dataToPd(train_book[stock_id], train_file, stock_id)
        df2['split_60s'] = df2['seconds_in_bucket']//interval
        df3 =  pd.DataFrame(df2.groupby(['stock_id','time_id','split_60s'])['log_return'].apply(realized_volatility))
# =============================================================================
#         Now build np dataFrames to use linear regression
# =============================================================================
        dfUnstack = df3.unstack()
        dfLabels=  train_file[train_file['stock_id'] == stock_id]
        x = np.array(dfUnstack)
        y = np.array(dfLabels['target']).reshape(-1,1)
        ##shuffle 5 times, each time compute a model, compute average r2 value 
        scores = []
        for i in range(20):
            x,y = shuffle(x,y)
            x_train, y_train, x_val, y_val = trainValidSplitNp(x,y,0.5)
            regModel = sklearn.linear_model.Ridge(alpha = 0.0000001)
            regModel.fit(x_train, y_train)
            modelsReturn[intervalNumber].append(regModel)
            r2Score = regModel.score(x_val, y_val)
            scores.append(r2Score)
        meanScore = np.array(scores).mean()
        std = np.array(scores).std()
        with open(file, 'a') as f:
            f.writelines('\n'+"{stock_id}, interval = {interval}, score = {meanScore}, std ={std}".format(stock_id = stock_id, interval = interval, 
                                                                                    meanScore = meanScore, std = std))
        print("interval = {interval}, score = {meanScore}, std ={std}".format(interval = interval, meanScore = meanScore, std = std))
# =============================================================================
#     return modelsReturn    ##This can be used to return all the fitted models
# =============================================================================
        
def TestResultsParseToDataframe(txtpath):
    """
    
    Parameters 
    ----------
    txtpath : string
        DESCRIPTION: path of the txt to be parsed

    Returns
    -------
    The parsed df

    """
    df = pd.DataFrame(columns=('stock_id', 'interval', 'score', 'std'))
    return df

# =============================================================================
# The following trunk runs linear regression, for each stock_id, interval, for 20 times, and stores the resulting statistics
# into the file 'LinearRegressionResults.csv'
# =============================================================================
if __name__ == "__main__":
    train_file = pd.read_csv("../train.csv")
    train_file['stock_id'] = train_file['stock_id'].apply(lambda x: 'stock_id='+str(x))
    list_of_stocks = ["stock_id={num}".replace("{num}",str(i)) for i in range(20,130)] #the number of stocks loaded
    train_book = {}
    train_trade = {}
    for stock_id in list_of_stocks:
        try:
            path = glob.glob("../book_train.parquet/{id}/*".replace("{id}",stock_id))[0]
            path_trade = glob.glob("../trade_train.parquet/{id}/*".replace("{id}",stock_id))[0]
            train_book[stock_id] = pd.read_parquet(path)
            train_trade[stock_id] = pd.read_parquet(path_trade)
        except:
            None 
    start = time.time()
    with open('LinearModelResultsMultiprocessing.txt','a') as f:
        f.writelines('This process starts at {time}'.format(time = start))
    queue = multiprocessing.SimpleQueue()
    for stock in train_book.keys():
        print('here starts', stock)
        Process = multiprocessing.Process(target=BestLinearModel, args=(train_book, train_file, stock, 'LinearModelResultsMultiprocessing.txt'))
        Process.start()
        # BestLinearModel(train_book, train_file, stock_id=stock, file= 'linearModelsTestResults.txt')  
    for _ in train_book.keys():
        print(queue.get().keys())          
    df = pd.DataFrame(columns=('stock_id', 'interval', 'score', 'std'))
    with open('LinearModelResultsMultiprocessing.txt', 'r') as f:
        for line in f:
            line = line.replace(" ", "").replace('\n','')
            splitted = line.split(',')
            stock_id = splitted[0]
            interval= splitted[1].split('=')[1]
            score = splitted[2].split('=')[1]
            std = splitted[3].split('=')[1]
            row = {'stock_id':stock_id,'interval': interval, 'score':score, 'std':std}
            print(row)
            df = df.append(row, ignore_index = True)          
        f.close()
    df['interval'] = df['interval'].astype(int)
    df['score'] = df['score'].astype(float)
    df['std'] = df['std'].astype(float)
    df = df.sort_values(by = ['stock_id','interval'], ascending = [True, True])
    df.to_csv('LinearRegressionResults.csv', index = False)   
    df2 = pd.read_csv('LinearRegressionResults.csv')
    df2['adjustedScore'] = df2['score'] - 1.5*df2['std']
    df2
              
            