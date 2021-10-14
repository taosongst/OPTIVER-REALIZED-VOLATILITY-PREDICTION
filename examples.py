# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:03:59 2021

@author: taoso
"""

from DataProcessing import *
import DataProcessing as DP


train_book, train_file, train_trade = DP.dataInitialization([0,1])

df = pd.read_csv("LinearRegressionResults.csv")
