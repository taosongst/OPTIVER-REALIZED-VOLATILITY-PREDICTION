# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:02:19 2021

@author: taoso
"""

import pandas as pd
import numpy as np
import DataProcessing as DP

book, file, trade = DP.RawDataInitialization([60])

book = book['stock_id=60']
trade = trade['stock_id=60']

data, labels = DP.RawDataToBookAndTradeNp(60)

