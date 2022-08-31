#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import django
django.setup()
import math
import datetime
from collections import defaultdict

import backtrader as bt
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from bourse_refs_api.models import StockHistory, Stock, StockTrades

import os
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import json
import math
import datetime
from collections import defaultdict
import pickle
import backtrader as bt
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
warnings.filterwarnings('ignore')
import argparse
import datetime

import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind


# In[2]:



begin_date = '2020-01-01'
end_date = '2020-06-01'
# from begin date we start and train on (begin_date + train_len) and test on next (test_len) months 
# and so on until (number_of_months)
number_of_months = 2 # how many months should use for both train and test
train_len = 6 # number of months to train on it
test_len = 2 # number of moths to test on it
begin_date_test = '2020-06-01'

end_date_test = '2020-12-29'
forbidden_stocks = ['تسه']
correlation_percentage = 0.9
correlation_percentage_gap = 0.8
spread_ad_p_val = 0.02
ratio_ad_p_val = 0.02
coint_p_val = 0.02
params = {'begin_date': '2019-01-01', 'end_date':'2019-5-29', 'forbidden_stocks': ['تسه'] ,
         'correlation_percentage': 0.95, 'spread_ad_p_val': 0.01, 'ratio_ad_p_val': 0.01, 
         'coint_p_val': 0.01}


# Run cell below just once and it will make local files for faster usage

# In[3]:


from bourse_refs_api import models
stocks = models.Stock.objects.all()
stock_table = pd.DataFrame(list(stocks.values()))
stock_history = models.StockHistory.objects.all()
stock_history_table = pd.DataFrame(list(stock_history.values()))

stock_id = []
for i in stock_history:
    stock_id.append(i.stock_id)
name_id = {}
id_name = {}
for i in stocks:
    name_id[i.id] = i.name
    id_name[i.name] = i.id   

with open('name_id.pkl', 'wb') as handle:
    pickle.dump(name_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
vals = [{} for _ in range(len(stock_id))]
soli = dict(zip(stock_id, vals))

for i in stock_history:
    soli[i.stock_id][i.date.strftime('%d/%m/%Y')] = {
        "price": i.last_price,
        "volume": i.volume,
        "count": i.count
    }
out = open("soli.pkl", 'wb')
pickle.dump(soli, out)
out.close()


# In[4]:


pkl_file = open('soli.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()
ids = [item for item in data.keys()]


# In[5]:


pkl_file = open('name_id.pkl', 'rb')
name_id = pickle.load(pkl_file)
pkl_file.close()


# In[6]:


def return_sorted_pandas(_id, from_date, to_date):
    df = pd.DataFrame()
    df = pd.DataFrame.from_dict(data[_id])
    temp = df.T
    temp['Date'] = temp.index
    temp['Date'] = pd.to_datetime(temp['Date'], format='%d/%m/%Y')
    temp.sort_values(by=['Date'], inplace=True)


    mask = (temp['Date'] > from_date) & (temp['Date'] <= to_date)
    ret = temp[mask]
    return ret


# In[7]:


def handle_nan(df1, df2):
    
    df = pd.concat([df1, df2], axis=1)
    head1 = df.columns[0]
    head2 = df.columns[1]

    df = df.apply (pd.to_numeric, errors='coerce')
    df = df.dropna()

    return df[head1], df[head2]


# In[8]:


def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr



# In[9]:


def data_to_df(begin, end):
    
    names = []
    df = pd.DataFrame()
    data_num = len(ids)
    idx = pd.date_range(begin, end)
    empty_pairs = []
#     print(idx)
    for i in range(data_num):
        x = True
        for stock in forbidden_stocks:
            if stock in name_id[ids[i]]:
                x = False
        if x:
            idd = ids[i]
            s = return_sorted_pandas(idd, begin, end)
            s.index = pd.DatetimeIndex(s.index, dayfirst=True)
            s = s.reindex(idx)
            if s['price'].isnull().all():
                empty_pairs.append(i)
            if not s['price'].isnull().all():
                df = pd.concat([df,s['price']], axis = 1)
                names.append(i)
    df.columns = names
    return df, empty_pairs


# In[10]:


def pairs_filter(dataset, good_pairs):
    best_pairs = []
    for i, j in good_pairs:
        try:
            x, y = handle_nan(dataset[i], dataset[j])
            Spread_ADF = adfuller(x - y)
            result = ts.coint(x, y)
            p_val = result[1]
            Ratio_ADF = adfuller(x / y)

            if Spread_ADF[1] < spread_ad_p_val and p_val < coint_p_val and Ratio_ADF[1] < ratio_ad_p_val:
                best_pairs.append([i, j])
#                 print('Top correlated pairs are ',i, j , corr_matrix[i][j])
        except:
            pass
    return best_pairs


# In[11]:


def correlated_pairs(dataset):
    good_pairs = []
    for i in dataset.columns:
        for j in dataset.columns:
            if i < j:
                if dataset[i].corr(dataset[j]) > correlation_percentage:
                    good_pairs.append([i, j])
#     temp = get_top_abs_correlations(dataset)
    
#     for index, value in temp.items():
#         if value > correlation_percentage:
#             good_pairs.append(index)
    return pairs_filter(dataset, good_pairs)
    


# In[12]:


def return_pairs_for_trading(begin_time=begin_date, train_len=train_len):
    date1 = datetime.datetime.strptime(begin_time, "%Y-%m-%d")
    date2 = date1 + relativedelta(months=+train_len)
    data1, empty1 = data_to_df(str(date1.date()), str(date2.date()))
    remain_good_pairs = correlated_pairs(data1)
    
    return remain_good_pairs


# ### This Section is added to parsa's code for prepairing data to feed to backtrader

# In[ ]:


class BourseDataBase(bt.feed.DataBase):
    lines = ('last_deal_price', 'yesterday_price', 'value', 'count')
    
    def start(self):
        raise NotImplementedError("'start' should be implemented to provide 'self.it'")
    
    def _load(self):
        try:
            stock_history = next(self.it)
        except StopIteration:
            return False
        self.lines.datetime[0] = bt.date2num(stock_history.date)
        self.lines.high[0] = stock_history.max_price
        self.lines.low[0] = stock_history.min_price
        self.lines.close[0] = stock_history.last_price
        self.lines.last_deal_price[0] = stock_history.last_deal_price
        self.lines.open[0] = stock_history.first_price
        self.lines.yesterday_price[0] = stock_history.yesterday_price
        self.lines.value[0] = stock_history.value
        self.lines.volume[0] = stock_history.volume
        self.lines.count[0] = stock_history.count
        return True


# In[ ]:


class LocalBourseData(BourseDataBase):
    params = (('local_repo', None),)
    
    def start(self):
        stock_name = self.p.dataname
        stock_histories = []
        if stock_name in self.p.local_repo:
            for delta_days in range((self.p.todate - self.p.fromdate).days + 1):
                date = self.p.fromdate.date() + datetime.timedelta(days=delta_days)
                if date in self.p.local_repo[stock_name]:
                    stock_histories.append(self.p.local_repo[stock_name][date]) 
        self.it = iter(stock_histories)

