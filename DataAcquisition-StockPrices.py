# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 2020
Author: Brandi Beals
Description: Thesis Data Acquisition - Stock Prices
"""

######################################
# IMPORT PACKAGES
######################################

from datetime import datetime
import pandas as pd
import yfinance as yf
import os

######################################
# DEFIINITIONS
######################################

os.chdir(r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\Data\Prices')
now_time = datetime.now()
date_format = '%Y%m%d'
today = now_time.strftime(date_format)
end_time = datetime(2020, 10, 31)
start_time = datetime(2018, 1 , 1)
universe = pd.read_csv(r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\Data\Universe.csv')
ticker_list = universe['Ticker'].to_list()

######################################
# GET DATA
######################################

for i in ticker_list:
    # format tickers
    i = i.replace('/', '-')
    print('Retrieving data for %s...' % i)
    
    # check available dates
    mindate = yf.Ticker(i).history(period='max').index.min()
    maxdate = yf.Ticker(i).history(period='max').index.max()
    
    if mindate > start_time:
        start = mindate
    else:
        start = start_time
    
    if maxdate < end_time:
        end = maxdate
    else:
        end = end_time
    
#    print(mindate)
#    print(start)
#    print(maxdate)
#    print(end)
    
    # download data for available dates
    df = yf.download(i, start=start, end=end)
    df['Ticker'] = i
    df.to_csv(r'Prices_%s.csv' %i)

