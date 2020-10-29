# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 2020
Author: Brandi Beals
Description: Thesis Data Acquisition - Market Indicators
"""

######################################
# IMPORT PACKAGES
######################################

from datetime import datetime
import pandas_datareader.data as web
import os

######################################
# DEFIINITIONS
######################################

os.chdir(r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\Data')
now_time = datetime.now()
date_format = '%Y%m%d'
today = now_time.strftime(date_format)
end_time = datetime(2020, 10, 31)
start_time = datetime(2018, 1 , 1)

######################################
# GET DATA
######################################

# Federal Reserve Economic Data (FRED)
# https://fred.stlouisfed.org/
# found this blog helpful: https://medium.com/swlh/pandas-datareader-federal-reserve-economic-data-fred-a360c5795013

# https://fred.stlouisfed.org/series/DFF
FREDinterestrate = web.DataReader('DFF', 'fred', start_time, end_time)
# https://fred.stlouisfed.org/series/DEXUSEU
FREDexchangerate = web.DataReader('DEXUSEU', 'fred', start_time, end_time)

# Fama-French Data
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html
famafrench = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench')
famafrench = famafrench[0]
# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html
famafrenchMom = web.DataReader('F-F_Momentum_Factor_daily', 'famafrench')
famafrenchMom = famafrenchMom[0]

######################################
# COMPILE DATA FILE
######################################

output_df = FREDinterestrate.join(FREDexchangerate)
output_df = output_df.join(famafrench)
output_df = output_df.join(famafrenchMom)

output_df.to_csv(r'Market Indicators %s.csv' % today)

