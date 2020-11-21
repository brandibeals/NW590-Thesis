# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2020
Author: Brandi Beals
Description: Thesis Data Preparation
"""

######################################
# IMPORT PACKAGES
######################################

import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

######################################
# LOAD DATA
######################################

data = pd.read_csv(r'C:\Users\bbeals\Desktop\Thesis Data\Data.csv')
#X = data.values #convert dataframe to series

######################################
# PREPARE DATA
######################################

features = data.loc[:,'BETA_ACWI':]
features = features.drop(['MM FORMULA'], axis=1)

return_1D_adjclose = data.loc[:,['Return_1D_AdjClose']]
return_1D_close = data.loc[:,['Return_1D_Close']]
return_30D_adjclose = data.loc[:,['Return_30D_AdjClose']]
return_30D_close = data.loc[:,['Return_30D_Close']]

######################################
# DEFIINITIONS
######################################

returns = return_1D_close
train_size = len(data) * 0.60
validation_size = len(data) * 0.20
test_size = len(data) * 0.20

######################################
# SPLIT DATA
######################################

x_train = features.loc[0:train_size-1,:]
y_train = returns.loc[0:train_size-1,:]
x_validation = features.loc[train_size:train_size+validation_size-1,:]
y_validation = returns.loc[train_size:train_size+validation_size-1,:]
x_test = features.loc[train_size+validation_size:len(data),:]
y_test = returns.loc[train_size+validation_size:len(data),:]

pyplot.title('How are the observations split?')
pyplot.plot(y_train, color='black')
pyplot.plot(y_validation, color='tab:blue')
pyplot.plot(y_test, color='tab:gray')
pyplot.legend(['Training Set', 'Validation Set', 'Testing Set'])
pyplot.show()

######################################
# NORMALIZATION/SCALING
######################################

scaler = MinMaxScaler(feature_range=(0, 1))

x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)
x_validation = scaler.fit_transform(x_validation)
y_validation = scaler.fit_transform(y_validation)
x_test = scaler.fit_transform(x_test)
y_test = scaler.fit_transform(y_test)

######################################
# DIMENSIONALITY REDUCTION
######################################


