# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2020
Author: Brandi Beals
Description: Thesis Data Preparation
"""

######################################
# IMPORT PACKAGES
######################################

import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import random

######################################
# DEFIINITIONS
######################################

## Define settings for plots
plt.rcParams["font.family"] = "serif"

## Set working directory
#path = r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\NW590-Thesis'
path = r'C:\Users\brand\Dropbox\Masters in Predictive Analytics\590-Thesis\NW590-Thesis'
os.chdir(path)

## Get current datetime
today = datetime.now().strftime('%Y%m%d')

## Set seed for reproducibility
np.random.seed(1)
random.set_seed(1)

######################################
# PREPARE DATA
######################################

## Load data
data = pd.read_csv(r'..\Data.csv')

## Fill missing values
data = data.fillna(0)

## Define independent, numerical variables
features = data.loc[:,'BETA_ACWI':]
features = features.drop(['1M_TREASURY'], axis=1)
features = features.drop(['1Y_TREASURY'], axis=1)
features = features.drop(['MM FORMULA'], axis=1)

## One-hot encode categorical variables
onehot_sector = pd.get_dummies(data.SECTOR)
features = features.join(onehot_sector)

## Define date variables
date = data['DATE']
date = pd.to_datetime(date, format='%Y-%m-%d')

date_set = pd.DataFrame(set(date), columns=['DATE'])
date_set.sort_values(by=['DATE'], inplace=True)
date_set.reset_index(drop=True, inplace=True)

## Add TICKER and DATE back in
features = features.join(data['TICKER'])
features = features.join(date)

## Define dependent variables
simple_returns = data.loc[:,['30D_RETURNS']]
sharpe = data.loc[:,['30D_SHARPE']]

# Select 30D_RETURNS or 30D_SHARPE
#returns_type = '30D_RETURNS'
#returns = data.loc[:,[returns_type]]

# Add TICKER and DATE back in
returns = simple_returns.join(sharpe)
returns = returns.join(data['TICKER'])
returns = returns.join(date)

######################################
# SPLIT DATA
######################################

## Define size of training, validation, and test sets
train_size = int(len(date_set) * 0.60)
validation_size = int(len(date_set) * 0.20)
test_size = len(date_set) - train_size - validation_size

## Define dates used in each set
date_train = date_set.loc[0:train_size-1,:]
date_validation = date_set.loc[train_size:train_size+validation_size-1,:]
date_test = date_set.loc[train_size+validation_size:len(date_set),:]

## List the start and end dates of each set
print(date_train.DATE.iloc[0])
print(date_train.DATE.iloc[-1])
print(date_validation.DATE.iloc[0])
print(date_validation.DATE.iloc[-1])
print(date_test.DATE.iloc[0])
print(date_test.DATE.iloc[-1])

## Create training, validation, and test sets
x_train = features[(features.DATE <= date_train.DATE.iloc[-1])]
y_train = returns[(returns.DATE <= date_train.DATE.iloc[-1])]

x_validation = features[(features.DATE >= date_validation.DATE.iloc[0]) & 
                        (features.DATE <= date_validation.DATE.iloc[-1])]
y_validation = returns[(returns.DATE >= date_validation.DATE.iloc[0]) & 
                       (returns.DATE <= date_validation.DATE.iloc[-1])]

x_test = features[(features.DATE >= date_test.DATE.iloc[0])]
y_test = returns[(returns.DATE >= date_test.DATE.iloc[0])]

## Generate plot showing how data is split
plt.clf()
plt.figure(figsize=(10, 6))
plt.title('Split Using 30 Day Returns')
plt.plot(y_train.loc[:,['30D_RETURNS','DATE']].groupby('DATE').agg(['sum']), color='black')
plt.plot(y_validation.loc[:,['30D_RETURNS','DATE']].groupby('DATE').agg(['sum']), color='tab:blue')
plt.plot(y_test.loc[:,['30D_RETURNS','DATE']].groupby('DATE').agg(['sum']), color='tab:gray')
plt.legend(['Training Set', 'Validation Set', 'Testing Set'])
plt.savefig('..\Images\data_split.png', bbox_inches='tight', dpi=300)
plt.show()

plt.clf()
plt.figure(figsize=(10, 6))
plt.title('Split Using 30 Day Sharpe Ratio')
plt.plot(y_train.loc[:,['30D_SHARPE','DATE']].groupby('DATE').agg(['sum']), color='black')
plt.plot(y_validation.loc[:,['30D_SHARPE','DATE']].groupby('DATE').agg(['sum']), color='tab:blue')
plt.plot(y_test.loc[:,['30D_SHARPE','DATE']].groupby('DATE').agg(['sum']), color='tab:gray')
plt.legend(['Training Set', 'Validation Set', 'Testing Set'])
plt.savefig('..\Images\data_split_sharpe.png', bbox_inches='tight', dpi=300)
plt.show()

######################################
# SCALING
######################################

## Define scaler
scaler = MinMaxScaler(feature_range=(0, 1))

## Scale only numeric columns
#x_train = scaler.fit_transform(x_train)
x_train.iloc[:,:-2] = scaler.fit_transform(x_train.iloc[:,:-2])
y_train.iloc[:,:-2] = scaler.fit_transform(y_train.iloc[:,:-2])

#x_validation = scaler.fit_transform(x_validation)
x_validation.iloc[:,:-2] = scaler.fit_transform(x_validation.iloc[:,:-2])
y_validation.iloc[:,:-2] = scaler.fit_transform(y_validation.iloc[:,:-2])

#x_test = scaler.fit_transform(x_test)
x_test.iloc[:,:-2] = scaler.fit_transform(x_test.iloc[:,:-2])
y_test.iloc[:,:-2] = scaler.fit_transform(y_test.iloc[:,:-2])

## Correlation plot
plt.matshow(x_train.corr())

######################################
# DATA RESHAPING: OPTION 1
######################################

## 3-dimensional array with 1 timestep
x_train_reshape = x_train.to_numpy()
x_train_reshape = x_train_reshape.reshape(x_train_reshape.shape[0], 1, x_train_reshape.shape[1])
y_train_reshape = y_train.to_numpy()
y_train_reshape = y_train_reshape.reshape(y_train_reshape.shape[0], 1, y_train_reshape.shape[1])
x_validation_reshape = x_validation.to_numpy()
x_validation_reshape = x_validation_reshape.reshape(x_validation_reshape.shape[0], 1, x_validation_reshape.shape[1])
y_validation_reshape = y_validation.to_numpy()
y_validation_reshape = y_validation_reshape.reshape(y_validation_reshape.shape[0], 1, y_validation_reshape.shape[1])
x_test_reshape = x_test.to_numpy()
x_test_reshape = x_test_reshape.reshape(x_test_reshape.shape[0], 1, x_test_reshape.shape[1])
y_test_reshape = y_test.to_numpy()
y_test_reshape = y_test_reshape.reshape(y_test_reshape.shape[0], 1, y_test_reshape.shape[1])

# Print shape
print('3-Dimensional Array with 1 Timestep Shape')
print('X Train: ' + str(x_train_reshape.shape))
print('Y Train: ' + str(y_train_reshape.shape))
print('X Validation: ' + str(x_validation_reshape.shape))
print('Y Validation: ' + str(y_validation_reshape.shape))
print('X Test: ' + str(x_test_reshape.shape))
print('Y Test: ' + str(y_test_reshape.shape))

# Drop DATE and TICKER
x_train_reshape_drop = x_train_reshape[:,:,:-2]
y_train_reshape_drop = y_train_reshape[:,:,:-2]
x_validation_reshape_drop = x_validation_reshape[:,:,:-2]
y_validation_reshape_drop = y_validation_reshape[:,:,:-2]
x_test_reshape_drop = x_test_reshape[:,:,:-2]
y_test_reshape_drop = y_test_reshape[:,:,:-2]

# Convert to float type
x_train_reshape_drop = np.asarray(x_train_reshape_drop).astype('float32')
x_validation_reshape_drop = np.asarray(x_validation_reshape_drop).astype('float32')
x_test_reshape_drop = np.asarray(x_test_reshape_drop).astype('float32')

# Save data
np.savez_compressed(r'..\Data\x_train_3d_1timestep.npz', x_train_reshape_drop)
np.savez_compressed(r'..\Data\y_train_3d_1timestep.npz', y_train_reshape_drop)
np.savez_compressed(r'..\Data\x_validation_3d_1timestep.npz', x_validation_reshape_drop)
np.savez_compressed(r'..\Data\y_validation_3d_1timestep.npz', y_validation_reshape_drop)
np.savez_compressed(r'..\Data\x_test_3d_1timestep.npz', x_test_reshape_drop)
np.savez_compressed(r'..\Data\y_test_3d_1timestep.npz', y_test_reshape_drop)

# Save DATE and TICKER
y_train.loc[:,['TICKER','DATE']].to_csv(r'..\Data\train_dateticker_1timestep.csv')
y_validation.loc[:,['TICKER','DATE']].to_csv(r'..\Data\validation_dateticker_1timestep.csv')
y_test.loc[:,['TICKER','DATE']].to_csv(r'..\Data\test_dateticker_1timestep.csv')

######################################
# DATA RESHAPING: OPTION 2
######################################

## 3-dimensional ticker array
def array3d(df):
    X = list()
    ticker_set = list(set(df.TICKER))
    z = len(set(df.DATE))
    for i in range(len(ticker_set)):
        t = ticker_set[i]
        #print(t)
        a = np.array(df[(df.TICKER == t)])
        #print(a.shape)
        b = np.pad(a, [(0,z-a.shape[0]),(0,0)], mode='constant', constant_values=(-1))
        X.append(b)
    return np.array(X)

# Call function to convert dataframe to padded 3D array
x_train_array = array3d(x_train)
y_train_array = array3d(y_train)
x_validation_array = array3d(x_validation)
y_validation_array = array3d(y_validation)
x_test_array = array3d(x_test)
y_test_array = array3d(y_test)

# Print shape
print('3-Dimensional Array by Ticker Shape')
print('X Train: ' + str(x_train_array.shape))
print('Y Train: ' + str(y_train_array.shape))
print('X Validation: ' + str(x_validation_array.shape))
print('Y Validation: ' + str(y_validation_array.shape))
print('X Test: ' + str(x_test_array.shape))
print('Y Test: ' + str(y_test_array.shape))

# Drop DATE and TICKER
x_train_array_drop = x_train_array[:,:,:-2]
y_train_array_drop = y_train_array[:,:,:-2]
x_validation_array_drop = x_validation_array[:,:,:-2]
y_validation_array_drop = y_validation_array[:,:,:-2]
x_test_array_drop = x_test_array[:,:,:-2]
y_test_array_drop = y_test_array[:,:,:-2]

# Convert to float type
x_train_array_drop = np.asarray(x_train_array_drop).astype('float32')
x_validation_array_drop = np.asarray(x_validation_array_drop).astype('float32')
x_test_array_drop = np.asarray(x_test_array_drop).astype('float32')

# Save data
np.savez_compressed(r'..\Data\x_train_3d_ticker.npz', x_train_array_drop)
np.savez_compressed(r'..\Data\y_train_3d_ticker.npz', y_train_array_drop)
np.savez_compressed(r'..\Data\x_validation_3d_ticker.npz', x_validation_array_drop)
np.savez_compressed(r'..\Data\y_validation_3d_ticker.npz', y_validation_array_drop)
np.savez_compressed(r'..\Data\x_test_3d_ticker.npz', x_test_array_drop)
np.savez_compressed(r'..\Data\y_test_3d_ticker.npz', y_test_array_drop)

# Save DATE and TICKER
np.savez_compressed(r'..\Data\train_dateticker_3d_ticker.npz', y_train_array[:,:,-2:])
np.savez_compressed(r'..\Data\validation_dateticker_3d_ticker.npz', y_validation_array[:,:,-2:])
np.savez_compressed(r'..\Data\test_dateticker_3d_ticker.npz', y_test_array[:,:,-2:])

######################################
# DATA RESHAPING: OPTION 3
######################################

## 3-dimensional array with 5 timesteps
## https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-2], sequences[end_ix-1, -2:]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def matrix5day(df):
    X2, y2 = list(), list()
    ticker_set = list(set(df.TICKER))
    for i in range(len(ticker_set)):
        t = ticker_set[i]
        #print(t)
        a = np.array(df[(df.TICKER == t)])
        X3, y3 = split_sequences(a, n_steps)
        #print(X3.shape, y3.shape)
        X2.append(X3)
        y2.append(y3)
    return np.array(X2), np.array(y2)

## Set the number of timesteps
n_steps = 5

# Build combined array
matrix_train = x_train.join(y_train[['30D_RETURNS','30D_SHARPE']])
matrix_validation = x_validation.join(y_validation[['30D_RETURNS','30D_SHARPE']])
matrix_test = x_test.join(y_test[['30D_RETURNS','30D_SHARPE']])
#matrix_train = x_train.join(y_train[[returns_type]])
#matrix_validation = x_validation.join(y_validation[[returns_type]])
#matrix_test = x_test.join(y_test[[returns_type]])

# Call function to generate sequences
x_train_matrix, y_train_matrix = matrix5day(matrix_train)
x_validation_matrix, y_validation_matrix = matrix5day(matrix_validation)
x_test_matrix, y_test_matrix = matrix5day(matrix_test)

# Handle empty records
for i in range(x_train_matrix.size-1):
    if x_train_matrix[i].size == 0:
        #print(i)
        x_train_matrix = np.delete(x_train_matrix, i)
        y_train_matrix = np.delete(y_train_matrix, i)

for v in range(x_validation_matrix.size-1):
    if x_validation_matrix[v].size == 0:
        #print(v)
        x_validation_matrix = np.delete(x_validation_matrix, v)
        y_validation_matrix = np.delete(y_validation_matrix, v)

for z in range(x_test_matrix.size-1):
    if x_test_matrix[z].size == 0:
        #print(z)
        x_test_matrix = np.delete(x_test_matrix, z)
        y_test_matrix = np.delete(y_test_matrix, z)

# Convert list to array
x_train_matrix = np.concatenate(x_train_matrix)
y_train_matrix = np.concatenate(y_train_matrix)
x_validation_matrix = np.concatenate(x_validation_matrix)
y_validation_matrix = np.concatenate(y_validation_matrix)
x_test_matrix = np.concatenate(x_test_matrix)
y_test_matrix = np.concatenate(y_test_matrix)

# Print shape
print('3-Dimensional Array with 5 Timesteps Shape')
print('X Train: ' + str(x_train_matrix.shape))
print('Y Train: ' + str(y_train_matrix.shape))
print('X Validation: ' + str(x_validation_matrix.shape))
print('Y Validation: ' + str(y_validation_matrix.shape))
print('X Test: ' + str(x_test_matrix.shape))
print('Y Test: ' + str(y_test_matrix.shape))

# Drop DATE and TICKER
x_train_matrix_drop = x_train_matrix[:,:,:-2]
x_validation_matrix_drop = x_validation_matrix[:,:,:-2]
x_test_matrix_drop = x_test_matrix[:,:,:-2]

# Convert to float type
x_train_matrix_drop = np.asarray(x_train_matrix_drop).astype('float32')
x_validation_matrix_drop = np.asarray(x_validation_matrix_drop).astype('float32')
x_test_matrix_drop = np.asarray(x_test_matrix_drop).astype('float32')

# Save data
np.savez_compressed(r'..\Data\x_train_3d_5timestep.npz', x_train_matrix_drop)
np.savez_compressed(r'..\Data\y_train_3d_5timestep.npz', y_train_matrix)
np.savez_compressed(r'..\Data\x_validation_3d_5timestep.npz', x_validation_matrix_drop)
np.savez_compressed(r'..\Data\y_validation_3d_5timestep.npz', y_validation_matrix)
np.savez_compressed(r'..\Data\x_test_3d_5timestep.npz', x_test_matrix_drop)
np.savez_compressed(r'..\Data\y_test_3d_5timestep.npz', y_test_matrix)

# Save DATE and TICKER
np.savez_compressed(r'..\Data\train_dateticker_3d_5timestep.npz', x_train_matrix[:,:,-2:])
np.savez_compressed(r'..\Data\validation_dateticker_3d_5timestep.npz', x_validation_matrix[:,:,-2:])
np.savez_compressed(r'..\Data\test_dateticker_3d_5timestep.npz', x_test_matrix[:,:,-2:])
