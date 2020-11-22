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
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Flatten,Reshape
from tensorflow.keras.optimizers import SGD

######################################
# LOAD DATA
######################################

data = pd.read_csv(r'C:\Users\bbeals\Desktop\Thesis Data\Data.csv')
#X = data.values #convert dataframe to series

######################################
# PREPARE DATA
######################################

## One-hot encode categorical variables
#train_labels = to_categorical(train_labels)

## Define independent variables
features = data.loc[:,'BETA_ACWI':]
features = features.drop(['MM FORMULA'], axis=1)

## Define dependent variables
return_1D_adjclose = data.loc[:,['Return_1D_AdjClose']]
return_1D_close = data.loc[:,['Return_1D_Close']]
return_30D_adjclose = data.loc[:,['Return_30D_AdjClose']]
return_30D_close = data.loc[:,['Return_30D_Close']]

######################################
# DEFIINITIONS
######################################

## Define which dependent variable is used
returns = return_1D_close

## Define size of training, validation, and test sets
train_size = len(data) * 0.60
validation_size = len(data) * 0.20
test_size = len(data) * 0.20

## Define settings for plots
plt.rcParams["font.family"] = "serif"

######################################
# SPLIT DATA
######################################

## Create training, validation, and test sets
x_train = features.loc[0:train_size-1,:]
y_train = returns.loc[0:train_size-1,:]
x_validation = features.loc[train_size:train_size+validation_size-1,:]
y_validation = returns.loc[train_size:train_size+validation_size-1,:]
x_test = features.loc[train_size+validation_size:len(data),:]
y_test = returns.loc[train_size+validation_size:len(data),:]

## Generate plot showing how data is split
plt.title('How are the observations split?')
plt.plot(y_train, color='black')
plt.plot(y_validation, color='tab:blue')
plt.plot(y_test, color='tab:gray')
plt.legend(['Training Set', 'Validation Set', 'Testing Set'])
plt.show()

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

## https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
## runs, but autoencoder returns nan (not a number)

# Encoder
encoder = Sequential()
encoder.add(Flatten(input_shape=[192,]))
#encoder.add(Dense(400, activation="relu"))
#encoder.add(Dense(200, activation="relu"))
encoder.add(Dense(48, activation="relu"))
encoder.add(Dense(12, activation="relu"))
#encoder.add(Dense(3, activation="relu"))

# Decoder
decoder = Sequential()
decoder.add(Dense(48,input_shape=[12], activation='relu'))
#decoder.add(Dense(100, activation='relu'))
#decoder.add(Dense(200, activation='relu'))
#decoder.add(Dense(400, activation='relu'))
decoder.add(Dense(192, activation="relu"))
decoder.add(Reshape([192,]))

# Autoencoder
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss="mse")
autoencoder.fit(x_train, x_train, epochs=50)

# Generate plot showing reduced dimensions
encoded_2dim = encoder.predict(x_train)
AE = pd.DataFrame(encoded_2dim, columns = ['X1', 'X2'])
AE['target'] = y_train
sns.lmplot(x='X1', y='X2', data=AE, hue='target', fit_reg=False, size=10)


## https://quantdare.com/dimensionality-reduction-method-through-autoencoders/
## runs, but autoencoder returns nan (not a number)

# Fixed dimensions
input_dim = x_train.shape[1]
encoding_dim = 3

# Number of neurons in each Layer of encoders
input_layer = Input(shape=(input_dim, ))
encoder_layer_1 = Dense(48, activation="tanh")(input_layer)
encoder_layer_2 = Dense(12, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)

# Crear encoder model
encoder = Model(inputs=input_layer, outputs=encoder_layer_3)

# Use the model to predict the factors which sum up the information of interest rates.
encoded_data = pd.DataFrame(encoder.predict(x_train))
encoded_data.columns = ['factor_1', 'factor_2', 'factor_3']


## https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial



## https://blog.keras.io/building-autoencoders-in-keras.html



