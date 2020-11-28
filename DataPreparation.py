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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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

## Define independent variables
features = data.loc[:,'BETA_ACWI':]
features = features.drop(['MM FORMULA'], axis=1)

## One-hot encode categorical variables
onehot = pd.get_dummies(data.SECTOR)
features = features.join(onehot)

## Fill missing values
features = features.fillna(0)

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

# Fixed dimensions
input_dim = x_train.shape[1]
encoding_dim = 3


## https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
## runs, but autoencoder returns nan (not a number)

# Encoder
encoder = Sequential()
encoder.add(Flatten(input_shape=[input_dim,]))
#encoder.add(Dense(400, activation="relu"))
#encoder.add(Dense(200, activation="relu"))
encoder.add(Dense(48, activation="relu"))
encoder.add(Dense(12, activation="relu"))
#encoder.add(Dense(encoding_dim, activation="relu"))

# Decoder
decoder = Sequential()
decoder.add(Dense(48,input_shape=[12], activation='relu'))
#decoder.add(Dense(100, activation='relu'))
#decoder.add(Dense(200, activation='relu'))
#decoder.add(Dense(400, activation='relu'))
decoder.add(Dense(input_dim, activation="relu"))
decoder.add(Reshape([input_dim,]))

# Autoencoder
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(loss="mse")
autoencoder.fit(x_train, x_train, epochs=50)

# Generate predictions
AE_1 = pd.DataFrame(encoder.predict(x_train))
AE_1['target'] = y_train

# Generate plot showing reduced dimensions
plt.title('First two dimensions of encoded data, colored by single day returns')
plt.scatter(AE_1[0], AE_1[1], c=AE_1['target'], s=1, alpha=0.3)
plt.show()


## https://quantdare.com/dimensionality-reduction-method-through-autoencoders/
## runs, but autoencoder returns nan (not a number)

# Number of neurons in each Layer of encoders
input_layer = Input(shape=(input_dim, ))
encoder_layer_1 = Dense(48, activation="tanh")(input_layer)
encoder_layer_2 = Dense(12, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)

# Crear encoder model
encoder = Model(inputs=input_layer, outputs=encoder_layer_3)

# Generate predictions
AE_2 = pd.DataFrame(encoder.predict(x_train))
AE_2['target'] = y_train

# Generate plot showing reduced dimensions
plt.title('First two dimensions of encoded data, colored by single day returns')
plt.scatter(AE_2[0], AE_2[1], c=AE_2['target'], s=1, alpha=0.3)
plt.show()


## https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial



## https://blog.keras.io/building-autoencoders-in-keras.html



