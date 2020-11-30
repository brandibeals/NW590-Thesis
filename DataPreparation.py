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
from tensorflow.keras.layers import Input,Dense,Flatten,Reshape,LSTM,Dropout

######################################
# LOAD DATA
######################################

data = pd.read_csv(r'C:\Users\bbeals\Desktop\Thesis Data\Data.csv')
#X = data.values #convert dataframe to series

######################################
# PREPARE DATA
######################################

## Fill missing values
data = data.fillna(0)
#features = features.fillna(0)

## Define independent variables
features = data.loc[:,'BETA_ACWI':]
features = features.drop(['MM FORMULA'], axis=1)

## One-hot encode categorical variables
onehot_sector = pd.get_dummies(data.SECTOR)
features = features.join(onehot_sector)

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

# Compile and fit autoencoder
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

# Encoder
input_layer = Input(shape=(input_dim, ))
encoder_layer_1 = Dense(48, activation="tanh")(input_layer)
encoder_layer_2 = Dense(12, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)

# Create model
encoder2 = Model(inputs=input_layer, outputs=encoder_layer_3)

# Generate predictions
AE_2 = pd.DataFrame(encoder2.predict(x_train), columns=['factor1','factor2','factor3'])
AE_2['target'] = y_train

# Generate plot showing reduced dimensions
plt.title('First two dimensions of encoded data, colored by single day returns')
plt.scatter(AE_2['factor1'], AE_2['factor2'], c=AE_2['target'], s=1, alpha=0.3)
plt.show()


## https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial



## https://blog.keras.io/building-autoencoders-in-keras.html




######################################
# NEURAL NETWORK
######################################

# Fixed dimensions
input_dim = x_train.shape[1]
batch_size = 50
dropout = 0.2


x_train_reshape = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_reshape = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

AE_2 = AE_2.drop(['target'], axis=1)
AE_2_reshape = np.reshape(AE_2.to_numpy(), (AE_2.shape[0], AE_2.shape[1], 1))

## https://datascienceplus.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python/

# LSTM
model = Sequential()
model.add(LSTM(units=batch_size, return_sequences=True, input_shape=[input_dim,1]))
model.add(Dropout(dropout))
model.add(LSTM(units=batch_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=batch_size, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(units=batch_size))
model.add(Dropout(dropout))
model.add(Dense(units=1))

# Compile and fit model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_reshape, y_train, epochs=1, batch_size=batch_size)

# Generate predictions
predictions = model.predict(x_test_reshape)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))


# LSTM with SAE 2 results
model2 = Sequential()
model2.add(LSTM(units=batch_size, return_sequences=True, input_shape=[encoding_dim,1]))
model2.add(Dropout(dropout))
model2.add(LSTM(units=batch_size, return_sequences=True))
model2.add(Dropout(dropout))
model2.add(LSTM(units=batch_size, return_sequences=True))
model2.add(Dropout(dropout))
model2.add(LSTM(units=batch_size))
model2.add(Dropout(dropout))
model2.add(Dense(units=1))

# Compile and fit model
model2.compile(optimizer='adam', loss='mean_squared_error')
model2.fit(AE_2_reshape, y_train, epochs=1, batch_size=batch_size)

# Generate predictions
predictions2 = model2.predict(x_test_reshape)
predictions2 = scaler.inverse_transform(predictions2)
rmse2 = np.sqrt(np.mean(((predictions2 - y_test)**2)))








