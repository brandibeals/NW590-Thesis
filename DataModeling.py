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
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,Flatten,Reshape,LSTM,Dropout
from tensorflow.keras.utils import plot_model

######################################
# DEFIINITIONS
######################################

## Define settings for plots
plt.rcParams["font.family"] = "serif"

## Set working directory
path = r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\NW590-Thesis'
os.chdir(path)

## Get current datetime
today = datetime.now().strftime('%Y%m%d')

## Set seed for reproducibility
np.random.seed(1)
random.set_seed(1)

## Set the number of timesteps
n_steps = 5

######################################
# PREPARE DATA
######################################

## Load data
data = pd.read_csv(r'C:\Users\bbeals\Desktop\Thesis Data\Data.csv')

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
# Select 30D_RETURNS or 30D_SHARPE
returns_type = '30D_RETURNS'
returns = data.loc[:,[returns_type]]

# Add TICKER and DATE back in
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
plt.title('How are observations split?')
plt.plot(y_train.loc[:,[returns_type,'DATE']].groupby('DATE').agg(['sum']), color='black')
plt.plot(y_validation.loc[:,[returns_type,'DATE']].groupby('DATE').agg(['sum']), color='tab:blue')
plt.plot(y_test.loc[:,[returns_type,'DATE']].groupby('DATE').agg(['sum']), color='tab:gray')
plt.legend(['Training Set', 'Validation Set', 'Testing Set'])
plt.savefig('..\Images\data_split.png', bbox_inches='tight', dpi=300)
plt.show()

######################################
# SCALING
######################################

## Define scaler
scaler = MinMaxScaler(feature_range=(0, 1))

## Scale only numeric columns
#x_train = scaler.fit_transform(x_train)
x_train.iloc[:,:-2] = scaler.fit_transform(x_train.iloc[:,:-2])
y_train[[returns_type]] = scaler.fit_transform(y_train[[returns_type]])

#x_validation = scaler.fit_transform(x_validation)
x_validation.iloc[:,:-2] = scaler.fit_transform(x_validation.iloc[:,:-2])
y_validation[[returns_type]] = scaler.fit_transform(y_validation[[returns_type]])

#x_test = scaler.fit_transform(x_test)
x_test.iloc[:,:-2] = scaler.fit_transform(x_test.iloc[:,:-2])
y_test[[returns_type]] = scaler.fit_transform(y_test[[returns_type]])

## Correlation plot
plt.matshow(x_train.corr())

######################################
# DATA RESHAPING
######################################

## Option 1: 2-dimensional dataframe
# Print shape
print('2-Dimensional Dataframe Shape')
print('X Train: ' + str(x_train.shape))
print('Y Train: ' + str(y_train.shape))
print('X Validation: ' + str(x_validation.shape))
print('Y Validation: ' + str(y_validation.shape))
print('X Test: ' + str(x_test.shape))
print('Y Test: ' + str(y_test.shape))

## Option 2: 3-dimensional array
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
print('3-Dimensional Array Shape')
print('X Train: ' + str(x_train_array.shape))
print('Y Train: ' + str(y_train_array.shape))
print('X Validation: ' + str(x_validation_array.shape))
print('Y Validation: ' + str(y_validation_array.shape))
print('X Test: ' + str(x_test_array.shape))
print('Y Test: ' + str(y_test_array.shape))

## Option 3: 5-day lag matrix
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
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
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

# Build combined array
matrix_train = x_train.join(y_train[[returns_type]])
matrix_validation = x_validation.join(y_validation[[returns_type]])
matrix_test = x_test.join(y_test[[returns_type]])

# Call function to generate sequences
x_train_matrix, y_train_matrix = matrix5day(matrix_train)
x_validation_matrix, y_validation_matrix = matrix5day(matrix_validation)
x_test_matrix, y_test_matrix = matrix5day(matrix_test)

# Handle empty records
for i in range(x_train_matrix.size-1):
    if x_train_matrix[i].size == 0:
        print(i)
        x_train_matrix = np.delete(x_train_matrix, i)
        y_train_matrix = np.delete(y_train_matrix, i)

# Convert list to array
x_train_matrix = np.concatenate(x_train_matrix)
y_train_matrix = np.concatenate(y_train_matrix)
x_validation_matrix = np.concatenate(x_validation_matrix)
y_validation_matrix = np.concatenate(y_validation_matrix)
x_test_matrix = np.concatenate(x_test_matrix)
y_test_matrix = np.concatenate(y_test_matrix)

# Print shape
print('3-Dimensional Timestep Shape')
print('X Train: ' + str(x_train_matrix.shape))
print('Y Train: ' + str(y_train_matrix.shape))
print('X Validation: ' + str(x_validation_matrix.shape))
print('Y Validation: ' + str(y_validation_matrix.shape))
print('X Test: ' + str(x_test_matrix.shape))
print('Y Test: ' + str(y_test_matrix.shape))

######################################
# FINALIZE DATA FOR MODELING
######################################

## Ticker Array Data Set (Option 2)
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

## Timestep Matrix Data Set (Option 3)
# Drop DATE and TICKER
x_train_matrix_drop = x_train_matrix[:,:,:-2]
x_validation_matrix_drop = x_validation_matrix[:,:,:-2]
x_test_matrix_drop = x_test_matrix[:,:,:-2]

# Convert to float type
x_train_matrix_drop = np.asarray(x_train_matrix_drop).astype('float32')
x_validation_matrix_drop = np.asarray(x_validation_matrix_drop).astype('float32')
x_test_matrix_drop = np.asarray(x_test_matrix_drop).astype('float32')

# Save data
np.savez_compressed(r'..\Data\x_train_3d.npz', x_train_matrix_drop)

######################################
# DIMENSIONALITY REDUCTION
######################################

## Fixed parameters
batch_size = 50
epochs = 50
encoding_dim = 3
num_recs = x_train_matrix_drop.shape[0]
input_dim = x_train_matrix_drop.shape[2]

## https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
# Encoder
encoder = Sequential(name='SAE_encoder')
encoder.add(Flatten(input_shape=(n_steps, input_dim)))
encoder.add(Dense(96, activation='relu', name='encoderlayer1'))
encoder.add(Dense(48, activation='relu', name='encoderlayer2'))
encoder.add(Dense(12, activation='relu', name='encoderlayer3'))
#encoder.add(Dense(encoding_dim, activation="relu"))

# Decoder
decoder = Sequential(name='SAE_decoder')
decoder.add(Dense(48, activation='relu', name='decoderlayer1', input_shape=(12,)))
decoder.add(Dense(96, activation='relu', name='decoderlayer2'))
decoder.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer3'))
decoder.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder = Sequential([encoder, decoder], name='SAE1')
autoencoder.compile(loss='mse', metrics=['accuracy'])
AE_1_history = autoencoder.fit(x_train_matrix_drop, x_train_matrix_drop, epochs=epochs)

# Print summary
print(autoencoder.summary())
print(autoencoder.layers)
# Note, the following assumes that you have the graphviz graph library and the Python interface installed
plot_model(autoencoder, to_file='\Images\model_plot.png', show_shapes=True, show_layer_names=True)

# Save model
autoencoder.save('AE_1_%s' % today)
autoencoder.save('Models/SAE_1_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
AE_1_loss = AE_1_history.history['loss']
#AE_1_val_loss = AE_1_history.history['val_loss']
plt.plot(epochs_range, AE_1_loss, label='Training loss', color='tab:blue')
#plt.plot(epochs_range, AE_1_val_loss, label='Validation loss', color='tab:gray')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE1_training_loss.png', bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
AE_1_acc = AE_1_history.history['accuracy']
#AE_1_val_acc = AE_1_history.history['val_acc']
plt.plot(epochs_range, AE_1_acc, label='Training acc', color='tab:blue')
#plt.plot(epochs_range, AE_1_val_acc, label='Validation acc', color='tab:gray')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE1_training_accuracy.png', bbox_inches='tight', dpi=300)
plt.show()

# Evaluate results on validation data
AE_1_results = autoencoder.evaluate(x_validation_matrix_drop, x_validation_matrix_drop)

# Generate encoded predictions
AE_1 = encoder.predict(x_train_matrix_drop)
print(AE_1.shape)

# Add in ticker, date, and y
print(x_train_matrix.shape)
ticker_date = x_train_matrix[:,-1:,-2:]
ticker_date = ticker_date.reshape((ticker_date.shape[0], ticker_date.shape[2]))
AE_1_output = np.column_stack((AE_1, ticker_date))

print(y_train_matrix.shape)
AE_1_output = np.column_stack((AE_1_output, y_train_matrix))

# Save predictions
np.savetxt('..\Data\SAE_training_encoded.csv', AE_1_output, delimiter=',', fmt='%s')

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First two dimensions of encoded data')
plt.scatter(AE_1[:,0], AE_1[:,2], s=1, alpha=0.3)
plt.show()

# Correlation plot of reduced dimensions
plt.matshow(np.corrcoef(AE_1))


## https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial


## https://blog.keras.io/building-autoencoders-in-keras.html






######################################
# DEFINE HOW MANY STEPS, AUTOCORRELATION
######################################

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data['Count'], lags=10)
plot_pacf(data['Count'], lags=10)


######################################
# DATA RESHAPING
######################################

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

## https://towardsdatascience.com/recurrent-neural-network-to-predict-multivariate-commodity-prices-8a8202afd853
# shifted prediction column
new_data['pred'] = new_data.Oil.shift(-1)

# data as type float
values = new_data.values
values = values.astype('float32')

# predictions and inverted scaling
yhat = model_lstm.predict(xtest)
xtest = xtest.reshape((xtest.shape[0], xtest.shape[2]))
inv_yhat = concatenate((yhat, xtest[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

ytest = ytest.reshape((len(ytest), 1))
inv_y = concatenate((ytest, xtest[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

print("MAE:  %f" % sklearn.metrics.mean_absolue_error(inv_y, inv_yhat))
print("MSE:  %f" % sklearn.metrics.mean_squared_error(inv_y, inv_yhat))
print("RMSE: %f" % math.sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat)))
print("R2:   %f" % sklearn.metrics.r2_score(inv_y, inv_yhat))

plt.plot(inv_y, label='Actual')
plt.plot(inv_yhat, label='Predicted')
plt.legend()
plt.show()




######################################
# LSTM
######################################

## Fixed dimensions
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













######################################
# POSSIBLE RESOURCES
######################################
#https://stackoverflow.com/questions/58449353/lstm-deal-with-multiple-rows-in-a-date
#https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00
#https://www.datatechnotes.com/2018/12/time-series-data-prediction-with-lstm.html
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://towardsdatascience.com/autoencoders-in-practice-dimensionality-reduction-and-image-denoising-ed9b9201e7e1
#https://machinelearningmastery.com/autoencoder-for-regression/
#https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
#https://www.datasciencecentral.com/profiles/blogs/stock-price-prediction-using-lstm-long-short-term-memory 

######################################
# UNUSED CODE SNIPPETS
######################################
def array3d(df):
    array = np.empty(shape=(0, df.shape[1]), dtype='object')
    ticker_set = list(set(df.TICKER))
    z = len(set(df.DATE))
    for i in range(len(ticker_set)):
        t = ticker_set[i]
        #print(t)
        a = np.array(df[(df.TICKER == t)])
        #print(a.shape)
        b = np.pad(a, [(0,z-a.shape[0]),(0,0)], mode='constant', constant_values=(-1))
        #print(b)
        array = np.append(array, b, axis=0)
    array = array.reshape((len(ticker_set), z, df.shape[1]))
    return array

def matrix5day2(df):
    Xarray = np.empty(shape=(0, 5, df.shape[1]-1), dtype='object')
    yarray = np.empty(shape=(0, ), dtype='object')
    ticker_set = list(set(df.TICKER))
    for i in range(len(ticker_set)):
        t = ticker_set[i]
        print(t)
        a = np.array(df[(df.TICKER == t)])
        X3, y3 = split_sequences(a, n_steps)
        print(X3.shape, y3.shape)
        Xarray = np.append(Xarray, X3, axis=0)
        yarray = np.append(yarray, y3, axis=0)
        print(Xarray.shape, yarray.shape)
    return np.array(Xarray), np.array(yarray)

brandi = list()
for i in range(x_train_matrix.shape[0]):
    print(x_train_matrix[i].shape)
    brandi.append(x_train_matrix[i].shape)
list(set(matrix_train.TICKER))[1596]
matrix_train[(matrix_train.TICKER == 'HONE')]

## https://quantdare.com/dimensionality-reduction-method-through-autoencoders/
# Encoder
input_layer = Input(shape=(input_dim, ))
encoder_layer_1 = Dense(48, activation="tanh")(input_layer)
encoder_layer_2 = Dense(12, activation="tanh")(encoder_layer_1)
encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)
# Create model
SAE = Model(inputs=input_layer, outputs=encoder_layer_3)
print(SAE.summary())
# Generate predictions
AE_2 = pd.DataFrame(SAE.predict(x_train), columns=['factor1','factor2','factor3'])
AE_2['target'] = y_train
# Generate plot showing reduced dimensions
plt.title('First two dimensions of encoded data, colored by single day returns')
plt.scatter(AE_2['factor1'], AE_2['factor2'], c=AE_2['target'], s=1, alpha=0.3)
plt.show()

# Reshaping the array from 3D array to 2D array
np.savez_compressed(r'..\Data\x_train.npz', x_train_matrix_drop.reshape(x_train_matrix_drop.shape[0], -1))
# Load data
x_train_loaded = np.load(r'..\Data\x_train.npz')
print(x_train_loaded.files)
x_train_loaded = x_train_loaded['arr_0']
# Reshape to 3D array
x_train_loaded = x_train_loaded.reshape(x_train_loaded.shape[0], x_train_loaded.shape[1] // 204, 204)