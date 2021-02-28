# -*- coding: utf-8 -*-
"""
Created on Fri Feb 06 2021
Author: Brandi Beals
Description: Thesis LSTM Modeling
"""

######################################
# IMPORT PACKAGES
######################################

import os
from datetime import datetime
import numpy as np
from tensorflow import random
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
#random.set_seed(1)
random.set_random_seed(1)

######################################
# LOAD DATA
######################################

# Load data
SAE_train_loaded = np.load(r'..\Data\SAE_training_encoded_5timestep.npz')
print(SAE_train_loaded.files)
SAE_train_loaded = SAE_train_loaded['arr_0']

# Load dependent variable
y_train_loaded = np.load(r'..\Data\y_train_3d_5timestep.npz')
print(y_train_loaded.files)
y_train_loaded = y_train_loaded['arr_0']

# Load DATE and TICKER
train_dateticker = np.load(r'..\Data\train_dateticker_3d_5timestep.npz')
print(train_dateticker.files)
train_dateticker = train_dateticker['arr_0']

######################################
# LSTM
######################################

## Fixed parameters
y_train_loaded = y_train_loaded[:,1] # 0 = 30D_RETURNS, 1 = 30D_SHARPE
batch_size = 32
epochs = 100
obs = SAE_train_loaded.shape[0]
n_steps = SAE_train_loaded.shape[1]
input_dim = SAE_train_loaded.shape[2]

## LSTM
model = Sequential()
model.add(LSTM(160, return_sequences=True, input_shape=(n_steps, input_dim)))
model.add(LSTM(96))
model.add(Dense(1))

# Compile and fit model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(SAE_train_loaded, y_train_loaded, epochs=epochs, batch_size=batch_size)

# Print summary
print(model.summary())

# Save model
model.save('Models/LSTM_%s.h5' % today)

# Note, the following assumes that you have the graphviz graph library and the Python interface installed
#plot_model(model, to_file='\Images\LSTM_model_plot.png', show_shapes=True, show_layer_names=True)

# Plot training loss
plt.clf()
plt.figure(figsize=(10, 6))
epochs_range = range(1, 100+1)
LSTM_loss = history.history['loss']
plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\LSTM_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
plt.figure(figsize=(10, 6))
LSTM_acc = history.history['accuracy'] #acc
plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\LSTM_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate predictions
predicted = model.predict(SAE_train_loaded)
actuals = y_train_loaded.reshape(y_train_loaded.shape[0], 1)

# Plot prediction accuracy
plt.clf()
plt.figure(figsize=(10, 6))
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.scatter(actuals, predicted, s=1, alpha=0.3)
plt.title('Actual Values vs Predicted Values')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
plt.savefig('..\Images\LSTM_training_actuals_v_predicted_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Save predictions
np.savez_compressed(r'..\Data\LSTM_training_predictions.npz', predicted)

# Build combined array
ticker_date = train_dateticker[:,-1,-2:]
LSTM_predictions_analyze = np.column_stack((predicted, actuals, ticker_date))

# Save predictions for analysis
np.savetxt('..\Data\LSTM_training_predictions_analyze.csv', LSTM_predictions_analyze, delimiter=',', fmt='%s')

######################################
# EVALUATION
######################################

## Load model
LSTM_loaded = load_model('Models/LSTM_20210224.h5', compile = False)
LSTM_loaded.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
print(LSTM_loaded.summary())

## Validation data set
# Load data
x_validation_loaded = np.load(r'..\Data\SAE_train_encoded_5timestep_validation.npz')
print(x_validation_loaded.files)
x_validation_loaded = x_validation_loaded['arr_0']

y_validation_loaded = np.load(r'..\Data\y_validation_3d_5timestep.npz')
print(y_validation_loaded.files)
y_validation_loaded = y_validation_loaded['arr_0']
y_validation_loaded = y_validation_loaded[:,1] # 0 = 30D_RETURNS, 1 = 30D_SHARPE

# Calculate accuracy score
loss, acc = LSTM_loaded.evaluate(x_validation_loaded, y_validation_loaded)
print('Restored model accuracy: {:5.2f}%'.format(100 * acc))

# Generate predictions
LSTM_predictions_val = LSTM_loaded.predict(x_validation_loaded)
LSTM_actuals_val = y_validation_loaded.reshape(y_validation_loaded.shape[0], 1)

# Plot prediction accuracy
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(LSTM_actuals_val, LSTM_predictions_val, s=1, alpha=0.3)
plt.title('Actual Values vs Predicted Values')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
#plt.savefig('..\Images\LSTM_validation_actuals_v_predicted_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Save predictions
np.savez_compressed(r'..\Data\LSTM_validation_predictions.npz', LSTM_predictions_val)

# Load DATE and TICKER
validation_dateticker = np.load(r'..\Data\validation_dateticker_3d_5timestep.npz')
print(validation_dateticker.files)
validation_dateticker = validation_dateticker['arr_0']

# Build combined array
ticker_date_val = validation_dateticker[:,-1,-2:]
LSTM_predictions_val_analyze = np.column_stack((LSTM_predictions_val, LSTM_actuals_val, ticker_date_val))
np.savetxt('..\Data\LSTM_validation_predictions_analyze.csv', LSTM_predictions_val_analyze, delimiter=',', fmt='%s')

## Test data set
# Load data
x_test_loaded = np.load(r'..\Data\SAE_train_encoded_5timestep_test.npz')
print(x_test_loaded.files)
x_test_loaded = x_test_loaded['arr_0']

y_test_loaded = np.load(r'..\Data\y_test_3d_5timestep.npz')
print(y_test_loaded.files)
y_test_loaded = y_test_loaded['arr_0']
y_test_loaded = y_test_loaded[:,1] # 0 = 30D_RETURNS, 1 = 30D_SHARPE

# Calculate accuracy score
loss, acc = LSTM_loaded.evaluate(x_test_loaded, y_test_loaded)
print('Restored model accuracy: {:5.2f}%'.format(100 * acc))

# Generate predictions
LSTM_predictions_test = LSTM_loaded.predict(x_test_loaded)
LSTM_actuals_test = y_test_loaded.reshape(y_test_loaded.shape[0], 1)

# Plot prediction accuracy
plt.clf()
plt.figure(figsize=(10, 6))
plt.scatter(LSTM_actuals_test, LSTM_predictions_test, s=1, alpha=0.3)
plt.title('Actual Values vs Predicted Values')
plt.xlabel('Actuals')
plt.ylabel('Predicted')
#plt.savefig('..\Images\LSTM_test_actuals_v_predicted_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Save predictions
np.savez_compressed(r'..\Data\LSTM_test_predictions.npz', LSTM_predictions_test)

# Load DATE and TICKER
test_dateticker = np.load(r'..\Data\test_dateticker_3d_5timestep.npz')
print(test_dateticker.files)
test_dateticker = test_dateticker['arr_0']

# Build combined array
ticker_date_test = test_dateticker[:,-1,-2:]
LSTM_predictions_test_analyze = np.column_stack((LSTM_predictions_test, LSTM_actuals_test, ticker_date_test))
np.savetxt('..\Data\LSTM_test_predictions_analyze.csv', LSTM_predictions_test_analyze, delimiter=',', fmt='%s')

######################################
# INVERSE SCALING
######################################

## Scaler defined in DataPreparation.py
# Validation nverse transformation
y_validation = y_validation.iloc[:,1] # 0 = 30D_RETURNS, 1 = 30D_SHARPE
y_validation = y_validation.to_numpy()
y_validation = y_validation.reshape(y_validation.shape[0], 1)
y_val_scaled = scaler.fit_transform(y_validation)
LSTM_predictions_val_inverse = scaler.inverse_transform(LSTM_predictions_val)

# Validation nverse transformation
y_test = y_test.iloc[:,1] # 0 = 30D_RETURNS, 1 = 30D_SHARPE
y_test = y_test.to_numpy()
y_test = y_test.reshape(y_test.shape[0], 1)
y_test_scaled = scaler.fit_transform(y_test)
LSTM_predictions_test_inverse = scaler.inverse_transform(LSTM_predictions_test)

######################################
# ALTERNATE ATTEMPTS
######################################

### Fixed parameters
#batch_size = 32
#epochs = 10
#dropout = 0.2
#encoding_dim = 3
#num_recs = x_train_loaded.shape[0] #SAE_x.shape[0]
#n_steps = x_train_loaded.shape[1] #SAE_x.shape[1]
#input_dim = x_train_loaded.shape[2] #SAE_x.shape[2]
#
### https://datascienceplus.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python/
### LSTM 1
#model = Sequential()
#model.add(LSTM(units=batch_size, return_sequences=True, input_shape=(n_steps,input_dim)))
#model.add(Dropout(dropout))
#model.add(LSTM(units=batch_size))
#model.add(Dropout(dropout))
#model.add(Dense(units=1))
#
## Compile and fit model
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#history = model.fit(SAE_x, y_train_loaded, epochs=5, batch_size=batch_size)
#
## Print summary
#print(model.summary())
#
## Save model
#model.save('Models/LSTM_1_%s.h5' % today)
#
## Plot training loss
#plt.clf()
#epochs_range = range(1, epochs+1)
#LSTM_loss = history.history['loss']
#plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
#plt.title('Training loss for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('..\Images\LSTM1_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
## Plot training accuracy
#plt.clf()
#LSTM_acc = history.history['accuracy']
#plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
#plt.title('Training accuracy for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig('..\Images\LSTM1_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
#
### LSTM 2
#model = Sequential()
#model.add(LSTM(480, return_sequences=True, input_shape=(n_steps,input_dim)))
#model.add(LSTM(240))
#model.add(Dense(1))
#
## Compile and fit model
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#history = model.fit(SAE_x, y_train_loaded, epochs=epochs)
#
## Print summary
#print(model.summary())
#
## Save model
#model.save('Models/LSTM_2_%s.h5' % today)
#
## Plot training loss
#plt.clf()
#epochs_range = range(1, epochs+1)
#LSTM_loss = history.history['loss']
#plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
#plt.title('Training loss for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('..\Images\LSTM2_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
## Plot training accuracy
#plt.clf()
#LSTM_acc = history.history['accuracy']
#plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
#plt.title('Training accuracy for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig('..\Images\LSTM2_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
#
### LSTM 3
#model = Sequential()
#model.add(LSTM(480, return_sequences=True, input_shape=(n_steps,input_dim)))
#model.add(Dropout(dropout))
#model.add(LSTM(320, return_sequences=True))
#model.add(Dropout(dropout))
#model.add(LSTM(160, return_sequences=True))
#model.add(Dropout(dropout))
#model.add(LSTM(240))
#model.add(Dropout(dropout))
#model.add(Dense(1))
#
## Compile and fit model
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#history = model.fit(SAE_x, y_train_loaded, epochs=epochs)
#
## Print summary
#print(model.summary())
#
## Save model
#model.save('Models/LSTM_3_%s.h5' % today)
#
## Plot training loss
#plt.clf()
#epochs_range = range(1, epochs+1)
#LSTM_loss = history.history['loss']
#plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
#plt.title('Training loss for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('..\Images\LSTM3_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
## Plot training accuracy
#plt.clf()
#LSTM_acc = history.history['accuracy']
#plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
#plt.title('Training accuracy for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig('..\Images\LSTM3_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
#
### LSTM 4
#model = Sequential()
#model.add(LSTM(160, return_sequences=True, input_shape=(n_steps,60)))
#model.add(LSTM(96))
#model.add(Dense(1))
#
## Compile and fit model
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#history = model.fit(SAE_x, y_train_loaded, epochs=100)
#
## Print summary
#print(model.summary())
#
## Save model
#model.save('Models/LSTM_4_%s.h5' % today)
#
## Plot training loss
#plt.clf()
#epochs_range = range(1, 100+1)
#LSTM_loss = history.history['loss']
#plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
#plt.title('Training loss for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('..\Images\LSTM3_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
## Plot training accuracy
#plt.clf()
#LSTM_acc = history.history['accuracy']
#plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
#plt.title('Training accuracy for LSTM')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig('..\Images\LSTM3_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
#
## Make predictions
#predicted = model.predict(SAE_x)
#actuals = y_train_loaded.reshape(y_train_loaded.shape[0], 1)
#
## Plot prediction accuracy
#plt.clf()
#plt.title('Predicted vs Actuals')
#plt.scatter(actuals, predicted, s=1, alpha=0.3)
##plt.savefig('..\Images\SAE14_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
#plt.show()
