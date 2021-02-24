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
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input,Dense,Flatten,Reshape,LSTM,Dropout,RepeatVector,TimeDistributed
import matplotlib.pyplot as plt

######################################
# DEFIINITIONS
######################################

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
# LOAD DATA AND MODEL
######################################

# Load data
SAE_train_loaded = np.load(r'..\Data\SAE_training_encoded_5timestep.npz')
print(SAE_train_loaded.files)
SAE_train_loaded = SAE_train_loaded['arr_0']




######################################
# ALTERNATE ATTEMPTS
######################################

# Load data
x_train_loaded = np.load(r'..\Data\x_train_3d_matrix.npz')
#print(x_train_loaded.files)
x_train_loaded = x_train_loaded['arr_0']

y_train_loaded = np.load(r'..\Data\y_train_3d_matrix.npz')
#print(y_train_loaded.files)
y_train_loaded = y_train_loaded['arr_0']

# Load SAE model weights
SAE = load_model('Models/Encoder_16_20210213.h5')  #SAE_14_20210131.h5')
print(SAE.summary())

# Calculate SAE accuracy
loss, acc = SAE.evaluate(x_train_loaded, x_train_loaded)
print('Restored model accuracy: {:5.2f}%'.format(100 * acc))

# Calculate SAE predictions 
SAE_x = SAE.predict(x_train_loaded)
SAE_x = SAE_x.reshape(SAE_x.shape[0], n_steps, SAE_x.shape[1] // n_steps)

######################################
# LSTM
######################################

## Fixed parameters
batch_size = 32
epochs = 10
dropout = 0.2
encoding_dim = 3
num_recs = x_train_loaded.shape[0] #SAE_x.shape[0]
n_steps = x_train_loaded.shape[1] #SAE_x.shape[1]
input_dim = x_train_loaded.shape[2] #SAE_x.shape[2]

## https://datascienceplus.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python/
# LSTM
model = Sequential()
model.add(LSTM(units=batch_size, return_sequences=True, input_shape=(n_steps,input_dim)))
model.add(Dropout(dropout))
model.add(LSTM(units=batch_size))
model.add(Dropout(dropout))
model.add(Dense(units=1))

# Compile and fit model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(SAE_x, y_train_loaded, epochs=5, batch_size=batch_size)

# Print summary
print(model.summary())

# Save model
model.save('Models/LSTM_1_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
LSTM_loss = history.history['loss']
plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\LSTM1_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
LSTM_acc = history.history['accuracy']
plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\LSTM1_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


# Vanilla LSTM
model = Sequential()
model.add(LSTM(480, return_sequences=True, input_shape=(n_steps,input_dim)))
model.add(LSTM(240))
model.add(Dense(1))

# Compile and fit model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(SAE_x, y_train_loaded, epochs=epochs)

# Print summary
print(model.summary())

# Save model
model.save('Models/LSTM_2_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
LSTM_loss = history.history['loss']
plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\LSTM2_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
LSTM_acc = history.history['accuracy']
plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\LSTM2_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


# LSTM 3
model = Sequential()
model.add(LSTM(480, return_sequences=True, input_shape=(n_steps,input_dim)))
model.add(Dropout(dropout))
model.add(LSTM(320, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(160, return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(240))
model.add(Dropout(dropout))
model.add(Dense(1))

# Compile and fit model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(SAE_x, y_train_loaded, epochs=epochs)

# Print summary
print(model.summary())

# Save model
model.save('Models/LSTM_3_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
LSTM_loss = history.history['loss']
plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\LSTM3_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
LSTM_acc = history.history['accuracy']
plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\LSTM3_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


# LSTM 4
model = Sequential()
model.add(LSTM(160, return_sequences=True, input_shape=(n_steps,60)))
model.add(LSTM(96))
model.add(Dense(1))

# Compile and fit model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(SAE_x, y_train_loaded, epochs=100)

# Print summary
print(model.summary())

# Save model
model.save('Models/LSTM_4_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, 100+1)
LSTM_loss = history.history['loss']
plt.plot(epochs_range, LSTM_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\LSTM3_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
LSTM_acc = history.history['accuracy']
plt.plot(epochs_range, LSTM_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for LSTM')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\LSTM3_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Make predictions
predicted = model.predict(SAE_x)
actuals = y_train_loaded.reshape(y_train_loaded.shape[0], 1)

# Plot prediction accuracy
plt.clf()
plt.title('Predicted vs Actuals')
plt.scatter(actuals, predicted, s=1, alpha=0.3)
#plt.savefig('..\Images\SAE14_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()
