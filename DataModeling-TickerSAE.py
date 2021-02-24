# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 2021
Author: Brandi Beals
Description: Thesis Dimensionality Reduction
"""

######################################
# IMPORT PACKAGES
######################################

import os
from datetime import datetime
import numpy as np
from tensorflow import random
from tensorflow.keras.models import Sequential,Model
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
# LOAD DATA
######################################

# Load data
x_train_loaded = np.load(r'..\Data\x_train_3d_ticker.npz')
print(x_train_loaded.files)
x_train_loaded = x_train_loaded['arr_0']

# Load DATE and TICKER
train_dateticker = np.load(r'..\Data\train_dateticker_3d_ticker.npz')
print(train_dateticker.files)
train_dateticker = train_dateticker['arr_0']

######################################
# HYPERPARAMETERS
######################################

## Fixed parameters
n_features = 24
batch_size = 1
epochs = 100
obs = x_train_loaded.shape[0]
n_steps = x_train_loaded.shape[1]
input_dim = x_train_loaded.shape[2]
encoding_dim = n_steps*n_features

######################################
# ALTERNATE ATTEMPTS
######################################

## AUTOENCODER 8 - 31% ACCURACY
## https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
# Encoder
encoder = Sequential(name='SAE_encoder')
encoder.add(Flatten(input_shape=(n_steps, input_dim)))
encoder.add(Dense(2560, activation='relu', name='encoderlayer1'))
encoder.add(Dense(2024, activation='relu', name='encoderlayer2'))
encoder.add(Dense(1488, activation='relu', name='encoderlayer3'))
encoder.add(Dense(952, activation='relu', name='encoderlayer4'))
encoder.add(Dense(416, activation='relu', name='encoderlayer5'))

# Decoder
decoder = Sequential(name='SAE_decoder')
decoder.add(Dense(952, activation='relu', name='decoderlayer1', input_shape=(416,)))
decoder.add(Dense(1488, activation='relu', name='decoderlayer2'))
decoder.add(Dense(2024, activation='relu', name='decoderlayer3'))
decoder.add(Dense(2560, activation='relu', name='decoderlayer4'))
decoder.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer5'))
decoder.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder = Sequential([encoder, decoder], name='SAE')
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE_history = autoencoder.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder.summary())

# Save model
autoencoder.save('Models/SAE_8_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_1_loss = SAE_history.history['loss']
plt.plot(epochs_range, SAE_1_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE8_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_1_acc = SAE_history.history['accuracy']
plt.plot(epochs_range, SAE_1_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE8_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_1 = encoder.predict(x_train_loaded)
SAE_1 = SAE_1.reshape(SAE_1.shape[0], n_steps, SAE_1.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('Two Dimensions of Encoded Data')
plt.scatter(SAE_1[:,:,0], SAE_1[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE8_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 9 - 35% ACCURACY
# Encoder
encoder2 = Sequential(name='SAE2_encoder')
encoder2.add(Flatten(input_shape=(n_steps, input_dim)))
encoder2.add(Dense(416, activation='relu', name='encoderlayer1'))

# Decoder
decoder2 = Sequential(name='SAE2_decoder')
decoder2.add(Dense(n_steps*input_dim, activation='relu', name='decoderlayer1', input_shape=(416,)))
decoder2.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder2 = Sequential([encoder2, decoder2], name='SAE2')
autoencoder2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE2_history = autoencoder2.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder2.summary())

# Save model
autoencoder2.save('Models/SAE_9_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_2_loss = SAE2_history.history['loss']
plt.plot(epochs_range, SAE_2_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE9_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_2_acc = SAE2_history.history['accuracy']
plt.plot(epochs_range, SAE_2_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE9_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_2 = encoder2.predict(x_train_loaded)
SAE_2 = SAE_2.reshape(SAE_2.shape[0], n_steps, SAE_2.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_2[:,:,0], SAE_2[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE9_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 10 - 5% ACCURACY
# Encoder
encoder3 = Sequential(name='SAE3_encoder')
encoder3.add(Flatten(input_shape=(n_steps, input_dim)))
encoder3.add(Dense(416, activation='relu', name='encoderlayer1'))

# Decoder
decoder3 = Sequential(name='SAE3_decoder')
decoder3.add(Dense(n_steps*input_dim, activation='relu', name='decoderlayer1', input_shape=(416,)))
decoder3.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder3 = Sequential([encoder3, decoder3], name='SAE3')
autoencoder3.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE3_history = autoencoder3.fit(x_train_loaded, x_train_loaded, epochs=epochs, batch_size=batch_size)

# Print summary
print(autoencoder3.summary())

# Save model
autoencoder3.save('Models/SAE_10_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_3_loss = SAE3_history.history['loss']
plt.plot(epochs_range, SAE_3_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE10_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_3_acc = SAE3_history.history['accuracy']
plt.plot(epochs_range, SAE_3_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE10_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_3 = encoder3.predict(x_train_loaded)
SAE_3 = SAE_3.reshape(SAE_3.shape[0], n_steps, SAE_3.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_3[:,:,0], SAE_3[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE10_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 11 - 36% ACCURACY
# Encoder
encoder4 = Sequential(name='SAE4_encoder')
encoder4.add(Flatten(input_shape=(n_steps, input_dim)))
encoder4.add(Dense(4992, activation='relu', name='encoderlayer1'))
encoder4.add(Dense(2496, activation='relu', name='encoderlayer8'))

# Decoder
decoder4 = Sequential(name='SAE4_decoder')
decoder4.add(Dense(4992, activation='relu', name='decoderlayer1', input_shape=(2496,)))
decoder4.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer8'))
decoder4.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder4 = Sequential([encoder4, decoder4], name='SAE4')
autoencoder4.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE4_history = autoencoder4.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder4.summary())

# Save model
autoencoder4.save('Models/SAE_11_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_4_loss = SAE4_history.history['loss']
plt.plot(epochs_range, SAE_4_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE11_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_4_acc = SAE4_history.history['accuracy']
plt.plot(epochs_range, SAE_4_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE11_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_4 = encoder4.predict(x_train_loaded)
SAE_4 = SAE_4.reshape(SAE_4.shape[0], n_steps, SAE_4.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_4[:,:,0], SAE_4[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE11_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 12 - 4% ACCURACY
# Encoder/Decoder
encoder5 = Sequential(name='SAE5_encoder')
encoder5.add(LSTM(102, input_shape=(n_steps, input_dim), name='encoderlayer1'))
encoder5.add(RepeatVector(n_steps))
encoder5.add(LSTM(102, return_sequences=True, name='decoderlayer1'))
encoder5.add(TimeDistributed(Dense(input_dim, activation='softmax')))

# Compile and fit autoencoder
encoder5.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE5_history = encoder5.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(encoder5.summary())

# Save model
encoder5.save('Models/SAE_12_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_5_loss = SAE5_history.history['loss']
plt.plot(epochs_range, SAE_5_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE12_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_5_acc = SAE5_history.history['accuracy']
plt.plot(epochs_range, SAE_5_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE12_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_5 = encoder5.predict(x_train_loaded)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_5[:,:,0], SAE_5[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE12_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()
