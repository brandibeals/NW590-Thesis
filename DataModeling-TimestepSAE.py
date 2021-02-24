# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 2021
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

## Set the number of timesteps
n_steps = 5

######################################
# LOAD DATA
######################################

# Load data
x_train_loaded = np.load(r'..\Data\x_train_3d_5timestep.npz')
print(x_train_loaded.files)
x_train_loaded = x_train_loaded['arr_0']

# Load DATE and TICKER
train_dateticker = np.load(r'..\Data\train_dateticker_3d_5timestep.npz')
print(train_dateticker.files)
train_dateticker = train_dateticker['arr_0']

######################################
# AUTOENCODER
######################################

## Fixed parameters
n_features = 24
batch_size = 32
epochs = 100
obs = x_train_loaded.shape[0]
n_steps = x_train_loaded.shape[1]
input_dim = x_train_loaded.shape[2]
encoding_dim = n_steps*n_features

# Encoder
encoder = Sequential(name='Encoder')
encoder.add(LSTM(encoding_dim, activation='relu', input_shape=(n_steps, input_dim), name='encoderlayer1'))

# Decoder
decoder = Sequential(name='Decoder')
decoder.add(RepeatVector(n_steps))
decoder.add(LSTM(encoding_dim, return_sequences=True, name='decoderlayer1'))
decoder.add(TimeDistributed(Dense(input_dim)))

# Compile and fit autoencoder
SAE = Sequential([encoder, decoder], name='SAE')
SAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
encoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE_history = SAE.fit(x_train_loaded, x_train_loaded, epochs=epochs, batch_size=batch_size)

# Save model
SAE.save('Models/SAE_5timestep_%s.h5' % today)
encoder.save('Models/Encoder_5timestep_%s.h5' % today)

# Note, the following assumes that you have the graphviz graph library and the Python interface installed
#plot_model(SAE, to_file='\Images\SAE_model_plot.png', show_shapes=True, show_layer_names=True)

# Plot training loss
plt.clf()
plt.figure(figsize=(10, 6))
epochs_range = range(1, epochs+1)
SAE_loss = SAE_history.history['loss']
plt.plot(epochs_range, SAE_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE_training_loss_5timestep_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
plt.figure(figsize=(10, 6))
SAE_acc = SAE_history.history['accuracy'] #acc
plt.plot(epochs_range, SAE_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE_training_accuracy_5timestep_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_predictions = encoder.predict(x_train_loaded)
SAE_predictions = SAE_predictions.reshape((obs, n_steps, n_features))

# Save predictions
np.savez_compressed(r'..\Data\SAE_training_encoded_5timestep.npz', SAE_predictions)

# Build combined array
ticker_date = train_dateticker[:,-1,-2:]
SAE_predictions_analyze = SAE_predictions[:,-1,:]
SAE_predictions_analyze = np.column_stack((SAE_predictions_analyze, ticker_date))

# Save predictions for analysis
np.savetxt('..\Data\SAE_training_encoded_5timestep_analyze.csv', SAE_predictions_analyze, delimiter=',', fmt='%s')

######################################
# EVALUATION
######################################

## Validation data set
# Load data
x_validation_loaded = np.load(r'..\Data\x_validation_3d_5timestep.npz')
print(x_validation_loaded.files)
x_validation_loaded = x_validation_loaded['arr_0']

# Calculate accuracy score


# Generate encoded predictions
SAE_predictions_val = encoder.predict(x_validation_loaded)
SAE_predictions_val = SAE_predictions_val.reshape((obs, n_steps, n_features))

# Save predictions
np.savez_compressed(r'..\Data\SAE_validation_encoded_5timestep.npz', SAE_predictions_val)

## Validation data set
# Load data
x_test_loaded = np.load(r'..\Data\x_test_3d_5timestep.npz')
print(x_test_loaded.files)
x_test_loaded = x_test_loaded['arr_0']

# Calculate accuracy score


# Generate encoded predictions
SAE_predictions_test = encoder.predict(x_test_loaded)
SAE_predictions_test = SAE_predictions_test.reshape((obs, n_steps, n_features))

# Save predictions
np.savez_compressed(r'..\Data\SAE_test_encoded_5timestep.npz', SAE_predictions_test)

######################################
# ALTERNATE ATTEMPTS
######################################

## AUTOENCODER 1 - 40% ACCURACY
## https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
# Encoder
encoder = Sequential(name='SAE_encoder')
encoder.add(Flatten(input_shape=(n_steps, input_dim)))
encoder.add(Dense(600, activation='relu', name='encoderlayer1'))
encoder.add(Dense(300, activation='relu', name='encoderlayer2'))
encoder.add(Dense(180, activation='relu', name='encoderlayer3'))
encoder.add(Dense(60, activation='relu', name='encoderlayer4'))
encoder.add(Dense(10, activation='relu', name='encoderlayer5'))

# Decoder
decoder = Sequential(name='SAE_decoder')
decoder.add(Dense(60, activation='relu', name='decoderlayer1', input_shape=(10,)))
decoder.add(Dense(180, activation='relu', name='decoderlayer2'))
decoder.add(Dense(300, activation='relu', name='decoderlayer3'))
decoder.add(Dense(600, activation='relu', name='decoderlayer4'))
decoder.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer5'))
decoder.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder = Sequential([encoder, decoder], name='SAE')
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE_history = autoencoder.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder.summary())

# Save model
autoencoder.save('Models/SAE_1_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_1_loss = SAE_history.history['loss']
plt.plot(epochs_range, SAE_1_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE1_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_1_acc = SAE_history.history['accuracy']
plt.plot(epochs_range, SAE_1_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE1_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_1 = encoder.predict(x_train_loaded)
SAE_1 = SAE_1.reshape(SAE_1.shape[0], n_steps, SAE_1.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('Two Dimensions of Encoded Data')
plt.scatter(SAE_1[:,:,0], SAE_1[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE1_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 2 - 60% ACCURACY
# Encoder
encoder2 = Sequential(name='SAE2_encoder')
encoder2.add(Flatten(input_shape=(n_steps, input_dim)))
encoder2.add(Dense(600, activation='relu', name='encoderlayer1'))
encoder2.add(Dense(300, activation='relu', name='encoderlayer2'))
encoder2.add(Dense(180, activation='relu', name='encoderlayer3'))
encoder2.add(Dense(60, activation='relu', name='encoderlayer4'))

# Decoder
decoder2 = Sequential(name='SAE2_decoder')
decoder2.add(Dense(180, activation='relu', name='decoderlayer1', input_shape=(60,)))
decoder2.add(Dense(300, activation='relu', name='decoderlayer2'))
decoder2.add(Dense(600, activation='relu', name='decoderlayer3'))
decoder2.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder2.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder2 = Sequential([encoder2, decoder2], name='SAE2')
autoencoder2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE2_history = autoencoder2.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder2.summary())

# Save model
autoencoder2.save('Models/SAE_2_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_2_loss = SAE2_history.history['loss']
plt.plot(epochs_range, SAE_2_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE2_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_2_acc = SAE2_history.history['accuracy']
plt.plot(epochs_range, SAE_2_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE2_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_2 = encoder2.predict(x_train_loaded)
SAE_2 = SAE_2.reshape(SAE_2.shape[0], n_steps, SAE_2.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_2[:,:,0], SAE_2[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE2_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 3 - 62% ACCURACY
# Encoder
encoder3 = Sequential(name='SAE3_encoder')
encoder3.add(Flatten(input_shape=(n_steps, input_dim)))
encoder3.add(Dense(780, activation='relu', name='encoderlayer1'))
encoder3.add(Dense(540, activation='relu', name='encoderlayer2'))
encoder3.add(Dense(300, activation='relu', name='encoderlayer3'))
encoder3.add(Dense(60, activation='relu', name='encoderlayer4'))

# Decoder
decoder3 = Sequential(name='SAE3_decoder')
decoder3.add(Dense(300, activation='relu', name='decoderlayer1', input_shape=(60,)))
decoder3.add(Dense(540, activation='relu', name='decoderlayer2'))
decoder3.add(Dense(780, activation='relu', name='decoderlayer3'))
decoder3.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder3.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder3 = Sequential([encoder3, decoder3], name='SAE3')
autoencoder3.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE3_history = autoencoder3.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder3.summary())

# Save model
autoencoder3.save('Models/SAE_3_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_3_loss = SAE3_history.history['loss']
plt.plot(epochs_range, SAE_3_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE3_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_3_acc = SAE3_history.history['accuracy']
plt.plot(epochs_range, SAE_3_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE3_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_3 = encoder3.predict(x_train_loaded)
SAE_3 = SAE_3.reshape(SAE_3.shape[0], n_steps, SAE_3.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_3[:,:,0], SAE_3[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE3_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 4 - 50% ACCURACY
# Encoder
encoder4 = Sequential(name='SAE4_encoder')
encoder4.add(Flatten(input_shape=(n_steps, input_dim)))
encoder4.add(Dense(900, activation='relu', name='encoderlayer1'))
encoder4.add(Dense(780, activation='relu', name='encoderlayer2'))
encoder4.add(Dense(660, activation='relu', name='encoderlayer3'))
encoder4.add(Dense(540, activation='relu', name='encoderlayer4'))
encoder4.add(Dense(420, activation='relu', name='encoderlayer5'))
encoder4.add(Dense(300, activation='relu', name='encoderlayer6'))
encoder4.add(Dense(180, activation='relu', name='encoderlayer7'))
encoder4.add(Dense(60, activation='relu', name='encoderlayer8'))

# Decoder
decoder4 = Sequential(name='SAE4_decoder')
decoder4.add(Dense(180, activation='relu', name='decoderlayer1', input_shape=(60,)))
decoder4.add(Dense(300, activation='relu', name='decoderlayer2'))
decoder4.add(Dense(420, activation='relu', name='decoderlayer3'))
decoder4.add(Dense(540, activation='relu', name='decoderlayer4'))
decoder4.add(Dense(660, activation='relu', name='decoderlayer5'))
decoder4.add(Dense(780, activation='relu', name='decoderlayer6'))
decoder4.add(Dense(900, activation='relu', name='decoderlayer7'))
decoder4.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer8'))
decoder4.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder4 = Sequential([encoder4, decoder4], name='SAE4')
autoencoder4.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE4_history = autoencoder4.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder4.summary())

# Save model
autoencoder4.save('Models/SAE_4_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_4_loss = SAE4_history.history['loss']
plt.plot(epochs_range, SAE_4_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE4_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_4_acc = SAE4_history.history['accuracy']
plt.plot(epochs_range, SAE_4_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE4_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_4 = encoder4.predict(x_train_loaded)
SAE_4 = SAE_4.reshape(SAE_4.shape[0], n_steps, SAE_4.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_4[:,:,0], SAE_4[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE4_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 5 - 58% ACCURACY
# Encoder
encoder5 = Sequential(name='SAE5_encoder')
encoder5.add(Flatten(input_shape=(n_steps, input_dim)))
encoder5.add(Dense(780, activation='relu', name='encoderlayer1'))
encoder5.add(Dense(540, activation='relu', name='encoderlayer2'))
encoder5.add(Dense(300, activation='relu', name='encoderlayer3'))
encoder5.add(Dense(180, activation='relu', name='encoderlayer4'))

# Decoder
decoder5 = Sequential(name='SAE5_decoder')
decoder5.add(Dense(300, activation='relu', name='decoderlayer1', input_shape=(180,)))
decoder5.add(Dense(540, activation='relu', name='decoderlayer2'))
decoder5.add(Dense(780, activation='relu', name='decoderlayer3'))
decoder5.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder5.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder5 = Sequential([encoder5, decoder5], name='SAE5')
autoencoder5.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE5_history = autoencoder5.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder5.summary())

# Save model
autoencoder5.save('Models/SAE_5_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_5_loss = SAE5_history.history['loss']
plt.plot(epochs_range, SAE_5_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE5_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_5_acc = SAE5_history.history['accuracy']
plt.plot(epochs_range, SAE_5_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE5_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_5 = encoder5.predict(x_train_loaded)
SAE_5 = SAE_5.reshape(SAE_5.shape[0], n_steps, SAE_5.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_5[:,:,0], SAE_5[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE5_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 6 - 43% ACCURACY, BUT SMOOTHER
# Encoder
encoder6 = Sequential(name='SAE6_encoder')
encoder6.add(Flatten(input_shape=(n_steps, input_dim)))
encoder6.add(Dense(780, activation='relu', name='encoderlayer1'))
encoder6.add(Dense(540, activation='relu', name='encoderlayer2'))
encoder6.add(Dense(300, activation='relu', name='encoderlayer3'))
encoder6.add(Dense(60, activation='relu', name='encoderlayer4'))

# Decoder
decoder6 = Sequential(name='SAE6_decoder')
decoder6.add(Dense(300, activation='relu', name='decoderlayer1', input_shape=(60,)))
decoder6.add(Dense(540, activation='relu', name='decoderlayer2'))
decoder6.add(Dense(780, activation='relu', name='decoderlayer3'))
decoder6.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder6.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder6 = Sequential([encoder6, decoder6], name='SAE6')
autoencoder6.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
SAE6_history = autoencoder6.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder6.summary())

# Save model
autoencoder6.save('Models/SAE_6_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_6_loss = SAE6_history.history['loss']
plt.plot(epochs_range, SAE_6_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE6_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_6_acc = SAE6_history.history['accuracy']
plt.plot(epochs_range, SAE_6_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE6_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_6 = encoder6.predict(x_train_loaded)
SAE_6 = SAE_6.reshape(SAE_6.shape[0], n_steps, SAE_6.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_6[:,:,0], SAE_6[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE6_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 7 - UNKNOWN
# Encoder
encoder7 = Sequential(name='SAE7_encoder')
encoder7.add(Flatten(input_shape=(n_steps, input_dim)))
encoder7.add(Dense(780, activation='relu', name='encoderlayer1'))
encoder7.add(Dense(540, activation='relu', name='encoderlayer2'))
encoder7.add(Dense(300, activation='relu', name='encoderlayer3'))
encoder7.add(Dense(60, activation='relu', name='encoderlayer4'))

# Decoder
decoder7 = Sequential(name='SAE7_decoder')
decoder7.add(Dense(300, activation='relu', name='decoderlayer1', input_shape=(60,)))
decoder7.add(Dense(540, activation='relu', name='decoderlayer2'))
decoder7.add(Dense(780, activation='relu', name='decoderlayer3'))
decoder7.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder7.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder7 = Sequential([encoder7, decoder7], name='SAE7')
autoencoder7.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE7_history = autoencoder7.fit(x_train_loaded, x_train_loaded, epochs=1000)

# Print summary
print(autoencoder7.summary())

# Save model
autoencoder7.save('Models/SAE_7_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_7_loss = SAE7_history.history['loss']
plt.plot(epochs_range, SAE_7_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE7_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_7_acc = SAE7_history.history['accuracy']
plt.plot(epochs_range, SAE_7_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE7_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_7 = encoder7.predict(x_train_loaded)
SAE_7 = SAE_7.reshape(SAE_7.shape[0], n_steps, SAE_7.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_7[:,:,0], SAE_7[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE7_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 13 - 62% ACCURACY
# Encoder
encoder3 = Sequential(name='SAE3_encoder')
encoder3.add(Flatten(input_shape=(n_steps, input_dim)))
encoder3.add(Dense(780, activation='relu', name='encoderlayer1'))
#encoder3.add(Dense(540, activation='relu', name='encoderlayer2'))
encoder3.add(Dense(300, activation='relu', name='encoderlayer3'))
encoder3.add(Dense(60, activation='relu', name='encoderlayer4'))

# Decoder
decoder3 = Sequential(name='SAE3_decoder')
decoder3.add(Dense(300, activation='relu', name='decoderlayer1', input_shape=(60,)))
#decoder3.add(Dense(540, activation='relu', name='decoderlayer2'))
decoder3.add(Dense(780, activation='relu', name='decoderlayer3'))
decoder3.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer4'))
decoder3.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder3 = Sequential([encoder3, decoder3], name='SAE3')
autoencoder3.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE3_history = autoencoder3.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder3.summary())

# Save model
autoencoder3.save('Models/SAE_13_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_3_loss = SAE3_history.history['loss']
plt.plot(epochs_range, SAE_3_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE13_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_3_acc = SAE3_history.history['accuracy']
plt.plot(epochs_range, SAE_3_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE13_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_3 = encoder3.predict(x_train_loaded)
SAE_3 = SAE_3.reshape(SAE_3.shape[0], n_steps, SAE_3.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_3[:,:,0], SAE_3[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE13_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 14 - 65% ACCURACY, ROLLERCOASTER
# Encoder
encoder14 = Sequential(name='SAE3_encoder')
encoder14.add(Flatten(input_shape=(n_steps, input_dim)))
encoder14.add(Dense(780, activation='relu', name='encoderlayer1'))
encoder14.add(Dense(300, activation='relu', name='encoderlayer2'))

# Decoder
decoder14 = Sequential(name='SAE3_decoder')
decoder14.add(Dense(780, activation='relu', name='decoderlayer1', input_shape=(300,)))
decoder14.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer2'))
decoder14.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
autoencoder14 = Sequential([encoder14, decoder14], name='SAE3')
autoencoder14.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE14_history = autoencoder14.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder14.summary())

# Save model
autoencoder14.save('Models/SAE_14_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_14_loss = SAE14_history.history['loss']
plt.plot(epochs_range, SAE_14_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE14_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_14_acc = SAE14_history.history['accuracy']
plt.plot(epochs_range, SAE_14_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE14_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_14 = encoder14.predict(x_train_loaded)
SAE_14 = SAE_14.reshape(SAE_14.shape[0], n_steps, SAE_14.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_14[:,:,0], SAE_14[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE14_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 15 - 70% ACCURACY
# Encoder/Decoder
encoder15 = Sequential(name='SAE15_encoder')
encoder15.add(LSTM(300, activation='relu', input_shape=(n_steps, input_dim), name='encoderlayer1'))
encoder15.add(RepeatVector(n_steps))
encoder15.add(LSTM(300, return_sequences=True, name='decoderlayer1'))
encoder15.add(TimeDistributed(Dense(204)))

# Compile and fit autoencoder
autoencoder15 = Sequential(encoder15, name='SAE15')
autoencoder15.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE15_history = autoencoder15.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder15.summary())

# Save model
autoencoder15.save('Models/SAE_15_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_15_loss = SAE15_history.history['loss']
plt.plot(epochs_range, SAE_15_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE15_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_15_acc = SAE15_history.history['accuracy']
plt.plot(epochs_range, SAE_15_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE15_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
model = Model(inputs=autoencoder15.inputs, outputs=autoencoder15.layers[0].output)
SAE_15 = model.predict(x_train_loaded)
SAE_15 = SAE_15.reshape(SAE_15.shape[0], n_steps, SAE_15.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_15[:,:,0], SAE_15[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE15_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()


## AUTOENCODER 16 - UNKNOWN
# Encoder
encoder16 = Sequential(name='SAE16_encoder')
encoder16.add(LSTM(300, activation='relu', input_shape=(n_steps, input_dim), name='encoderlayer1'))

# Decoder
decoder16 = Sequential(name='SAE16_decoder')
decoder16.add(RepeatVector(n_steps))
decoder16.add(LSTM(300, return_sequences=True, name='decoderlayer1'))
decoder16.add(TimeDistributed(Dense(204)))

# Compile and fit autoencoder
autoencoder16 = Sequential([encoder16, decoder16], name='SAE16')
autoencoder16.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE16_history = autoencoder16.fit(x_train_loaded, x_train_loaded, epochs=epochs)

# Print summary
print(autoencoder16.summary())

# Save model
autoencoder16.save('Models/SAE_16_%s.h5' % today)
encoder16.save('Models/Encoder_16_%s.h5' % today)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_15_loss = SAE15_history.history['loss']
plt.plot(epochs_range, SAE_15_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE15_training_loss_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_15_acc = SAE15_history.history['accuracy']
plt.plot(epochs_range, SAE_15_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE15_training_accuracy_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
model = Model(inputs=autoencoder15.inputs, outputs=autoencoder15.layers[0].output)
SAE_15 = model.predict(x_train_loaded)
SAE_15 = SAE_15.reshape(SAE_15.shape[0], n_steps, SAE_15.shape[1] // n_steps)

# Generate plot showing reduced dimensions
plt.clf()
plt.title('First Two Dimensions of Encoded Data')
plt.scatter(SAE_15[:,:,0], SAE_15[:,:,1], s=1, alpha=0.3)
plt.savefig('..\Images\SAE14_encoding_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()
