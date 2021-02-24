# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:30:17 2021

@author: bbeals
"""

######################################
# DIMENSIONALITY REDUCTION: OPTION 1
######################################

## Fixed parameters
n_features = 24
batch_size = 32
epochs = 10
num_recs = x_train_reshape_drop.shape[0]
n_steps = x_train_reshape_drop.shape[1]
input_dim = x_train_reshape_drop.shape[2]
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
SAE_history = SAE.fit(x_train_reshape_drop, x_train_reshape_drop, epochs=epochs, batch_size=batch_size)

# Save model
SAE.save('Models/SAE_1timestep_%s.h5' % today)
# Note, the following assumes that you have the graphviz graph library and the Python interface installed
#plot_model(SAE, to_file='\Images\SAE_model_plot.png', show_shapes=True, show_layer_names=True)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_loss = SAE_history.history['loss']
plt.plot(epochs_range, SAE_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE_training_loss_1timestep_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_acc = SAE_history.history['acc'] #accuracy
plt.plot(epochs_range, SAE_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE15_training_accuracy_1timestep_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_predictions = encoder.predict(x_train_reshape_drop)

# Build combined array
ticker_date = x_train_reshape[:,:,-2:]
ticker_date = ticker_date.reshape((ticker_date.shape[0], ticker_date.shape[2]))
SAE_predictions = np.column_stack((SAE_predictions, ticker_date))

# Save predictions
np.savetxt('..\Data\SAE_training_encoded_1timestep.csv', SAE_predictions, delimiter=',', fmt='%s')

# Reverse scaling
#SAE_predictions_inverse = scaler.inverse_transform(SAE_predictions)
#rmse = np.sqrt(np.mean(((SAE_predictions_inverse - y_test)**2)))

# Evaluate results of validation data
SAE.evaluate(x_validation_reshape_drop, x_validation_reshape_drop)



######################################
# DIMENSIONALITY REDUCTION: OPTION 2
######################################

## Fixed parameters
n_features = 24
batch_size = 32
epochs = 10
num_recs = x_train_array_drop.shape[0]
n_steps = x_train_array_drop.shape[1]
input_dim = x_train_array_drop.shape[2]
encoding_dim = n_steps*n_features

# Encoder
encoder = Sequential(name='Encoder')
encoder.add(Flatten(input_shape=(n_steps, input_dim)))
encoder.add(Dense(9984, activation='relu', name='encoderlayer1'))
encoder.add(Dense(encoding_dim, activation='relu', name='encoderlayer2'))

# Decoder
decoder = Sequential(name='Decoder')
decoder.add(Dense(9984, activation='relu', name='decoderlayer1', input_shape=(encoding_dim,)))
decoder.add(Dense(n_steps*input_dim, activation="relu", name='decoderlayer2'))
decoder.add(Reshape((n_steps, input_dim)))

# Compile and fit autoencoder
SAE = Sequential([encoder, decoder], name='SAE')
SAE.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
SAE_history = SAE.fit(x_train_array_drop, x_train_array_drop, epochs=epochs, batch_size=batch_size)

# Save model
SAE.save('Models/SAE_ticker_%s.h5' % today)
# Note, the following assumes that you have the graphviz graph library and the Python interface installed
#plot_model(SAE, to_file='\Images\SAE_model_plot.png', show_shapes=True, show_layer_names=True)

# Plot training loss
plt.clf()
epochs_range = range(1, epochs+1)
SAE_loss = SAE_history.history['loss']
plt.plot(epochs_range, SAE_loss, label='Training loss', color='tab:blue')
plt.title('Training loss for SAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('..\Images\SAE_training_loss_ticker_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Plot training accuracy
plt.clf()
SAE_acc = SAE_history.history['acc'] #accuracy
plt.plot(epochs_range, SAE_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE15_training_accuracy_ticker_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_predictions = encoder.predict(x_train_array_drop)

# Build combined array
ticker_date = x_train_array[:,-1,-2:]
SAE_predictions = SAE_predictions.reshape((num_recs, n_steps, n_features))
SAE_predictions = SAE_predictions[:,-1,:]
SAE_predictions = np.column_stack((SAE_predictions, ticker_date))

# Save predictions
np.savetxt('..\Data\SAE_training_encoded_ticker.csv', SAE_predictions, delimiter=',', fmt='%s')

# Reverse scaling
#SAE_predictions_inverse = scaler.inverse_transform(SAE_predictions)
#rmse = np.sqrt(np.mean(((SAE_predictions_inverse - y_test)**2)))

# Evaluate results of validation data
SAE.evaluate(x_validation_array_drop, x_validation_array_drop)



######################################
# DIMENSIONALITY REDUCTION: OPTION 3
######################################

## Fixed parameters
n_features = 24
batch_size = 1
epochs = 10
num_recs = x_train_matrix_drop.shape[0]
n_steps = x_train_matrix_drop.shape[1]
input_dim = x_train_matrix_drop.shape[2]
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
SAE_history = SAE.fit(x_train_matrix_drop, x_train_matrix_drop, epochs=epochs, batch_size=batch_size)

# Save model
SAE.save('Models/SAE_5timestep_%s.h5' % today)
# Note, the following assumes that you have the graphviz graph library and the Python interface installed
#plot_model(SAE, to_file='\Images\SAE_model_plot.png', show_shapes=True, show_layer_names=True)

# Plot training loss
plt.clf()
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
SAE_acc = SAE_history.history['acc'] #accuracy
plt.plot(epochs_range, SAE_acc, label='Training acc', color='tab:blue')
plt.title('Training accuracy for SAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('..\Images\SAE15_training_accuracy_5timestep_%s.png' % today, bbox_inches='tight', dpi=300)
plt.show()

# Generate encoded predictions
SAE_predictions = encoder.predict(x_train_matrix_drop)

# Build combined array
ticker_date = x_train_matrix[:,-1,-2:]
SAE_predictions = SAE_predictions.reshape((num_recs, n_steps, n_features))
SAE_predictions = SAE_predictions[:,-1,:]
SAE_predictions = np.column_stack((SAE_predictions, ticker_date))

# Save predictions
np.savetxt('..\Data\SAE_training_encoded_5timestep.csv', SAE_predictions, delimiter=',', fmt='%s')

# Reverse scaling
#SAE_predictions_inverse = scaler.inverse_transform(SAE_predictions)
#rmse = np.sqrt(np.mean(((SAE_predictions_inverse - y_test)**2)))

# Evaluate results of validation data
SAE.evaluate(x_validation_matrix_drop, x_validation_matrix_drop)




######################################
# POSSIBLE RESOURCES
######################################
#https://predictivehacks.com/autoencoders-for-dimensionality-reduction/
#https://stackoverflow.com/questions/58449353/lstm-deal-with-multiple-rows-in-a-date
#https://towardsdatascience.com/how-to-reshape-data-and-do-regression-for-time-series-using-lstm-133dad96cd00
#https://www.datatechnotes.com/2018/12/time-series-data-prediction-with-lstm.html
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://towardsdatascience.com/autoencoders-in-practice-dimensionality-reduction-and-image-denoising-ed9b9201e7e1
#https://machinelearningmastery.com/autoencoder-for-regression/
#https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
#https://www.datasciencecentral.com/profiles/blogs/stock-price-prediction-using-lstm-long-short-term-memory 
#https://towardsdatascience.com/recurrent-neural-network-to-predict-multivariate-commodity-prices-8a8202afd853
#https://datascienceplus.com/long-short-term-memory-lstm-and-how-to-implement-lstm-using-python/
#https://www.datacamp.com/community/tutorials/autoencoder-keras-tutorial
#https://blog.keras.io/building-autoencoders-in-keras.html
#https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
#https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
#https://quantdare.com/dimensionality-reduction-method-through-autoencoders/

######################################
# UNUSED CODE SNIPPETS
######################################

## Determine how many steps via autocorrelation
#from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(data['Count'], lags=10)
#plot_pacf(data['Count'], lags=10)
#
#def array3d(df):
#    array = np.empty(shape=(0, df.shape[1]), dtype='object')
#    ticker_set = list(set(df.TICKER))
#    z = len(set(df.DATE))
#    for i in range(len(ticker_set)):
#        t = ticker_set[i]
#        #print(t)
#        a = np.array(df[(df.TICKER == t)])
#        #print(a.shape)
#        b = np.pad(a, [(0,z-a.shape[0]),(0,0)], mode='constant', constant_values=(-1))
#        #print(b)
#        array = np.append(array, b, axis=0)
#    array = array.reshape((len(ticker_set), z, df.shape[1]))
#    return array
#
#def matrix5day2(df):
#    Xarray = np.empty(shape=(0, 5, df.shape[1]-1), dtype='object')
#    yarray = np.empty(shape=(0, ), dtype='object')
#    ticker_set = list(set(df.TICKER))
#    for i in range(len(ticker_set)):
#        t = ticker_set[i]
#        print(t)
#        a = np.array(df[(df.TICKER == t)])
#        X3, y3 = split_sequences(a, n_steps)
#        print(X3.shape, y3.shape)
#        Xarray = np.append(Xarray, X3, axis=0)
#        yarray = np.append(yarray, y3, axis=0)
#        print(Xarray.shape, yarray.shape)
#    return np.array(Xarray), np.array(yarray)
#
#brandi = list()
#for i in range(x_train_matrix.shape[0]):
#    print(x_train_matrix[i].shape)
#    brandi.append(x_train_matrix[i].shape)
#list(set(matrix_train.TICKER))[1596]
#matrix_train[(matrix_train.TICKER == 'HONE')]
#
## Encoder
#input_layer = Input(shape=(input_dim, ))
#encoder_layer_1 = Dense(48, activation="tanh")(input_layer)
#encoder_layer_2 = Dense(12, activation="tanh")(encoder_layer_1)
#encoder_layer_3 = Dense(encoding_dim, activation="tanh")(encoder_layer_2)
## Create model
#SAE = Model(inputs=input_layer, outputs=encoder_layer_3)
#print(SAE.summary())
## Generate predictions
#AE_2 = pd.DataFrame(SAE.predict(x_train), columns=['factor1','factor2','factor3'])
#AE_2['target'] = y_train
## Generate plot showing reduced dimensions
#plt.title('First two dimensions of encoded data, colored by single day returns')
#plt.scatter(AE_2['factor1'], AE_2['factor2'], c=AE_2['target'], s=1, alpha=0.3)
#plt.show()
#
## Reshaping the array from 3D array to 2D array
#np.savez_compressed(r'..\Data\x_train.npz', x_train_matrix_drop.reshape(x_train_matrix_drop.shape[0], -1))
## Load data
#x_train_loaded = np.load(r'..\Data\x_train.npz')
#print(x_train_loaded.files)
#x_train_loaded = x_train_loaded['arr_0']
## Reshape to 3D array
#x_train_loaded = x_train_loaded.reshape(x_train_loaded.shape[0], x_train_loaded.shape[1] // 204, 204)
#
## Predictions and inverted scaling
#yhat = model_lstm.predict(xtest)
#xtest = xtest.reshape((xtest.shape[0], xtest.shape[2]))
#inv_yhat = concatenate((yhat, xtest[:, 1:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:,0]
#
#ytest = ytest.reshape((len(ytest), 1))
#inv_y = concatenate((ytest, xtest[:, 1:]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,0]
#
#print("MAE:  %f" % sklearn.metrics.mean_absolue_error(inv_y, inv_yhat))
#print("MSE:  %f" % sklearn.metrics.mean_squared_error(inv_y, inv_yhat))
#print("RMSE: %f" % math.sqrt(sklearn.metrics.mean_squared_error(inv_y, inv_yhat)))
#print("R2:   %f" % sklearn.metrics.r2_score(inv_y, inv_yhat))
#
#plt.plot(inv_y, label='Actual')
#plt.plot(inv_yhat, label='Predicted')
#plt.legend()
#plt.show()

# Reverse scaling
#SAE_predictions_inverse = scaler.inverse_transform(SAE_predictions)
#rmse = np.sqrt(np.mean(((SAE_predictions_inverse - y_test)**2)))