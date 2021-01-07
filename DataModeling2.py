# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2020
Author: Brandi Beals
Description: Thesis LSTM Neural Network
"""

######################################
# IMPORT PACKAGES
######################################

import os
from tensorflow.keras.models import load_model
import tensorflow as tf

## Set working directory
path = r'C:\Users\bbeals\Dropbox (Personal)\Masters in Predictive Analytics\590-Thesis\NW590-Thesis'
os.chdir(path)


## https://www.tensorflow.org/tutorials/keras/save_and_load

print(os.ls AE_1_20201227)

#SAE = load_model('AE_1_20201227')

SAE = tf.saved_model.load('AE_1_20201227')
SAE.summary()
outputs = SAE(x_validation)
outputs = SAE.predict(x_validation)
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)

SAE.set_weights(SAE.get_weights())

reconstructed_model = load_model("AE_1_20201227")
