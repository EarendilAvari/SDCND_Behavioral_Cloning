#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:32:40 2019

@author: earendilavari
"""

#%% LOADS SEPARATED SHUFFLED DATA
import pickle

with open('TrainingDataProcessed/dataset.p', 'rb') as pickleFile:
    pickleData = pickle.load(pickleFile)
    X_train_paths = pickleData['X_train_paths']
    Y_train = pickleData['Y_train']
    X_validation_paths = pickleData['X_validation_paths']
    Y_validation = pickleData['Y_validation']
    X_test_paths = pickleData['X_test_paths']
    Y_test = pickleData['Y_test']
    del pickleData
    

from keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize
import numpy as np

#%% GENERATOR USED TO LOAD DATA IN TRAINING TIME
## Credits to Ramin Rezaiifar, 
## link: https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4

class BigDataGenerator(Sequence):
    
    def __init__ (self, imageNames, imageMeasurements, batchSize):
        self.image_filenames = imageNames
        self.image_measurements = imageMeasurements
        self.batch_size = batchSize
        
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))
    
    def __getitem__(self, batchNumber):
        batchStart = batchNumber*self.batch_size
        batchEnd = batchStart + self.batch_size
        X_batch_filenames = self.image_filenames[batchStart:batchEnd]
        Y_batch = self.image_measurements[batchStart:batchEnd]
        
        return np.array([resize(imread(fileName), (160, 320)) for fileName in X_batch_filenames]), np.array(Y_batch)

#%% USING MODEL OF PROJECT 3 (TRAFFIC SIGN CLASSIFIER)
# As first model alternative for this task, the improved LeNet network used on the last project is used. Here it is programmed
# again using Keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['QT_STYLE_OVERRIDE']='gtk2'

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D


model = Sequential()
# Normalization layer (from 0-255 to 0-1)
model.add(Lambda(lambda x: x/255.0, input_shape = (160, 320, 3)))
# First convolutional layer
model.add(Conv2D(24, kernel_size=(5,5), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))
# Second convolutional layer
model.add(Conv2D(36, kernel_size=(5,5), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))
# Third convolutional layer
model.add(Conv2D(48, kernel_size=(5,5), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))
# Fourth convolutional layer
model.add(Conv2D(64, kernel_size=(3,3), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))
# Fifth convolutional layer
model.add(Conv2D(64, kernel_size=(3,3), strides = (2,2), padding = 'valid'))
model.add(Activation('relu'))
# Flatten layer
model.add(Flatten())
# First fully connected layer
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Second fully connected layer
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# Third fully connected layer
model.add(Dense(10))
model.add(Activation('relu'))
# Output layer
model.add(Dense(1))

model.summary()

#%% PREPARATION OF THE DATA

BATCH_SIZE = 50

trainDataGenerator = BigDataGenerator(X_train_paths, Y_train, BATCH_SIZE)
validDataGenerator = BigDataGenerator(X_validation_paths, Y_validation, BATCH_SIZE)

#%% CALLBACKS FOR TRAINING  
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.trainingLoss = []

    def on_batch_end(self, batch, logs={}):
        self.trainingLoss.append(logs.get('loss'))
        
## Callback to save the best model
modelCheckpoint = ModelCheckpoint(filepath = 'modelBest.h5', monitor = 'val_loss', save_best_only = True)
earlyStopper = EarlyStopping(monitor = 'val_loss', min_delta = 0.0003, patience = 5)

#%% TRAINING OF THE MODEL

EPOCHS = 30

model.compile(loss = 'mse', optimizer = 'adam')
datalogBatches = LossHistory()
datalogEpochs = model.fit_generator(generator = trainDataGenerator, steps_per_epoch=(len(X_train_paths)//BATCH_SIZE),
                                    epochs = EPOCHS, verbose = 1, validation_data = validDataGenerator, 
                                    validation_steps = (len(X_validation_paths)//BATCH_SIZE), 
                                    callbacks = [datalogBatches, modelCheckpoint, earlyStopper])

model.save('model.h5')

#%%

import pickle
with open('modelDatalog.p', 'wb') as pickleFile:
    pickle.dump(datalogBatches.trainingLoss, pickleFile)
    pickle.dump(datalogEpochs.history['loss'], pickleFile)
    pickle.dump(datalogEpochs.history['val_loss'], pickleFile)




















    