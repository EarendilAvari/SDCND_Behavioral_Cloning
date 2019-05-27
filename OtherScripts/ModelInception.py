#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:15:13 2019

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
        
        return np.array([imread(fileName) for fileName in X_batch_filenames]), np.array(Y_batch)

#%% USING GoogLeNet INCEPTION MODEL

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['QT_STYLE_OVERRIDE']='gtk2'

import tensorflow as tf
tf.reset_default_graph()

from keras.models import Sequential
from keras.layers import Dense, Lambda, Input, GlobalAveragePooling2D
#from keras.layers.convolutional import Conv2D
#from keras.layers.pooling import MaxPool2D

from keras.applications.inception_v3 import InceptionV3

inception = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (160, 320, 3))

inception.summary()

#%%

model = Sequential()

model.add(Lambda(lambda x: x/255.0, input_shape = (160, 320, 3)))

# model.add(Lambda(lambda x: tf.image.resize_images(x, (80, 160))))

model.add(inception)

model.add(GlobalAveragePooling2D())

model.add(Dense(400, activation = 'relu'))

model.add(Dense(200, activation = 'relu'))

model.add(Dense(1))

#gAvgPooling = GlobalAveragePooling2D()(inception)

#Layer1 = Dense(400, activation = 'relu')(gAvgPooling)
#Layer2 = Dense(200, activation = 'relu')(Layer1)
#Layer3 = Dense(1)(Layer2)

#model = Model(inputs = normalizationLayer, outputs = Layer3)

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