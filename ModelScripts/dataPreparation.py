#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reads the data saved on the path TrainingData, separates in in three folders for
training, validation and test, while the measurements are separated in three lists. The paths to 
the images is also saved on three lists.
"""
#%%
import csv
import cv2
import random
import pickle

# Reads the csv file where the path to the images and the measurements are
csvLines = []
with open('TrainingData3/driving_log.csv') as csvFile:
    reader = csv.reader(csvFile)
    for line in reader:
        csvLines.append(line)
        
## Batch size corresponds to the quantity of lines in the csv file which are processed at the 
## same time
batch_size = 1000
batch_quantity = len(csvLines)/batch_size
angleCorrectionFactor = 15 # Corresponds to the correction factor used for the measurements of 
                           # lateral images

## Lists with the paths to the separated images
X_train_images_path = []
X_validation_images_path = []
X_test_images_path = []
## Lists with the measurements for the images
Y_train = []
Y_validation = []
Y_test = []
## Parent folder of training, validation and test images
X_train_path = 'TrainingDataProcessed/training/'
X_validation_path = 'TrainingDataProcessed/validation/'
X_test_path = 'TrainingDataProcessed/test/'
## Numbers used to name the training, validation and test images
X_train_counter = 0
X_validation_counter = 0
X_test_counter = 0
## List used in order to randomize the processing order of the batches
batchUsed = []


for batch in range(0, int(batch_quantity)):
    ## It takes a random batch to separate it into test, validation and train images
    ## If the random number received was received before on another loop, another random
    ## number is requested
    while True:    
        batchToProcess = random.randint(0, int(batch_quantity))
        if (batchToProcess not in batchUsed):
            break
    batchUsed.append(batchToProcess)
    ## Cuts the csv file to the batch selected
    batch_start = batchToProcess*batch_size
    batch_end = batch_start + batch_size
    csvLinesBatch = csvLines[batch_start:batch_end]
    for line in csvLinesBatch:
        for i in range(3):
            imgPath = line[i]
            imgName = imgPath.split('/')[-1]
            currPath = 'TrainingData3/IMG/' + imgName
            imgBGR = cv2.imread(currPath) # It is not needed to transform the image to RGB since
                                            # they are saved afterwards using imwrite of OpenCV.
            ## Applies angle correction to the lateral images
            if i == 0:
                angleMeasurement = float(line[3])
            if i == 1:
                angleMeasurement = float(line[3]) + angleCorrectionFactor
            if i == 2:
                angleMeasurement = float(line[3]) - angleCorrectionFactor
            ## 70% of the images go to training, 8% to validation and 22% to test
            dataSplitter = random.randint(0,99)    
            if (dataSplitter < 70):
                ## Saves image to training folder of "TrainingDataProcessed"
                imgNewName = X_train_path + str(X_train_counter).zfill(5) + '.jpg'
                cv2.imwrite(imgNewName, imgBGR)
                X_train_counter += 1 ## Increases counter for next image
                ## Saves path to saved image and angle measurement into list
                X_train_images_path.append(imgNewName)
                Y_train.append(angleMeasurement)
            elif (dataSplitter > 91):
                ## Saves image to validation folder of "TrainingDataProcessed"
                imgNewName = X_validation_path + str(X_validation_counter).zfill(5) + '.jpg'
                cv2.imwrite(imgNewName, imgBGR)
                X_validation_counter += 1 ## Increases counter for next image
                ## Saves path to saved image and angle measurement into list
                X_validation_images_path.append(imgNewName)
                Y_validation.append(angleMeasurement)
            else:
                ## Saves image to test folder of "TrainingDataProcessed"
                imgNewName = X_test_path + str(X_test_counter).zfill(5) + '.jpg'
                cv2.imwrite(imgNewName, imgBGR)
                X_test_counter += 1 ## Increases counter for next image
                ## Saves path to saved image and angle measurement into list
                X_test_images_path.append(imgNewName)
                Y_test.append(angleMeasurement)
                
#%%               
                
with open('TrainingDataProcessed/dataset.p', 'wb') as pickleFile:
    pickle.dump(
            {
                'X_train_paths': X_train_images_path,
                'Y_train': Y_train,
                'X_validation_paths': X_validation_images_path,
                'Y_validation': Y_validation,
                'X_test_paths': X_test_images_path,
                'Y_test': Y_test
            }, pickleFile, pickle.HIGHEST_PROTOCOL)
                
                
                
                
                
                
                
                
                
                
                
                
                
            
        
    