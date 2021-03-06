# Self driving car nanodegree

## Project 4: Behavioral cloning

This project consists of a convolutional neural network capable of driving a car by itself on a simulator. In order to do that, the neural network receives a picture looking front from the front of the car as an input and gives the steering angle as an output. The neural network is programmed and trained in Keras using data obtained with the Udacity's simulator. Then the saved network is used to drive the car autonomously on the same simulator. The programming language used is Python and the libraries used are TensorFlow version 1.4.0, Keras version 2.2.4, Numpy, OpenCV and Sklearn.

### Data collection

In order to collect the data needed to train the network, the [Udacity's car simulator ](https://github.com/udacity/self-driving-car-sim) is used. This simulator records data of a car driven by a human using its "training mode". This data consists on three images taken from the front mirror and the two side mirrors of the car and the steering angle applied by the human. In order to control the car, the keyboard, the mouse or a joystick can be used. The data used to train the models in this project are collected using a joystick since it gives the best steering angles because of being an analog input instead of a digital input like from a keyboard.

In order to train a model, about 15 minutes of driving were recorded. This corresponds to 11577 images. Within these 15 minutes the car is kept mostly on the center of the line. This training data can be downloaded [here](https://mega.nz/#!H5oxRCBC!cC9xr-hYPudB2hgmMPW9duAcGBAt3k9doaydc7NFBj0). 80% of the data is used for training and 20% for validation. The separation between training and validation data is done using the Sklearn function "train_test_split". No test data was generated since the model will be tested afterwards on the simulator.

Here an image taken from the front mirror:

![ Image8](./ImgsReport/08_TrainingCenterImage.jpg  "TrainingImageFront")

Here an image taken from the left mirror:

![ Image9](./ImgsReport/09_TrainingLeftImage.jpg  "TrainingImageLeft")

Here an image taken from the right mirror:

![ Image10](./ImgsReport/10_TrainingRightImage.jpg  "TrainingImageRight")


### Selection of the model

The model selected is the same used for the [traffic sign classifier ](https://github.com/EarendilAvari/SDCND_Traffic_Sign_Classifier) was selected, since it has shown a very high performance detecting street signs. This model was modified to the new task like this:

- Normalization layer for input from range 0-255 to -0.5-0.5
- Convolutional layer with kernel size 5x5, strides 1x1, 10 feature maps as output and Relu activation function.
- Max pooling layer with pool size 2x2 and strides 2x2.
- Convolutional layer with kernel size 4x4, strides 1x1, 18 feature maps as output and Relu activation function.
- Max pooling layer with pool size 2x2 and strides 2x2.
- Convolutional layer with kernel size 3x3, strides 1x1, 30 feature maps as output and Relu activation function.
- Max pooling layer with pool size 2x2 and strides 2x2.
- Flatten operation layer in order to convert output of last max pooling layer into a big array.
- Fully connected layer with 490 outputs, 50% percent of keeping nodes at training time and Relu activation function.
- Fully connected layer with 220 outputs, 50% percent of keeping nodes at training time and Relu activation function.
- Fully connected layer with 43 outputs and Relu activation function.
- Fully connected layer with 1 output and Relu activation function.

A visualization of the model here:

![ Image1](./ImgsReport/07_ModelArchitecture.png  "TrainingData3Cameras")

For training the Adam optimizer was selected and the default learning rate was not changed. In this case, since this is a regression model instead of a classification model, the loss function used is not the cross entropy.  Instead, the [mean squared error ](https://en.wikipedia.org/wiki/Mean_squared_error) between the training stearing angle and the prediction is what gets minimized.

The definition of the model and its training is done on the script "Model.py"

### Training using only front camera images

As first attempt the model was trained using only the frontal images which are 3859. No metrics where collected since it was only a test to see how the model was performing at the beginning.

By using the trained model to drive the car in autonomous mode, it dit a really good job, with few data and no data augmentation. It could not make the entire lap though, The car went out of the street on a curve with mud over the street line.  This can be seen in the following video.

[![Autonomous car, try 1](https://i.imgur.com/aR3TV6A.png)](https://www.youtube.com/watch?v=8-T4qriTvCE "Autonomous car, try 1")

#### Training with images of the three cameras

The performance of the model can be increased by using the images of the lateral cameras. With them, the model can learn how to center the car when it goes out from the center of the lane line. Having more data to learn also can prevent overfitting. 

In order to use the lateral images an offset of 0.2 radians for the angle measurements corresponding to the left camera is used and an offset of -0.2 radians for the angle measurements corresponding to the right camera is used. 

Using these images not only increases the amount of data from 3859 to 11577, it also helps the car to learn how to come back to the street after going out from it.

Using these images actually increased the performance of the model a bit, but at the end the car also went out of the road.

[![Autonomous car, try 2](https://i.imgur.com/S7PBb26.jpg)](https://www.youtube.com/watch?v=8Heo4AqVRbc "Autonomous car, try 2")

Analysing the training and validation loss, it can be seen that the training loss is lower than 0.0025 after training, but the validation loss is like 0.010 which is 4 times higher than the training loss. This indicates that the can be overfitting the training data, so data augmentation or more data is needed here.

![ Image1](./ImgsReport/01_TrainingLossBatches3Cams.png  "TrainingData3Cameras")

### Data augmentation

#### Fliping images horizontally

In the standard training loop, the car is normally always going straight or to the left, this can result that the car does not learn how to steer to the right because of lacking training data. One way to fight this is to flip the training images horizontally and negate the angle measurement.

By adding horizontally flipped images to the dataset, its size duplicated. And by training the network with this augmented dataset, the training and validation losses stayed almost the same. So this did not really help. Also the performance of the model on the test route was not better.  

![ Image2](./ImgsReport/02_TrainingLossBatchesFlippedImages.png  "TrainingDataFlippedImages")


#### Using Keras' image data generator

Other option is to use the built in Keras image data generator to create augmented data in training time. For this it is important to deactivate the options "horizontal flip" and "vertical flip" since that can result on bad commands for the neural network. In order to use this method, the function "fit_generator" needs to be used for training instead of "fit". This was programmed on a new script called "ModelAug.py".

By using the generator to train the model, it is still not able to drive a lap in the simulator like it can be seen on the next video. This trained model can be downloaded [here](https://mega.nz/#!Pl5QwQxa!cfDL3s71yGNc21w046E54wDKkmBbpPK1k0RwMSxP9Ik).

[![Autonomous car, try with augmented data](https://i.imgur.com/acV8zfl.jpg)](https://www.youtube.com/watch?v=HHQeYbk3iK0 "Autonomous car, try with augmented data")

It can be seen though that the training loss increased a lot, staying higher than the validation accuracy which is calculated without data augmentation. This means that the model is not overfitting anymore. So augmenting the data using this way is a good approach.

![ Image3](./ImgsReport/03_TrainingLossAugmentedImages.png  "TrainingDataAugmentedImages")

In order to improve the performance of the model, more data will be needed.

### New data

Until now, we are training the model with only 11577 images, this is considered a little dataset. The next step in order to improve the performance of the model is to get more data. 

Using the simulator, about 45 minutes of data were collected, this corresponds to 48651 images. This would still be considered a little dataset, but at least 4 times bigger than the original one. This dataset is also more diverse. It contains:

- Images of the training loop in clockwise direction.
- Images of the training loop in counter clockwise direction.
- Images of the training loop with the car reaching the border of the street and coming back.

This data can be downloaded [here](https://mega.nz/#!W4oQHYbI!MBQNmMcyB6TfsEiD7oTMgy7Vao6dOA6iFdEAZHvoGeo).

Since the performance of the last training was not that bad, being the car capable to drive like 50% of the lap, this trained model was used as a base. In other words, the already trained model was trained further using the new data. In order to avoid overfitting since the new data is similar to the first one, early stopping was used, monitoring the changes on the validation loss.

For this, the new script "ModelContTraining.py" was programmed. Contrary to the training scripts used before, the entire dataset was not loaded directly into the RAM memory before training since the dataset is now way bigger. Instead, a data generator was programmed, which loads batches of data in training time. No augmentation was performed on this data, but the images of the three cameras were used.

Using this approach the car was able to drive 3 laps correctly without going out from the street. On the fourth lap the car went out of the street in the part where the street border cannot be seen. This is the same part where it was going out on the last cases. This model can be downloaded [here](https://mega.nz/#!y5oQASZI!1nb1d4eqmHjyM-fb-fkR3dnh4gI3y300RmZE4ZHOI0Q).

[![Autonomous car, try with new data](https://i.imgur.com/ERFYOSU.jpg)](https://www.youtube.com/watch?v=nVgSUO4TSx4& "Autonomous car, try with new data")

From the image it can be seen that the validation loss now reached a value of 0.004, which is lower than the validation losses obtained on the last experiments. It is still not a big difference though, Also the training loss stayed lower than the validation loss during most of the training, what is normal, considering that the data was not augmented this time. The training loss reached a value of 0.002. 

![ Image4](./ImgsReport/05_TrainingLossMoreData.png  "TrainingDataMoreData")

The model could still be trained using only data from the part where it goes out of the street so it memorizes it, but this can worsen the performance on other parts of the track. For this project no further training on this track will be done.

### Other experiments

It is important to mention that other more complex models were trained and tested for this problem without success. The models where the Inception V3 of Google and the [DAVE-2 of Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which is the inspiration for this project. This can be explained by the fact that my model has fewer convolutional layers than these other two and the track does not really have complex forms to be recognized in order to need more convolutional layers. An approach with fewer convolutional layers is better since the input to the fully connected layers is closer to the original image than the case when using more convolutional layers.

### Training for a new route

The model is now able to drive a car in the route 1. Now it could be trained for a second route using the last trained model as a base. This new route looks more like a normal street and is way more difficult to drive than the first route since it has more curves and slopes.  

In order to train the model to be able to drive this route, about 1.5 hours of driving data on this route were recorded, this is equivalent to 87897 images. 3 laps in clockwise direction and 3 laps in counterclockwise direction. This data can be downloaded [here](https://mega.nz/#!f4hQDIxT!U5L-_bu_gxOD4SkctEJSNr8vcGmy4PxBogxOsaC8EJk).

The model was trained using the weights obtained on the last training. The processing is done on a new script called "Model2Track.py". 

With the model trained on the new track's data, the car was able to drive 1 minute long over 3 difficult curves, at the fourth curve, which is a very sharp U curve, the model did not know what to do and crashed the car against the containment barrier.  This model can be downloaded [here](https://mega.nz/#!Os4gHYYL!2flqsZUeOadgEQyxA6sEnGQyucMdbNFHE3hRzqNJavA).

[![Autonomous car, try with new track](https://i.imgur.com/CK8L11O.jpg)](https://www.youtube.com/watch?v=cEU1FuCzP9Q "Autonomous car, try with new track")

Analysing the performance of the model on this new track, it looks like the steering commands are fine, but the speed is inconsistent. The model is right now not controlling the speed. Instead, it gets controlled by a PI controller with fixed setpoint by the file "drive.py". A way to improve the performance of the model would be to add it the desired throttle value as an output and to train it using the throttle values recorded on the training data. In this case, the PI controller would need to be eliminated from the file drive.py and replaced by the new output of the model. Because of time reasons, this will not be executed right now.

![ Image5](./ImgsReport/06_TrainingLossNewTrack.png  "TrainingDataOtherTrack")

It can be seen that the validation loss reached a value of 0.04, while the training loss reached a value of 0.021 and it was still going down. This is an indication that the model was starting to overfit.

### Conclusions and future work

This is an experimental project which shows the strength of deep learning not only on classification tasks but also on controlling tasks. But it also shows how difficult is to correct a bug in deep learning since they are black box models. Sometimes, to correct a deep learning bug can mean to train the model completely from zero with new data or to change the model completely and then train it from zero.

A more traditional approach for this task would be [detecting the lane line and the curvature radius using computer vision techniques](https://github.com/EarendilAvari/SDCND_Advanced_Lane_Finding) and with the radius of curvature control the stearing angle using a PI or an adaptative controller. 

As said before, in order to get a better performance from the model on the second track, it will need to learn also which speed is needed on every curve, because now the speed is inconsistent since it is a constant given by a PI controller. In order to do this, the file drive.py and the model will need to be modified and the model will need to be trained again from zero using the data of the second track.






















 






