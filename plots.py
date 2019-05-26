
#%% Load data from pickle file 
import pickle

with open('OldModels/modelDatalog6.p', 'rb') as pickleFile:
    trainingLossBatches = pickle.load(pickleFile)
    trainingLossEpochs = pickle.load(pickleFile)
    validationLossEpochs = pickle.load(pickleFile)
    
    
#%%
import matplotlib.pyplot as plt    
import numpy as np
batches = np.linspace(0,len(trainingLossBatches)-1,len(trainingLossBatches))
epochs = np.linspace(1, len(trainingLossEpochs), len(trainingLossEpochs))
    
figure1, fig1_axes = plt.subplots(2, 1, figsize =(15,10))
figure1.suptitle('Training and validation loss for model with augmented data', fontsize = 20)
fig1_axes[0].plot(epochs, trainingLossEpochs, label = 'Training loss')
fig1_axes[0].plot(epochs, validationLossEpochs, label = 'Validation loss')
fig1_axes[0].legend(loc = 'upper right')
fig1_axes[0].set_title('Training and validation loss every epoch')
fig1_axes[0].set_xticks(np.arange(0, len(trainingLossEpochs)+1, step = 1))
fig1_axes[1].plot(batches, trainingLossBatches)
fig1_axes[1].set_title('Training and validation loss every batch')
figure1.savefig('ImgsReport/03_TrainingLossAugmentedImages')


