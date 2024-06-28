import keras
import numpy as np
from models import LSTMModel
from utils.preprocessing import getData
from utils.postprocessing import plotReconstruct, plotHistogram, predict, modelSstats

# Load data
trainData, valData, testData, normalTrainLabels, valLabels, anamolyTestLabels = getData()

input_shape = (512, 1)  # Input shape

# Reshape data
trainData = trainData[:,:,None] 
valData = valData[:,:,None] 
testData = testData[:,:,None] 
normalTrainLabels = np.array(normalTrainLabels)[:,None] 
valLabels = np.array(valLabels)[:,None]
anamolyTestLabels = np.array(anamolyTestLabels)[:,None]

# Training the model
# vae.fit([train_data, train_labels], 
#       [train_data, train_labels],
#       epochs=100, 
#       batch_size=16,
#       validation_data=([val_data, val_labels], [val_data, val_labels]))

# Evaluate the model on test data
# vae.evaluate([test_data, test_labels], [test_data, test_labels])
# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath='./training_checkpoints/LSTM.weights.h5',
    save_weights_only=True,
    verbose=1,
    save_best_only=True,
)

BUFFER_SIZE = len(trainData)
BATCH_SIZE = 500
EPOCHS = 1000

noiseDim = 256
nExamples = 5
train = True
if train:
    autoencoder = LSTMModel.VAE(input_shape)
    autoencoder.compile(optimizer='adam', loss='mae')
    history = autoencoder.fit(
        trainData,
        trainData,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(valData, valData),
        callbacks=[cp_callback],
        shuffle=True,
    )
else:
    autoencoder = LSTMModel.VAE(input_shape)
    autoencoder.load_weights('./training_checkpoints/LSTM.weights.h5')

# Use the trained VAE model to reconstruct test data
reconstructions = autoencoder.predict(trainData)
trainData = np.squeeze(trainData, axis=-1)
decoded_data = np.squeeze(reconstructions[0], axis=-1)
trainLoss = keras.losses.mean_absolute_error(decoded_data, trainData)
plotReconstruct(trainData[0], decoded_data[0], model='LSTM')

reconstructions = autoencoder.predict(testData)
testData = np.squeeze(testData, axis=-1)
decoded_data = np.squeeze(reconstructions[0], axis=-1)
testLoss = keras.losses.mean_absolute_error(decoded_data, testData)
plotReconstruct(testData[0], decoded_data[0], signal='anomolous', model='LSTM')

# Detect Anomoly
# reconstructions = autoencoder.predict(trainData)
# Set threshoild to one standard deviation from the mean
threshold = np.mean(trainLoss) + np.std(trainLoss)

plotHistogram(trainLoss, threshold, model='LSTM')

plotHistogram(testLoss, threshold, signal='anomolous', model='LSTM')

reconstructions = autoencoder.predict(valData)
decoded_data = np.squeeze(reconstructions[0], axis=-1)
valData = np.squeeze(valData, axis=-1)
loss = keras.losses.mean_absolute_error(decoded_data, valData)
import tensorflow as tf
preds = tf.math.less(loss, threshold)
modelSstats(preds, valLabels, threshold)
