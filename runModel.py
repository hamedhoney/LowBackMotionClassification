import keras
import numpy as np
from models import DCModel
from utils.preprocessing import getData
from utils.postprocessing import plotReconstruct, plotHistogram, predict, modelSstats

# Load data
trainData, valData, testData, normalTrainLabels, valLabels, anamolyTestLabels = getData()

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath='./training_checkpoints/best.keras',
    save_weights_only=False,
    verbose=1,
    save_best_only=True,
)

BUFFER_SIZE = len(trainData)
BATCH_SIZE = 500
EPOCHS = 2

noiseDim = 256
nExamples = 5
autoencoder = DCModel.AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(trainData, trainData,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(valData, valData),
          callbacks=[cp_callback],
          shuffle=True)

encoded_data = autoencoder.encoder(trainData).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plotReconstruct(trainData[0], decoded_data[0])

encoded_data = autoencoder.encoder(testData).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plotReconstruct(testData[0], decoded_data[0], signal='anomolous')

# Detect Anomoly
reconstructions = autoencoder.predict(trainData)
train_loss = keras.losses.mean_absolute_error(reconstructions, trainData)
# Set threshoild to one standard deviation from the mean
threshold = np.mean(train_loss) + np.std(train_loss)

plotHistogram(train_loss[None, :], threshold)

reconstructions = autoencoder.predict(testData)
test_loss = keras.losses.mean_absolute_error(reconstructions, testData)

plotHistogram(test_loss[None, :], threshold, signal='anomolous')

preds = predict(autoencoder, valData, threshold)
modelSstats(preds, valLabels, threshold)
