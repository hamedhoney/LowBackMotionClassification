import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from models import DCGANModel
from utils.preprocessing import normalize, load, zeroPad
from utils.postprocessing import saveSignals
import time
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

anamoly, anamoly_labels, normal, normal_labels= load()
# Normalize normal Data to [0 1]
normal, anamoly = normalize(normal, anamoly)

maxSize=512 #Signal length to include
normal, maxSize = zeroPad(normal, maxSize=maxSize)
anamoly, maxSize = zeroPad(anamoly, maxSize=maxSize)
normalTrainData, normalValData, normalTrainLabels, normalValLabels = train_test_split(normal, normal_labels, test_size=0.2, random_state=42)
anamolyValData, anamolyTestData, anamolyValLabels, anamolyTestLabels = train_test_split(anamoly, anamoly_labels, test_size=0.5, random_state=42)

trainData = tf.reshape(normalTrainData, (len(normalTrainData), maxSize,))
ValData = tf.reshape(np.concatenate((normalValData,anamolyValData), axis=0), (len(normalValData)+len(anamolyValData), maxSize, ))
ValLabels = np.concatenate((normalValLabels,anamolyValLabels), axis=0)
testData = tf.reshape(anamolyTestData, (len(anamolyTestData), maxSize,))
# normalTrainData = tf.convert_to_tensor(normalTrainData,)

BUFFER_SIZE = len(trainData)
BATCH_SIZE = 10
EPOCHS = 10

noiseDim = 256
nExamples = 5
autoencoder = DCGANModel.AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(trainData, trainData,
          epochs=10000,
          batch_size=100,
          validation_data=(ValData, ValData),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

encoded_data = autoencoder.encoder(normalValData).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normalValData[10], 'b')
plt.plot(decoded_data[10], 'r')
# plt.fill_between(np.arange(140), decoded_data[0], normalValData[0, :], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

encoded_data = autoencoder.encoder(testData).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(testData[0], 'b')
plt.plot(decoded_data[0], 'r')
# plt.fill_between(np.arange(140), decoded_data[0], testData[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

# Detect Anomoly
reconstructions = autoencoder.predict(trainData)
train_loss = keras.losses.mean_absolute_error(reconstructions, trainData)

plt.hist(train_loss[None,:], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

reconstructions = autoencoder.predict(testData)
test_loss = keras.losses.mean_absolute_error(reconstructions, testData)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = keras.losses.mean_absolute_error(reconstructions, data)
    return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

preds = predict(autoencoder, ValData, threshold)
print_stats(preds, ValLabels)