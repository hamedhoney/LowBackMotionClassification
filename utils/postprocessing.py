from typing import Optional
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf

def plotSignals(dataset):
  for i in range(dataset.shape[0]):
    plt.figure()
    for j in range(dataset.shape[-1]):
      plt.subplot(j+1, 1, j+1)
      plt.plot(dataset[i, j, :, 0])
      plt.xlabel('sample number')
      plt.ylabel('degree')
    #   plt.title(trials[j])
    fig = plt.gcf()
    fig.set_size_inches((8.5, 11), forward=False)
    plt.savefig('output/real{:04d}.png'.format(i), dpi=500)

def saveSignals(model, epoch, test_input):
  # Notice training is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)


  for i in range(predictions.shape[0]):
    plt.figure()
    for j in range(predictions.shape[-1]):
      plt.subplot(j+1, 1, j+1)
      plt.plot(predictions[i, :, j])
      plt.xlabel('sample number')
      plt.ylabel('degree')
    #   plt.title(trials[j])

  plt.savefig('output/signal_at_epoch_{:04d}.png'.format(epoch))
  plt.close('all')

def plotHist(loss, threshold: Optional[float]):
  if not threshold:
    threshold = np.mean(loss) + np.std(loss)
  plt.hist(loss, 20)
  plt.savefig('output/histogram.png')
  plt.close('all')

def plotTrainingPerformance(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.savefig('output/training.png')
    plt.show()
     
def plotHistogram(loss, threshold, signal='normal', ):
  plt.hist(loss, bins=50)
  plt.axvline(x=threshold, color='red')
  plt.xlabel("Train loss")
  plt.ylabel("No of examples")
  plt.title(f"Histogram of {signal} signals.")
  plt.savefig(f'output/{signal}Hist.png')
  plt.show()
  plt.close()

def plotReconstruct(data, decoded_data, signal='normal', ):
  plt.plot(data, 'b')
  plt.plot(decoded_data, 'r')
  plt.fill_between(np.arange(data.shape[0]), decoded_data, data, color='lightcoral')
  plt.legend(labels=["Input", "Reconstruction", "Error"])
  plt.title(f"Sample {signal} signal")
  plt.savefig(f'output/{signal}signal.png')
  plt.show()
  plt.close()

def predict(model, data, threshold):
    reconstructions = model(data)
    loss = keras.losses.mean_absolute_error(reconstructions, data)
    return tf.math.less(loss, threshold)

def modelSstats(predictions, labels, threshold):
    import json
    with open('output/performance.json', 'w') as f:
        json.dump({
            "threshold":threshold,
            "Accuracy":accuracy_score(labels, predictions),
            "Precision":precision_score(labels, predictions),
            "Recall":recall_score(labels, predictions),
        }, f)

    