from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

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