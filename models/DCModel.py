import numpy as np
import tensorflow as tf
import keras

class AnomalyDetector(keras.models.Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = keras.Sequential([
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(8, activation="relu")])

    self.decoder = keras.Sequential([
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(512, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
class AnomalyDetector1(keras.models.Model):
  def __init__(self):
    super(AnomalyDetector1, self).__init__()
    self.encoder = keras.Sequential([
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(8, activation="relu")])

    self.decoder = keras.Sequential([
      keras.layers.Dense(16, activation="relu"),
      keras.layers.Dense(32, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(512, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  
def generatorModel(noise_dim=512, ):
  signalLength = 512
  model = keras.Sequential([
  keras.layers.Dense(256*8, use_bias=False, input_shape=(noise_dim,)),
  keras.layers.BatchNormalization(),
  keras.layers.LeakyReLU(),

  keras.layers.Reshape((8, 256)),
  keras.layers.Conv1DTranspose(256, 12, strides=2, padding='same', use_bias=False),
  keras.layers.BatchNormalization(),
  keras.layers.LeakyReLU(),

  keras.layers.Conv1DTranspose(128, 25, strides=2, padding='same', use_bias=False),
  keras.layers.BatchNormalization(),
  keras.layers.LeakyReLU(),

  keras.layers.Conv1DTranspose(64, 25, strides=2, padding='same', use_bias=False),
  keras.layers.BatchNormalization(),
  keras.layers.LeakyReLU(),

  keras.layers.Conv1DTranspose(32, 50, strides=2, padding='same', use_bias=False),
  keras.layers.BatchNormalization(),
  keras.layers.LeakyReLU(),

  keras.layers.Conv1DTranspose(5, 50, strides=2, padding='same', use_bias=False, activation='tanh'),

  keras.layers.Conv1DTranspose(1, 100, strides=2, padding='same', use_bias=False, activation='tanh'),
  ])
  return model

def discriminatorModel():
    signalLength = 512
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(64, 5, strides=2, padding='same',
                                     input_shape=[signalLength,1]))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv1D(128, 5, strides=2, padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.BatchNormalization(-2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Conv1D(256, 5, strides=2, padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.BatchNormalization(-2))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model

def discriminatorLoss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generatorLoss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

