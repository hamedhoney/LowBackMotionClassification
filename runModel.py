import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import numpy as np
from models import DCGANModel
from utils.preprocessing import normalize, load, zeroPad
from utils.postprocessing import saveSignals
import time
import os

anamoly, anamoly_labels, normal, normal_labels= load()
normal = normalize(normal)
normal = normalize(anamoly)
maxSize=512
normal, maxSize = zeroPad(normal, maxSize=maxSize)
anamoly, maxSize = zeroPad(anamoly, maxSize=maxSize)
trainData, valData, trainLabels, valLabels = train_test_split(normal, normal_labels, test_size=0.2, random_state=42)

# Normalize Data to [-1 1]
trainData = tf.reshape(trainData, (len(trainData), maxSize, 1,))
valData = tf.reshape(valData, (len(valData), maxSize, 1,))
testData = tf.reshape(anamoly, (len(anamoly), maxSize, 1,))
# trainData = tf.convert_to_tensor(trainData,)

BUFFER_SIZE = len(trainData)
BATCH_SIZE = 10
EPOCHS = 1000
trainDataset = tf.data.Dataset.from_tensor_slices(trainData).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
noiseDim = 256
nExamples = 5

generator = DCGANModel.generatorModel()
noise = tf.random.normal([nExamples, noiseDim])
generatedSignal = generator(noise, training=False)
# plt.plot(generatedSignal[0,:,0])
# plt.show()

discriminator = DCGANModel.discriminatorModel()
decision = discriminator(generatedSignal)
print(decision)

generatorOptimizer = keras.optimizers.Adam(1e-4)
discriminatorOptimizer = keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generatorOptimizer=generatorOptimizer,
                                 discriminatorOptimizer=discriminatorOptimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([nExamples, noiseDim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noiseDim])

    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generatedSignals = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generatedSignals, training=True)

      gen_loss = DCGANModel.generatorLoss(fake_output, cross_entropy)
      disc_loss = DCGANModel.discriminatorLoss(real_output, fake_output, cross_entropy)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generatorOptimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminatorOptimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    # display.clear_output(wait=True)
    saveSignals(generator,
                epoch + 1,
                seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  # display.clear_output(wait=True)
  saveSignals(generator,
              epochs,
              seed)


train(trainDataset, EPOCHS)

def reconstructionError(images):
  noise = tf.random.normal([images.shape[0], noiseDim])
  generatedSignals = generator(noise, training=False)
  train_loss = keras.losses.mean_absolute_error(generatedSignals, images)
  return train_loss
  
trainLoss = reconstructionError(trainData)

THRESHOLD = np.mean(trainLoss) + np.std(trainLoss)
print("Threshold: ", THRESHOLD)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# validate normal signals
valLoss = reconstructionError(valData)
correct = sum(l < THRESHOLD for l in valLoss)
print(f'Correct normal predictions: {correct}/{len(valData)}')
# test anomolous signals 
testLoss = reconstructionError(testData)
correct = sum(l > THRESHOLD for l in testLoss)
print(f'Correct anomaly predictions: {correct}/{len(testData)}')