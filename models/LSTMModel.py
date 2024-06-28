import numpy as np
import tensorflow as tf
import keras

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable()
class VAE(keras.Model):
  def __init__(self, input_shape, latent_dim=10):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
        
    self.encoder = keras.models.Sequential([
      keras.layers.Input(shape=input_shape),
      keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
      keras.layers.MaxPooling1D(2),
      keras.layers.Dropout(0.1),
      keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
      keras.layers.MaxPooling1D(2),
      keras.layers.Dropout(0.1),
      # LSTM layer in the encoder
      keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
      keras.layers.Dropout(0.1),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(latent_dim * 2)  # Outputs mean and log variance
    ])
    
    # Decoder
    self.decoder = keras.models.Sequential([
      keras.layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(128 * 128, activation='relu'),
      # Reshape the output for LSTM layer
      keras.layers.Reshape((128, 128)),
      # LSTM layer in the decoder
      keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
      keras.layers.Dropout(0.1),
      keras.layers.Conv1DTranspose(64, 3, activation='relu', padding='same'),
      keras.layers.Dropout(0.1),
      keras.layers.UpSampling1D(2),
      keras.layers.Conv1DTranspose(32, 3, activation='relu', padding='same'),
      keras.layers.UpSampling1D(2),
      keras.layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='same')
    ])
    
    # Classification head
    self.classifier = keras.models.Sequential([
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

  def encode(self, x):
    # Get mean and log variance from the encoder
    mean_log_var = self.encoder(x)
    mean = mean_log_var[:, :self.latent_dim]
    log_var = mean_log_var[:, self.latent_dim:]
    return mean, log_var
  
  def reparameterize(self, mean, log_var):
    # Reparameterization trick
    epsilon = tf.random.normal(shape=(tf.shape(mean)))
    z = mean + tf.exp(0.5 * log_var) * epsilon
    return z
  
  def decode(self, z):
    # Decode from latent space to reconstruct input
    reconstructed = self.decoder(z)
    return reconstructed

  def classify(self, z):
    # Predict labels
    predicted_labels = self.classifier(z)
    return predicted_labels
  
  def call(self, x):
    # Forward pass
    mean, log_var = self.encode(x)
    z = self.reparameterize(mean, log_var)
    reconstructed = self.decode(z)
    predicted_labels = self.classify(z)
    return reconstructed, predicted_labels, mean, log_var

  def get_config(self):
    config = super().get_config()
    # Update the config with the custom layer's parameters
    config.update(
      {
        "latent_dim": self.latent_dim,
        "encoder": self.encoder,
        "decoder": self.decoder,
        "classifier": self.classifier,
      }
    )
    return config

  @classmethod
  def from_config(cls, config):
      # Note that you can also use [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function) here
      config["latent_dim"] = keras.layers.deserialize(config["latent_dim"])
      config["encoder"] = keras.layers.deserialize(config["encoder"])
      config["decoder"] = keras.layers.deserialize(config["decoder"])
      config["classifier"] = keras.layers.deserialize(config["classifier"])
      return cls(**config)

def custom_loss(y_true, y_pred):
  # Unpack the predictions
  reconstructed = y_pred[0]
  predicted_labels = y_pred[1]
  mean = y_pred[2]
  log_var = y_pred[3]
  
  # Reconstruction loss (MSE)
  recon_loss = tf.reduce_mean(tf.square(y_true[0] - reconstructed))
  
  # KL loss (Kullback-Leibler divergence)
  kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
  
  # Classification loss (Binary cross-entropy)
  classification_loss = keras.losses.binary_crossentropy(y_true[1], predicted_labels)
  
  # Total loss
  total_loss = recon_loss + kl_loss + abs(classification_loss)
  
  # Remove tf.print statements and use print() for debugging
  print(f"Reconstruction loss: {recon_loss}")
  print(f"KL loss: {kl_loss}")
  print(f"Classification loss: {classification_loss}")
  print(f"Total loss: {total_loss}")
  
  return total_loss
