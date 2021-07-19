import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import scipy.stats as stats
import numpy as np
import tensorflow.keras as keras


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class ImageVAE(keras.Model):

    def __init__(self,data_dim,encodingConvArch,decodingConvArch,kernelsize = 3,latent_dim=2,stride=2,activation = "relu",**kwargs):
        super(ImageVAE, self).__init__(**kwargs)
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.encodingConvArch = encodingConvArch
        self.decodingConvArch = decodingConvArch
        self.kernelsize = kernelsize
        self.activation = activation
        self.stride = stride

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        print(self.data_dim)
        encoder_inputs = keras.Input(shape=self.data_dim)
        x = layers.Conv2D(self.encodingConvArch[0], self.kernelsize, strides= self.stride,activation=self.activation, padding="same",use_bias=False)(encoder_inputs)
        if len(self.encodingConvArch) > 1:
            for d in self.encodingConvArch[1:]:
                x = layers.Conv2D(d, self.kernelsize, activation=self.activation, strides= self.stride, padding="same",use_bias=False)(x)
        dimbeforeFlatten = keras.backend.int_shape(x)[1:]
        x = layers.Flatten()(x)
        #x = layers.Dense(int(10*latent_dim), activation=self.activation)(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        self.encoder = encoder

        decoder_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(dimbeforeFlatten[0] * dimbeforeFlatten[1] * decodingConvArch[0], activation=self.activation)(decoder_inputs)
        x = layers.Reshape((dimbeforeFlatten[0], dimbeforeFlatten[1], decodingConvArch[0]))(x)
        for d in self.decodingConvArch:
            x = layers.Conv2DTranspose(d, self.kernelsize, strides= self.stride,activation=self.activation, padding="same")(x)

        decoder_outputs = layers.Conv2DTranspose(self.data_dim[-1], self.kernelsize, strides= 1,activation="sigmoid", padding="same")(x)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        self.decoder = decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def fit_image_generator(self,tensor):
        latentReg = np.array(self.encoder.predict(tensor))
        self.latentSpaceDist = [stats.norm.fit(latentReg[:,x]) for x in range(self.latent_dim)]


    def generate_images(self,n=1):
        latent_images = []
        for _ in range(n):
            latent_images.append([stats.norm.rvs(*x) for x in self.latentSpaceDist])
        return self.decoder.predict(np.array(latent_images))

    def call(self,x):
        _,_,encoded = self.encoder(x)
        #print(len(encoded),len(encoded[0]))#.shape)
        decoded = self.decoder(encoded)
        return decoded


