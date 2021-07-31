import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import scipy.stats as stats
import numpy as np
import tensorflow.keras as keras


class ImageGAN(keras.Model):

    def __init__(self,data_dim,arch,desc_arch,kernelsize = 3,dropout=.3,latent_size=1000,stride=2,activation = "relu",**kwargs):
        super(ImageGAN, self).__init__(**kwargs)
        self.data_dim = data_dim
        self.arch = arch
        self.kernelsize = kernelsize
        self.activation = activation
        self.stride = stride
        self.latent_size = latent_size

        generatorInput = keras.Input(shape=self.latent_size)

        initialDim = int(data_dim[0] / (self.stride**len(arch)))


        x = layers.Dense(initialDim*initialDim*arch[0], activation=self.activation,use_bias=True)(generatorInput)
        x = layers.Reshape((initialDim, initialDim, arch[0]))(x)

        if len(self.arch) > 1:
            for d in self.arch:
                x = layers.Conv2DTranspose(d, self.kernelsize, activation=self.activation, strides= self.stride, padding="same",use_bias=False)(x)

        output = layers.Conv2DTranspose(self.data_dim[-1], self.kernelsize, strides= 1,activation="sigmoid", padding="same")(x)

        generator = keras.Model(generatorInput, output, name="generator")
        generator.summary()

        self.generator = generator

        descriminatorInput = keras.Input(shape=data_dim)

        x = layers.Conv2D(desc_arch[0], self.kernelsize,strides=self.stride, padding='same',activation=self.activation,use_bias=False)(descriminatorInput)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)

        for a in desc_arch[1:]:
            x = layers.Conv2D(a,self.kernelsize, strides=self.stride, padding='same',activation=self.activation)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(dropout)(x)

        x = layers.Flatten()(x)
        output = layers.Dense(1,activation="sigmoid")(x)

        self.desrciminator = keras.Model(descriminatorInput, output, name="descriminator")
        self.desrciminator.summary()

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
        _,_,latentReg = np.array(self.encoder.predict(tensor))
        self.latentSpaceDist = [stats.norm.fit(latentReg[:,x]) for x in range(self.latent_dim)]


    def generate_images(self,n=1,fixed_vars = {}):
        latent_images = []
        for _ in range(n):
            latent_images.append([stats.norm.rvs(*x) for x in self.latentSpaceDist])
        latent_images = np.array(latent_images)
        for ind,val in fixed_vars.items():
            latent_images[:,ind] = val
        return self.decoder.predict(np.array(latent_images))

    def call(self,x):
        _,_,encoded = self.encoder(x)
        #print(len(encoded),len(encoded[0]))#.shape)
        decoded = self.decoder(encoded)
        return decoded

    def encode_decode(self,x):
        return self.decoder.predict(self.encoder.predict(x)[2])


