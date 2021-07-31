import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import scipy.stats as stats
import numpy as np
import tensorflow.keras as keras


class ImageGAN(keras.Model):

    def __init__(self,data_dim,arch,desc_arch,batchsize=32,kernelsize = 3,dropout=.3,latent_size=1000,stride=2,activation = "relu",**kwargs):
        super(ImageGAN, self).__init__(**kwargs)
        self.data_dim = data_dim
        self.arch = arch
        self.kernelsize = kernelsize
        self.activation = activation
        self.stride = stride
        self.latent_size = latent_size
        self.batchsize = batchsize

        generatorInput = keras.Input(shape=self.latent_size)

        initialDim = int(data_dim[0] / (self.stride**len(arch)))

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

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

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


    @property
    def metrics(self):
        return [
            self.total_loss_tracker
        ]

    def train_step(self,images):

        noise = tf.random.normal([self.batchsize, self.latent_size])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.desrciminator(images, training=True)
            fake_output = self.desrciminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.desrciminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.desrciminator.trainable_variables))

        self.total_loss_tracker.update_state(gen_loss + disc_loss)

        return {
            "loss": self.total_loss_tracker.result()
        }

    def generate_images(self,n=1,fixed_vars = {}):
        latent_images = []
        for _ in range(n):
            latent_images.append([stats.norm.rvs(*x) for x in self.latentSpaceDist])
        latent_images = np.array(latent_images)
        for ind,val in fixed_vars.items():
            latent_images[:,ind] = val
        return self.decoder.predict(np.array(latent_images))



