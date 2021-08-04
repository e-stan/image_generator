import tensorflow.keras.layers as layers
import tensorflow as tf
import scipy.stats as stats
import numpy as np
import tensorflow.keras as keras

class Generator():
    def __init__(self,data_dim,arch,kernelsize = 3,latent_size=1000,stride=2):

        self.latent_size = latent_size

        self.data_dim = data_dim

        generatorInput = keras.Input(shape=latent_size)

        initialDim = int(data_dim[0] / (stride ** len(arch)))

        x = layers.Dense(initialDim * initialDim * arch[0], use_bias=True)(generatorInput)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Reshape((initialDim, initialDim, arch[0]))(x)

        if len(arch) > 1:
            for d in arch:
                x = layers.Conv2DTranspose(d, kernelsize, strides=stride,padding="same", use_bias=False)(x)
                x = layers.LeakyReLU(alpha=0.2)(x)

        output = layers.Conv2DTranspose(data_dim[-1], kernelsize, strides=1, activation="sigmoid",
                                        padding="same")(x)

        generator = keras.Model(generatorInput, output, name="generator")

        generator.summary()

        #generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))

        self.generator = generator

    def sampleLatentSpace(self,n):
        return tf.random.normal([n, self.latent_size])

    def generateImages(self,n):
        return self.generator.predict(self.sampleLatentSpace(n))


class Discriminator():

    def __init__(self,data_dim,desc_arch,kernelsize = 3,stride=2):
        descriminatorInput = keras.Input(shape=data_dim)

        x = layers.Conv2D(desc_arch[0], kernelsize, strides=stride, padding='same', use_bias=False)(descriminatorInput)
        x = layers.LeakyReLU(alpha=.2)(x)

        for a in desc_arch[1:]:
            x = layers.Conv2D(a, kernelsize, strides=stride, padding='same', use_bias=False)(x)
            x = layers.LeakyReLU(alpha=.2)(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(.4)(x)
        output = layers.Dense(1, activation="sigmoid")(x)

        discriminator = keras.Model(descriminatorInput, output, name="discriminator")
        discriminator.summary()

        #discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

        self.discriminator = discriminator
        self.data_dim = data_dim




def computeLoss(y1,y2):
    return tf.metrics.binary_crossentropy(y1,y2,from_logits=False)



class ImageGAN(keras.Model):

    def __init__(self,generator,discriminator,batch_size=16,opt=keras.optimizers.Adam(lr=0.0002, beta_1=0.5),**kwargs):
        super(ImageGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        #self.discriminator.discriminator.trainable = False
        #self.gan = keras.Sequential()
        #self.gan.add(generator.generator)
        #self.gan.add(discriminator.discriminator)
        self.batch_size = batch_size
        self.opt = opt

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


    def train_step(self,images):

        latentPoints = self.generator.sampleLatentSpace(self.batch_size)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            genImages = self.generator.generator(latentPoints,training=True)
            real_output = self.discriminator.discriminator(images, training=True)
            fake_output = self.discriminator.discriminator(genImages, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_dis = disc_tape.gradient(disc_loss,self.discriminator.discriminator.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_dis, self.discriminator.discriminator.trainable_variables))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.generator.trainable_variables)
        self.opt.apply_gradients(zip(gradients_of_generator, self.generator.generator.trainable_variables))

        return {
            "des loss": disc_loss,
            "gen loss": gen_loss
        }

    def call(self,inputs):
        return self.gan(self.generator.sampleLatentSpace(1))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)


