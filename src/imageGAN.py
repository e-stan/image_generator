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

        generator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4))

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

        discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

        self.discriminator = discriminator
        self.data_dim = data_dim


def computeLoss(y1,y2):
    return tf.metrics.binary_crossentropy(y1,y2,from_logits=False)



class ImageGAN(keras.Model):

    def __init__(self,generator,discriminator,batch_size=16,**kwargs):
        super(ImageGAN, self).__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.discriminator.trainable = False
        self.gan = keras.Sequential()
        self.gan.add(generator.generator)
        self.gan.add(discriminator.discriminator)
        self.batch_size = batch_size

    def train_step(self,images):

        latentPoints = self.generator.sampleLatentSpace(self.batch_size)

        genImages = self.generator.generator(latentPoints)

        y_images = tf.ones_like((self.batch_size,1))
        y_gen = tf.zeros_like((self.batch_size,1))


        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            y = tf.concat([y_images,y_gen],0)
            X = tf.concat([images,genImages],0)
            print(y.get_shape(),X.get_shape(),images.get_shape(),genImages.get_shape())
            print(y,self.batch_size,self.discriminator.discriminator(X))
            des_loss = computeLoss(y,self.discriminator.discriminator(X))

        gradients_of_dis = disc_tape.gradient(des_loss,self.discriminator.discriminator.trainable_variables)
        self.discriminator.discriminator.optimizer.apply_gradients(zip(gradients_of_dis, self.discriminator.discriminator.trainable_variables))
        #
        # real_loss = self.discriminator.discriminator.train_on_batch(images,y_images)
        # gen_loss = self.discriminator.discriminator.train_on_batch(genImages,y_gen)
        #
        # y_gen = np.ones((self.batch_size,1))
        #
        # gen_loss = self.gan.train_on_batch(latentPoints,y_gen)

        return {
            "des loss": des_loss,
            #"gen loss": gen_loss
        }

    def call(self,inputs):
        return self.gan(self.generator.sampleLatentSpace(1))




