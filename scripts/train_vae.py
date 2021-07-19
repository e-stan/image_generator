#%%
from dataset_helper import *
from imageVAE import *
import os

import matplotlib.pyplot as plt


if __name__ == "__main__":
    names = ["../data/skyline_images/" + x for x in os.listdir("../data/skyline_images/") if ".jpg" in x]
    dim = (256,256,3)
    batchsize=128
    epochs = 5
    latentDim = 100
    stride = 2
    arch = [32,64,64]

    archRev = list(arch)
    archRev.reverse()
    print(len(names))

    #names = cleanImageFiles(names)

    print(len(names))

    #tensor = getTrainingTensor(names,dim)

    vae = ImageVAE(dim,arch,archRev,latent_dim=latentDim,stride=stride)

    vae.compile(optimizer=keras.optimizers.Adam(),run_eagerly=True)

    vae.fit(getImageGenerator("../data/","skyline_images", dim, batchsize), epochs=epochs,batch_size=batchsize,)

    vae.save_weights("../models/skyline_vae_100_2")





