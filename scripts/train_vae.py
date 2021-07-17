#%%
from dataset_helper import *
from imageVAE import *
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    names = ["../data/abstract_images/" + x for x in os.listdir("../data/abstract_images/") if ".jpg" in x]

    tensor = getTrainingTensor(names)

    tensor = tensor.astype("float32") / 255

    vae = ImageVAE((128,128,3),[16,32],[32,16],latent_dim=100,stride=2)

    vae.compile(optimizer=keras.optimizers.Adam())

    vae.fit(tensor, epochs=5,batch_size=16)

    vae.save_weights("../models/abstract_vae_100")





