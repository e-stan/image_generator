from dataset_helper import *
datadir = "../data/dog_images/"
names = [datadir + x for x in os.listdir(datadir) if ".jpg" in x]
cleanImageFiles(names)
