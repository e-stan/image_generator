import numpy as np
import urllib.request
from PIL import Image
import pandas as pd
import os
import uuid
import sys
import matplotlib.pyplot as plt

def downloadImages(urls,size,datadir,names=None):
    if type(names) == type(None):
        names = [str(x) + ".jpg" for x in range(len(urls))]
    names = [datadir + name for name in names]
    goodNames = []
    for url,name in zip(urls,names):
        try:
            urllib.request.urlretrieve(url, name)
            image = Image.open(name)
            image = image.resize(size, Image.ANTIALIAS)
            image.save(name)
            goodNames.append(name)
        except:
            pass

    return goodNames

def parseUrls(filename):
    df = pd.read_csv(filename,sep="\t",dtype=str,header=None)
    urls = df.loc[:,14].values
    return  urls

def download_all_iamges_from_tsv(fn,size,datadir):
    names = downloadImages(parseUrls(fn),size,datadir)
    return names

def getTrainingTensor(names):
    tensor = []
    for name in names:
        try:
            image = readImage(name)
            if len(tensor) == 0 or image.shape == tensor[-1].shape:
                tensor.append(image)
        except:
           os.remove(name)
    return np.array(tensor)

def saveImage(image,fn):
    image = image * 255.0
    Image.fromarray(image.astype(np.uint8)).save(fn)

def readImage(fn):
    image = np.asarray(Image.open(fn)).astype("float32") / 255.0
    return image

def enhanceImage(image,enhancement = 2,method = "NN"):
    if method == "NN":
        change = "_ne2x.jpg"
        fn = str(uuid.uuid1()) + ".jpg"
        saveImage(image,fn)
        dir = os.path.dirname(os.path.realpath(__file__))
        enh = 1.01
        while enh < enhancement:
            os.system("python3 " + dir + "/enhance.py --zoom=2 " + fn)
            os.remove(fn)
            fn = fn.replace(".jpg",change)
            enh = 2 * enh
            print("enhanced 2x")

        image = readImage(fn)
        os.remove(fn)
    elif method == "interp":
        originalDim = image.shape
        newDim = (int(enhancement) * originalDim[0],int(enhancement) * originalDim[1])
        image = image * 255
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize(newDim)
        image = np.asarray(image) / 255

    return image


