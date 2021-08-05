import numpy as np
import urllib.request
from PIL import Image
import pandas as pd
import os
import uuid
import sys
import matplotlib.pyplot as plt
import tensorflow.keras as keras

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

def getTrainingTensor(names,dim):
    tensor = np.zeros((len(names),dim[0],dim[1],dim[2]))
    goodInds = []
    i = 0
    for name in names:
        try:
            image = readImage(name)
            if image.shape == dim:
                goodInds.append(i)
                tensor[i] = image
                #tensor.append(image)
        except:
           os.remove(name)
        i += 1
    return tensor[goodInds]

def cleanImageFiles(names):
    goodNames = []
    for name in names:
        try:
            readImage(name)
            goodNames.append(name)
        except:
           os.remove(name)
    return goodNames

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

def getImageGenerator(datadir,subdir,dim,batch):
    return keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.0).flow_from_directory(datadir,classes = [subdir], target_size=dim[:2],color_mode="rgb",batch_size=batch,class_mode=None)


def makeMosiacImage(images,n):
    dim = images[0].shape
    mosiac = np.zeros((n * dim[0], n * dim[1], dim[2]))
    if len(images) >= n*n:
        ind = 0
        for x in range(n):
            for y in range(n):
                mosiac[x * dim[0]:(x + 1) * dim[0], y * dim[1]:(y + 1) * dim[0], :] = images[ind]
                ind += 1
    else:
        print("insufficient images provided")
    return mosiac

def meanImage(image):
    channels = image.shape[-1]
    meanImage = np.zeros(channels)
    for x in range(channels):
        meanImage[x] = np.mean(image[:,:,x])
    return meanImage

def getClosestImageToPixel(images,pixel):
    meanImages = [meanImage(x) for x in images]
    order = list(range(len(images)))
    order.sort(key=lambda x: np.sum(np.square(np.subtract(pixel,meanImages[x]))))
    return images[order[0]],order[0]

def makeMosiacImageWithBase(images,n,baseImage):
    dim = images[0].shape
    mosiac = np.zeros((n * dim[0], n * dim[1], dim[2]))
    baseImage = baseImage * 255
    baseImage = Image.fromarray(baseImage.astype(np.uint8))
    baseImage = baseImage.resize((n,n))
    baseImage = np.asarray(baseImage) / 255

    if len(images) >= n*n:
        for x in range(n):
            for y in range(n):
                pixel = baseImage[x,y,:]
                closestImage,ind = getClosestImageToPixel(images,pixel)
                images = [images[i] for i in range(len(images)) if i != ind]
                mosiac[x * dim[0]:(x + 1) * dim[0], y * dim[1]:(y + 1) * dim[0], :] = closestImage
    else:
        print("insufficient images provided")
    return mosiac

