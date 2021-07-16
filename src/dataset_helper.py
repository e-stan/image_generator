import numpy as np
import urllib.request
from PIL import Image
import pandas as pd
import os

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
            image = np.asarray(Image.open(name))
            if len(image.shape) == 3:
                tensor.append(image)
        except:
            os.remove(name)
    return np.array(tensor)

