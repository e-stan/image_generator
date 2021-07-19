from dataset_helper import *

fn = "../data/dog_images/imgs_vids_dog_None_eda39a3ee_e_02_02.tsv"
datadir = "../data/dog_images/"
size = (256,256)

download_all_iamges_from_tsv(fn,size,datadir)
