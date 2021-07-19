from dataset_helper import *

fn = "../data/skyline_images/imgs_vids_skyline_None_eda39a3ee_e_02_02.tsv"
datadir = "../data/skyline_images/"
size = (256,256)

download_all_iamges_from_tsv(fn,size,datadir)
