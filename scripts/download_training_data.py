from dataset_helper import *

fn = "../data/abstract_images/imgs_vids_abstract_None_eda39a3ee_e_02_02.tsv"
datadir = "../data/abstract_images"
size = (100,100)

download_all_iamges_from_tsv(fn,size,datadir)
