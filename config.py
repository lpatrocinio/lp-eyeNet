import os

TRAIN_DIR = './datasets/PROCESSED/TREINO'
TESTE_DIR = './datasets/PROCESSED/TESTE'
DATA_DIR = './datasets/PROCESSED'

TRAIN_DIR_SEG = './datasets/STARE/vessel/markups/png'
LABEL_DIR_SEG = './datasets/STARE/vessel/label_vessel/png'
CLASSES = 2


image_size = (256,256,3)
image_size_gen = (256,256)
batch = 10
epochs = 50


def root_path():
    return os.path.dirname(__file__)
def output_path():
    return os.path.join(root_path(),"output")
def weight_path():
    return os.path.join(root_path(),"weight")