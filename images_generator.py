import tensorflow as tf
import config
from model import pre_processing_image_layer

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_addons as tfa
from seg_train_generator import *


def tensorflow_addWeighted(img1, img2):
     
     img = img1 * tf.multiply(tf.ones((256, 256, 3)), 2) + img2 * tf.multiply(tf.ones((256, 256, 3)), -2)+128
     return img
 
 
def gaussian_filter(image, label):
    # Apply the Gaussian blur
    blurred_image = tfa.image.gaussian_filter2d(image, (7, 7), 10)
    result = tensorflow_addWeighted(image, blurred_image)
    
    return result, label

def visualize_dataset_inception(dataset):
    import matplotlib.pyplot as plt

    try:
        class_names=dataset.class_names
    except: 
        class_names=['0','1']
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
            
def inception_gen():
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=["0","1"],
        seed=42,
        shuffle=True,
        image_size=config.image_size_gen,
        batch_size=config.batch
        )
        
    train_ds_filter = train_ds.map(gaussian_filter)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TESTE_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=["0","1"],
        seed=42,
        shuffle=True,
        image_size=config.image_size_gen,
        batch_size=config.batch
        )
    
    val_ds_filter = val_ds.map(gaussian_filter)
    
    pre_processing_layer, resize_reescale = pre_processing_image_layer()
    
    train_dataset = train_ds_filter.map(lambda x, y: (pre_processing_layer(x), y))
    train_dataset = train_dataset.map(lambda x, y: (resize_reescale(x), y))
    val_dataset = val_ds_filter.map(lambda x, y: (resize_reescale(x), y))
    
    return train_dataset, val_dataset

def visualize_dataset_seg(dataset):
    import matplotlib.pyplot as plt

    data_set_label = dataset[1]
    data_set_train = dataset[0]
    
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(10, 20))
    for y in range(5):
        #axs[y, 0].subplot(5, 2, y+1)
        axs[y, 0].imshow(data_set_train[y])
        #plt.subplot(5, 2, y+2)
        axs[y, 1].imshow(data_set_label[y])
    
    plt.tight_layout()
    plt.show()
        
        

def seg_gen():
    
    
    image_dir = config.TRAIN_DIR_SEG
    label_dir = config.LABEL_DIR_SEG 
    shuffle = False # if True shuffles the dataframe
    df = create_df(image_dir, label_dir ,shuffle)
    
    train_split = 0.8 
    test_split = 0.1
    batch_size = 5
    height = 256
    width = 256
    channels = 3 
    image_shape = (height, width) 
    
    gen = jpgen(df, train_split=train_split, test_split=test_split)
    
    train_len = gen.tr_gen_len

    test_len = gen.test_gen_len

    valid_len = gen.valid_gen_len   
    
    # instantiate generators
    train_gen = gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=1/255, shuffle=False, subset='training', random_factor=0.7)
    valid_gen = gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=1/255, shuffle=False, subset='valid', random_factor=0.7)
    test_gen = gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=1/255, shuffle=False, subset='test', random_factor=0.7)
    
    return train_gen, test_gen, valid_gen, train_len, test_len, valid_len

        