import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import config
from utils import *
from keras.preprocessing.image import ImageDataGenerator
from seg_train_generator import *

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# TRAIN_PATH = 'stage1_train/'
# TEST_PATH = 'stage1_test/'

# train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]

# X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

# print('Resizing training images and masks')
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]  
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_train[n] = img  #Fill empty X_train with values from img
#     mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)  
            
#     Y_train[n] = mask   

# # test images
# X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
# sizes_test = []
# print('Resizing test images') 
# for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
#     path = TEST_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#     sizes_test.append([img.shape[0], img.shape[1]])
#     img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#     X_test[n] = img

# print('Done!')

# image_x = random.randint(0, len(train_ids))
# imshow(X_train[image_x])
# plt.show()
# imshow(np.squeeze(Y_train[image_x]))
# plt.show()


do = 0
iteration = 3
kernel = 32

#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(kernel, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(kernel, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(kernel * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(kernel * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(kernel * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(kernel * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(kernel * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(kernel * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(kernel * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(kernel * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(kernel * 8, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(kernel * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(kernel * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(kernel * 4, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(kernel * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(kernel * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(kernel * 2, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(kernel * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(kernel * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(kernel, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(kernel, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(kernel, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


pt_conv1a = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
pt_activation1a = tf.keras.layers.ReLU()
pt_dropout1a = tf.keras.layers.Dropout(do)
pt_conv1b = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
pt_activation1b = tf.keras.layers.ReLU()
pt_dropout1b = tf.keras.layers.Dropout(do)
pt_pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

pt_conv2a = tf.keras.layers.Conv2D(32 * 2, (3, 3), padding='same')
pt_activation2a = tf.keras.layers.ReLU()
pt_dropout2a = tf.keras.layers.Dropout(do)
pt_conv2b = tf.keras.layers.Conv2D(32 * 2, (3, 3), padding='same')
pt_activation2b = tf.keras.layers.ReLU()
pt_dropout2b = tf.keras.layers.Dropout(do)
pt_pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))

pt_conv3a = tf.keras.layers.Conv2D(32 * 4, (3, 3), padding='same')
pt_activation3a = tf.keras.layers.ReLU()
pt_dropout3a = tf.keras.layers.Dropout(do)
pt_conv3b = tf.keras.layers.Conv2D(32 * 4, (3, 3), padding='same')
pt_activation3b = tf.keras.layers.ReLU()
pt_dropout3b = tf.keras.layers.Dropout(do)

pt_tranconv8 = tf.keras.layers.Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')
pt_conv8a = tf.keras.layers.Conv2D(32 * 2, (3, 3), padding='same')
pt_activation8a = tf.keras.layers.ReLU()
pt_dropout8a = tf.keras.layers.Dropout(do)
pt_conv8b = tf.keras.layers.Conv2D(32 * 2, (3, 3), padding='same')
pt_activation8b = tf.keras.layers.ReLU()
pt_dropout8b = tf.keras.layers.Dropout(do)

pt_tranconv9 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
pt_conv9a = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
pt_activation9a = tf.keras.layers.ReLU()
pt_dropout9a = tf.keras.layers.Dropout(do)
pt_conv9b = tf.keras.layers.Conv2D(32, (3, 3), padding='same')
pt_activation9b = tf.keras.layers.ReLU()
pt_dropout9b = tf.keras.layers.Dropout(do)

conv9s = [c9]
outs = []
a_layers = [c1]
for iteration_id in range(iteration):
    out = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name=f'out1{iteration_id + 1}')(conv9s[-1])
    outs.append(out)

    conv1 = pt_dropout1a(pt_activation1a(pt_conv1a(conv9s[-1])))
    conv1 = pt_dropout1b(pt_activation1b(pt_conv1b(conv1)))
    a_layers.append(conv1)
    conv1 = tf.keras.layers.concatenate(a_layers, axis=3)
    conv1 = tf.keras.layers.Conv2D(32, (1, 1), padding='same')(conv1)
    pool1 = pt_pooling1(conv1)

    conv2 = pt_dropout2a(pt_activation2a(pt_conv2a(pool1)))
    conv2 = pt_dropout2b(pt_activation2b(pt_conv2b(conv2)))
    pool2 = pt_pooling2(conv2)

    conv3 = pt_dropout3a(pt_activation3a(pt_conv3a(pool2)))
    conv3 = pt_dropout3b(pt_activation3b(pt_conv3b(conv3)))

    up8 = tf.keras.layers.concatenate([pt_tranconv8(conv3), conv2], axis=3)
    conv8 = pt_dropout8a(pt_activation8a(pt_conv8a(up8)))
    conv8 = pt_dropout8b(pt_activation8b(pt_conv8b(conv8)))

    up9 = tf.keras.layers.concatenate([pt_tranconv9(conv8), conv1], axis=3)
    conv9 = pt_dropout9a(pt_activation9a(pt_conv9a(up9)))
    conv9 = pt_dropout9b(pt_activation9b(pt_conv9b(conv9)))

    conv9s.append(conv9)

seg_final_out = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='seg_final_out')(conv9)
outs.append(seg_final_out)

 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[seg_final_out])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_seg.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='accuracy'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]


# convert_ppm_to_png(config.TRAIN_DIR_SEG, f'{config.TRAIN_DIR_SEG}/png/')
# convert_ppm_to_png(config.LABEL_DIR_SEG, f'{config.LABEL_DIR_SEG}/png/')
image_dir = config.TRAIN_DIR_SEG# directory with clean images
label_dir = config.LABEL_DIR_SEG # directory with noisy images file names same as filenames in clean dir
shuffle=False # if True shuffles the dataframe
df=create_df(image_dir, label_dir ,shuffle) # create a dataframe with columns 'images' , 'labels'
                                            # where labels are the noisy images
train_split=.5 # use 80% of files for training
test_split=.2  # use 10% for test, automatically sets validation split at 1-train_split-test_split
batch_size=5 # set batch_size
height=128 # set image height for generator output images and labels
width=128 # set image width for generator output images and labels
channels=3 # set number of channel in images
image_shape=(height, width) 
rescale=1/255  # set value to rescale image pixels
gen=jpgen(df, train_split=train_split, test_split=test_split) # create instance of generator class
tr_gen_len=gen.tr_gen_len
test_gen_len= gen.test_gen_len
valid_gen_len=gen.valid_gen_len   
test_filenames=gen.test_gen_filenames # names of test file paths used for training 
train_steps=tr_gen_len//batch_size #  use this value in for steps_per_epoch in model.fit
valid_steps=valid_gen_len//batch_size # use this value for validation_steps in model.fit
test_steps=test_gen_len//batch_size  # use this value for steps in model.predict
# instantiate generators
train_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=rescale, shuffle=False, subset='training')
valid_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=rescale, shuffle=False, subset='valid')
test_gen=gen.flow(batch_size=batch_size, image_shape=image_shape, rescale=rescale, shuffle=False, subset='test')

# dataset_train = ImageDataGenerator(
#                                 rotation_range=40,
#                                 width_shift_range=0.2,
#                                 height_shift_range=0.2,
#                                 shear_range=0.2,
#                                 zoom_range=0.2,
#                                 horizontal_flip=True,
#                                 fill_mode='nearest'
#                                 )

# train_generator = dataset_train.flow_from_directory(config.TRAIN_DIR_SEG,
#                                                     target_size=(128,128,3),
#                                                     batch_size=5,
#                                                     class_mode=None)

# label_generator = dataset_train.flow_from_directory(config.LABEL_DIR_SEG,
#                                                     target_size=(128,128,3),
#                                                     batch_size=5,
#                                                     class_mode=None)
# train_generator = (np.expand_dims(x, axis=0) for x in train_generator)
# label_generator = (np.expand_dims(x, axis=0) for x in label_generator)

# train_data = zip(train_generator, label_generator)

results = model.fit(train_gen, batch_size=5, epochs=5, callbacks=callbacks, steps_per_epoch=train_steps, validation_steps=valid_steps, validation_data=valid_gen,  shuffle=True) # 20 num de fotos, 5 batch

# predictions=model.predict(test_gen, steps=test_steps)

# print(predictions)













