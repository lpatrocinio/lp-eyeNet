from keras.layers import GlobalMaxPooling2D,multiply,Conv2D
from keras.models import Sequential
from keras.layers import Input,BatchNormalization,Activation,Reshape,Lambda
from keras.layers import Concatenate,AveragePooling2D,Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import Dense
from keras.applications import InceptionResNetV2
from keras.layers import MaxPooling2D
from keras.utils import get_file

from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import tensorflow as tf
import os
import config

WEIGHT_NAME="inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
INITIAL_WEIGHT_PATH=config.output_path()

#weights=os.path.join(INITIAL_WEIGHT_PATH,WEIGHT_NAME)

def pre_processing_image_layer():
    
    layer = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
        ]
    )
    
    resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.Resizing(256, 256),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    
    return layer, resize_and_rescale



def custom_inceptionResnetV2_conv_global(input_shape=config.image_size,classes=config.CLASSES,is_base_trainable=False):
    base_model=InceptionResNetV2(include_top=False,input_shape = input_shape)
    print("base_model loaded")
    base_model.trainable=True
    set_trainable=False
    for layer in base_model.layers:
        if layer.name=="conv2d_203":
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False

    img_input = Input(shape = input_shape)
    x = base_model(img_input)
    x = GlobalAveragePooling2D(name='gl_avg_pool')(x)
    x = Dense(2, activation='softmax', name='fc1_3')(x)

    model = Model(img_input, x)

    return model

def custom_inceptionResnetV3_conv_global(input_shape=config.image_size,classes=config.CLASSES,is_base_trainable=False):
    
    base_model = PTModel(include_top=False,input_shape = input_shape)
    print("base_model loaded")
    
    base_model.trainable=True
    set_trainable=False
    
    for layer in base_model.layers:
        if layer.name=="conv2d_203":
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False

    img_input = Input(shape = input_shape)
    x = base_model(img_input)
    x = GlobalAveragePooling2D(name='gl_avg_pool')(x)
    x = Dense(2, activation='softmax', name='fc1_3')(x)

    model = Model(img_input, x)

    return model

def custom_inceptionV3_attention(input_shape=config.image_size):

    img_input = Input(shape = input_shape)
    
    base_pretrained_model = PTModel(input_shape = input_shape, include_top = False, weights = 'imagenet')
    base_pretrained_model.trainable = False
    
    # pt_depth = base_pretrained_model.layers[0].compute_output_shape(input_shape=input_shape)[-1]
    # pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
    pt_depth = base_pretrained_model.layers[-1].output_shape[-1]
    pt_features = base_pretrained_model(img_input)
    
    bn_features = BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off

    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(1, 
                        kernel_size = (1,1), 
                        padding = 'valid', 
                        activation = 'sigmoid')(attn_layer)
    
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D(name='feature_output')(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
    
    out_layer = Dense(2, activation = 'softmax')(dr_steps) #2 is the number os classes
    
    retina_model = Model(inputs = [img_input], outputs = [out_layer])
    
    from keras.metrics import top_k_categorical_accuracy
    
    def top_2_accuracy(in_gt, in_pred):
        return top_k_categorical_accuracy(in_gt, in_pred, k=2)

    retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                            metrics = ['categorical_accuracy', top_2_accuracy])
    retina_model.summary()
    
    return retina_model

def custom_inceptionV2_attention(input_tensor, input_shape=config.image_size):
    
    base_model=InceptionResNetV2(include_top=False,input_shape = input_shape)
    
    print("base_model loaded")
    base_model.trainable=True
    set_trainable=False
    
    for layer in base_model.layers:
        if layer.name=="conv2d_203":
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False

    img_input = input_tensor
    x = base_model(input_tensor)

    model = Model(img_input, x)
    
    pt_depth = base_model.layers[-1].output_shape[-1]
    pt_features = base_model(img_input)
    
    bn_features = BatchNormalization()(pt_features)

    # here we do an attention mechanism to turn pixels in the GAP on an off

    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
    attn_layer = Conv2D(1, 
                        kernel_size = (1,1), 
                        padding = 'valid', 
                        activation = 'sigmoid')(attn_layer)
    
    # fan it out to all of the channels
    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
                activation = 'linear', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features)
    gap_mask = GlobalAveragePooling2D()(attn_layer)
    
    # to account for missing values from the attention model
    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.25)(gap)
    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu', name='out_inception')(gap_dr))
    
    out_layer = Dense(2, activation = 'softmax')(dr_steps) #2 is the number os classes
    
    retina_model = Model(inputs = [img_input], outputs = [out_layer])
    
    from keras.metrics import top_k_categorical_accuracy
    
    def top_2_accuracy(in_gt, in_pred):
        return top_k_categorical_accuracy(in_gt, in_pred, k=2)

    #retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
    #                       metrics = ['categorical_accuracy', top_2_accuracy])
    #retina_model.summary()

    return retina_model

def model_segmentation(input_tensor):
    
    seed = 42
    np.random.seed = seed

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3

    do = 0.1
    iteration = 2
    kernel = 32

    #Build the model
    #inputs = tf.keras.layers.Input(input)
    s = tf.keras.layers.Lambda(lambda x: x / 255)(input_tensor)

    
    #Contraction path
    c1 = tf.keras.layers.Conv2D(kernel, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)
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
    pt_conv3b = tf.keras.layers.Conv2D(32 * 4, (3, 3), padding='same', name='output_feature')
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
    
    model = tf.keras.Model(inputs=[input_tensor], outputs=[seg_final_out])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model