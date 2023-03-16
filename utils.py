import os

import config

from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.models import model_from_json
from PIL import Image
import numpy as np
from skimage import transform
import config
import cv2

def load(filename):
   np_image = Image.open(filename)
   np_image = gaussian_filter(np_image)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, config.image_size)
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def gaussian_filter(image):
    sigmaX = 10
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)

def total_count_files(dir):
    """ count all files in a directory recursively
    """
    total = 0
    for root, dirs, files in os.walk(dir):
        total += len(files)
    return total

def set_early_stopping(monitor):
    return EarlyStopping(monitor=monitor,
                         patience = 5,
                         mode = "auto",
                         verbose = 2)
    
class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        self.val_losses=[]
        
    def on_epoch_end(self,batch,logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        
def set_reduce_lr():
    return ReduceLROnPlateau(monitor='val_loss',
                             factor = 0.5,
                             patience = 4,
                             min_lr = 1e-5)
    

def save_model(model):
    """
    save model to output directory
    """
    model_json=model.to_json()
    #write
    with open(os.path.join(config.output_path(), 'teste_1.json'), "w") as json_file:
        json_file.write(model_json)
        
    print("model saved")

    #save the weight
    #serialize weights to HDF5

    model.save_weights(os.path.join(config.output_path(),'teste'))
    print("weight saved") 
    
def convert_ppm_to_png(src_folder, dest_folder):
    for filename in os.listdir(src_folder):
        if filename.endswith('.ppm'):
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename.replace('.ppm', '.png'))
            img = Image.open(src_path)
            img.save(dest_path)
            print(f'Converted {src_path} to {dest_path}')
