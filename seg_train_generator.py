import os
import random
import pandas as pd
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import data_augmentation

def create_df(image_dir, label_dir, shuffle=True): 
      
    path=image_dir + '/**/*'     
    image_file_paths=glob.glob(path,recursive=True)  
      
    path=label_dir + '/**/*'    
    label_file_paths=glob.glob(path,recursive=True)   
    
    # run a check and make sure filename without extensions match
    df = pd.DataFrame({'image': image_file_paths, 'label':label_file_paths}).astype(str)
    
    if shuffle: 
        df = df.sample(frac=1.0, replace=False, weights=None, random_state=123, axis=0).reset_index(drop=True)     
        
    return df

class jpgen():
    
    batch_index=0  #tracks the number of batches generated  
    def __init__(self, df, train_split=None, test_split=None):
                 
        self.train_split = train_split  # float between 0 and 1 indicating the percentage of images to use for training
        self.test_split = test_split 
               
        self.df = df.copy() # create a copy of the data frame
        
        if self.train_split != None: # split the df to create a training df
            self.train_df, dummy_df = train_test_split(self.df, train_size=self.train_split, shuffle=False)
            
            if self.test_split != None: # create as test set and a validation set
                
                t_split=self.test_split/(1.0-self.train_split)
                
                self.test_df, self.valid_df=train_test_split(dummy_df, train_size=t_split, shuffle=False)
                
                self.valid_gen_len=len(self.valid_df['image'].unique())# create var to return no of samples in valid generator
                
                self.valid_gen_filenames=list(self.valid_df['image'])# create list ofjpg file names in valid generator
                
            else: self.test_df = dummy_df
            
            self.test_gen_len=len(self.test_df['image'].unique())#create var to return no of test samples
            self.test_gen_filenames=list(self.test_df['image']) # create list to return jpg file paths in test_gen
            
        else:
            self.train_df=self.df  
            
        self.tr_gen_len=len(self.train_df['image'].unique())  # crete variable to return no of samples in train generator
    
    def flow(self, random_factor=0.7, batch_size=32, image_shape=None, rescale=None, shuffle=True, subset=None, augment=True): 
        # flows batches of jpg images and png masks to model.fit
        self.batch_size = batch_size
        self.image_shape = image_shape        
        self.shuffle = shuffle 
        self.subset = subset
        self.rescale = rescale  
        self.augment = augment 
        self.random_factor = random_factor
        
        image_batch_list=[] # initialize list to hold a batch of jpg  images
        label_batch_list=[] # initialize list to hold batches of png masks 
        
        if self.subset=='training' or self.train_split ==None:
            op_df=self.train_df
        elif self.subset=='test':
            op_df=self.test_df
        else:
            op_df=self.valid_df
            
        if self.shuffle : # shuffle  the op_df then rest the index            
            op_df=op_df.sample(
                frac=1.0, 
                replace=False, 
                weights=None, 
                random_state=123, 
                axis=0).reset_index(drop=True) 
        #op_df will be either train, test or valid depending on subset
        # develop the batch of data
        while True:
            label_batch_list = []
            image_batch_list = []
            start = jpgen.batch_index * self.batch_size # set start value of iteration        
            end = start + self.batch_size   # set end value of iteration to yield 1 batch of data of length batch_size
            sample_count = len(op_df['image'])            
            
            for i in range(start, end): # iterate over one batch size of data
                j=i % sample_count # used to roll the images  back to the front if the end is reached
                k=j % self.batch_size         
                       
                path_to_image = op_df.iloc[j]['image']
                path_to_label = op_df.iloc[j]['label']      
                          
                label_image = cv2.imread(path_to_label, -1) # read unchanged to preserve 4 th channel                print (png_image.)
                label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB) 
                               
                image_image = cv2.imread(path_to_image)
                image_image = cv2.cvtColor(image_image, cv2.COLOR_BGR2RGB)
                
                label_image = cv2.resize(label_image, self.image_shape)                
                image_image = cv2.resize(image_image, self.image_shape)
                
                sigmaX = 10
                image_image = cv2.addWeighted(image_image, 2, cv2.GaussianBlur(image_image, (7,7), sigmaX), -2, 128)
                
                if rescale !=None:
                    label_image = label_image * self.rescale
                    image_image = image_image * self.rescale 
                
                flag_aug = random.uniform(0, 1)
                
                if self.augment:
                    if flag_aug > ( 1 - self.random_factor):
                        image_image, label_image = data_augmentation.random_augmentation(image_image, label_image)
                               
                label_batch_list.append(label_image)
                image_batch_list.append(image_image)
                
            image_array=np.array(image_batch_list) 
            label_array=np.array(label_batch_list)
                        
            jpgen.batch_index +=1            
            
            yield (image_array, label_array)