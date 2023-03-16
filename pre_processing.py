import numpy as np
import pandas as pd
import os
import shutil as sh

initial_dir = './datasets/'
pre_processed_dir = './datasets/PROCESSED'

datasets = ['ARIA','RFMiD','STARE']

def list_files(path):

    files = os.listdir(path)
    return np.asarray(files)

def ARIA_preprocess():
    
    path_diabetes = os.path.join(initial_dir, f'{dataset}/diabetes/markups')
    path_control = os.path.join(initial_dir, f'{dataset}/controle/markups')

    images_diabetes = list_files(path_diabetes)
    images_controle = list_files(path_control)
    
    random_split_d = np.random.choice([0,1], size=len(images_diabetes), p=[.3, .7])
    random_split_c = np.random.choice([0,1], size=len(images_controle), p=[.3, .7])

    for i, image in enumerate(images_diabetes):

        image_path = os.path.join(path_diabetes, image)

        test_path_diabete = os.path.join(pre_processed_dir, 'TESTE/1/')
        train_path_diabete = os.path.join(pre_processed_dir, 'TREINO/1/')

        if random_split_d[i] == 0:
            sh.copy(image_path, test_path_diabete)
        elif random_split_d[i] == 1:
            sh.copy(image_path, train_path_diabete)

    for i, image in enumerate(images_controle):

        image_path = os.path.join(path_control, image)

        test_path_control = os.path.join(pre_processed_dir, 'TESTE/0/')
        train_path_control = os.path.join(pre_processed_dir, 'TREINO/0/')

        if random_split_c[i] == 0:
            sh.copy(image_path, test_path_control)
        elif random_split_c[i] == 1:
            sh.copy(image_path, train_path_control)
            
def RFMiD_preprocess():
    
    dataset = 'RFMiD'
    parts = ['parte_1','parte_2','parte_3']

    for part in parts:
        path_images = os.path.join(initial_dir, f'{dataset}/{part}/')
        path_labels = os.path.join(initial_dir, f'{dataset}/{part}_label.csv')

        images_path = list_files(path_images)

        labels = pd.read_csv(path_labels)

        diabet_filter = labels[labels['DR'] == 1]
        control_filter = labels[labels['Disease_Risk'] == 0] 

        random_split_d = np.random.choice([0,1], size=len(diabet_filter), p=[.3, .7])
        random_split_c = np.random.choice([0,1], size=len(control_filter), p=[.3, .7])

        test_path_diabete = os.path.join(pre_processed_dir, f'TESTE/1/')
        train_path_diabete = os.path.join(pre_processed_dir, f'TREINO/1/')

        for i, r in enumerate(random_split_d):

            image_id = diabet_filter.iloc[i].ID
            image_path = os.path.join(path_images, f'{image_id}.png')


            if r == 0:
                sh.copy(image_path, f'{test_path_diabete}/{part}_{image_id}.png')
            elif r == 1:
                sh.copy(image_path, f'{train_path_diabete}/{part}_{image_id}.png')


        test_path_control = os.path.join(pre_processed_dir, f'TESTE/0/')
        train_path_control = os.path.join(pre_processed_dir, f'TREINO/0/')

        for i, r in enumerate(random_split_c):

            image_id = control_filter.iloc[i].ID
            image_path = os.path.join(path_images, f'{image_id}.png')

            if r == 0:
                sh.copy(image_path, f'{test_path_control}/{part}_{image_id}.png')
            elif r == 1:
                sh.copy(image_path, f'{train_path_control}/{part}_{image_id}.png')
                
def STARE_preprocess():
    dataset = 'STARE'

    path_images = os.path.join(initial_dir, f'{dataset}/markups/')
    labels = pd.read_csv(f"./datasets/{dataset}/tag_stare.txt", sep='\t', header=None)

    labels = labels[labels.columns[[0,1]]]

    labels['flag_diagnoses'] = np.where(labels[1].str.contains('Diabetic', regex=True), '1', np.where(labels[1].str.contains("Normal", regex=True), '0', None))
    labels.columns = ['id', 'diagnoses', 'flag_diagnoses']

    diabet_filter = labels[labels['flag_diagnoses'] == '1']
    control_filter = labels[labels['flag_diagnoses'] == '0'] 

    random_split_d = np.random.choice([0,1], size=len(diabet_filter), p=[.3, .7])
    random_split_c = np.random.choice([0,1], size=len(control_filter), p=[.3, .7])

    test_path_diabete = os.path.join(pre_processed_dir, f'TESTE/1/')
    train_path_diabete = os.path.join(pre_processed_dir, f'TREINO/1/')

    for i, r in enumerate(random_split_d):

        image_id = diabet_filter.iloc[i].id
        image_path = os.path.join(path_images, f'{image_id}.ppm')

        if r == 0:
            sh.copy(image_path, f'{test_path_diabete}/{dataset}_{image_id}.ppm')
        elif r == 1:
            sh.copy(image_path, f'{train_path_diabete}/{dataset}_{image_id}.ppm')

    test_path_control = os.path.join(pre_processed_dir, f'TESTE/0/')
    train_path_control = os.path.join(pre_processed_dir, f'TREINO/0/')

    for i, r in enumerate(random_split_c):

        image_id = diabet_filter.iloc[i].id
        image_path = os.path.join(path_images, f'{image_id}.ppm')

        if r == 0:
            sh.copy(image_path, f'{test_path_control}/{dataset}_{image_id}.ppm')
        elif r == 1:
            sh.copy(image_path, f'{train_path_control}/{dataset}_{image_id}.ppm')
            

def prepare_data():
    '''
    Função que chama os processos de ajustes da base para inicio do modelo.
    
    As imagens são separadas em treino e teste e são colocadas em suas respectivas classificações (diabete 1 e normal 0)
    
    '''
    for dataset in datasets:
        
        if dataset == 'ARIA':
            ARIA_preprocess()
                    
        if dataset == 'RFMiD':
            RFMiD_preprocess()
            
        if dataset == 'STARE':
            STARE_preprocess()