import cv2
import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def loading_data():
    # First, loading data from Skin MNIST dataset
    data_dir = "/home/crb/文档/working_ljx/datasets/HAM10000"
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    # norm_mean, norm_std = compute_img_mean_std(all_image_path)
    norm_mean, norm_std = [0.62485385, 0.62214285, 0.62006605], [0.17797484, 0.18013163, 0.18257399]

    # 在原始数据aframe中添加三列:path(图像路径)、cell_type(全称)、
    # cell_type_idx(单元格类型的对应索引，作为图像标签)

    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    print('\n')
    print(df_original['duplicates'].value_counts())

    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    # This set will be df_original excluding all rows that are in the val set
    # This function identifies if an image is part of the train or val set.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    print(df_train['cell_type'].value_counts())
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(df_val['cell_type'].value_counts())


    # Copy fewer class to balance the number of 7 classes
    # 把对应标签的样本相应的复制15,10.。。。倍
    data_aug_rate = [15, 10, 5, 50, 0, 40, 5]
    for i in range(7):
        if data_aug_rate[i]:
            df_train = df_train.append([df_train.loc[df_train['cell_type_idx'] == i, :]] * (data_aug_rate[i] - 1), ignore_index=True)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Data-Proceeding!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(df_train['cell_type'].value_counts())

    df_train = df_train.reset_index()
    df_val = df_val.reset_index()

    return df_train, df_val


# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.62485385, 0.62214285, 0.62006605], [0.17797484, 0.18013163, 0.18257399])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.62485385, 0.62214285, 0.62006605], [0.17797484, 0.18013163, 0.18257399]),
                                        
                                        ])
