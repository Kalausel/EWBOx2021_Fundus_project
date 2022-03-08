# -*- coding: utf-8 -*-
"""
The DataPrep class contains methods to read in images and labels from a drive
and prepare dataloaders that can be used to pass the data to a model.
"""

import os
from datetime import datetime
import torch.utils.data
import torchvision.io
import pandas as pd
from keywords2vec import keywords2vec

def remove_suffix(input_string, suffix):
    '''
    str.removesuffix is only available in Python 3.9+; make my own
    '''
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def listdirexcept(directory, exceptions):
    '''
    Returns the dir's content without files in the list exceptions
    '''
    files=os.listdir(directory)
    for k in exceptions:
        if k in files:
            files.remove(k)
    return files

class DataPrep:
    '''
    The DataPrep class contains methods to read in images and labels from a drive
    and prepare dataloaders that can be used to pass the data to a model.
    '''
    def __init__(self, hyperpars, img_dir, labels_path, transform=None,
                 target_transform=None, flip_right=False, skip_images=[],
                 label_multiplier=1):
        self.img_dir=img_dir
        self.labels_path=labels_path
        self.transform=transform
        self.target_transform=target_transform
        self.skip_images=skip_images
        self.flip_right=flip_right
        self.hyperpars=hyperpars
        # Extract relevant labels data and put into keywords dataframe
        df=pd.read_csv(self.labels_path)
        keywords=df[['ID','Left-Diagnostic Keywords','Right-Diagnostic Keywords']].set_index('ID')
        # The row index is now accessible via patient ID
        self.keywords=keywords
        self.label_multiplier=label_multiplier

    def data_loaders(self):
        '''
        This method returns the three dataloaders for training, validation and testing.
        '''
        # 1) Create instance of self.CustomDataset
        dataset=self.CustomDataset(self)
        # 2) Split this dataset into train, val and test
        trainset, valset, testset = torch.utils.data.random_split(dataset,
                            [self.hyperpars.train_size, self.hyperpars.val_size,
                             self.hyperpars.test_size])
        # Create three dataloaders from the datasets
        train_dataloader = torch.utils.data.DataLoader(trainset,
                            batch_size=self.hyperpars.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(valset, self.hyperpars.batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(testset, self.hyperpars.batch_size, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader

    def save_trainval_test_data_sets(self):
        '''
        This method returns two datasets, one for training, one for testing.
        It saves them into the './Datasets' folder
        '''
        # 1) Create instance of self.CustomDataset
        dataset=self.CustomDataset(self)
        # 2) Split this dataset into train, val and test
        trainvalset, testset = torch.utils.data.random_split(dataset,
                    [self.hyperpars.train_size+self.hyperpars.val_size, self.hyperpars.test_size])
        now=datetime.now().strftime("%d%m%Y_%H%M%S")
        torch.save({'train_val_dataset': trainvalset,
                    'test_dataset': testset},
                    './Datasets/datasets_'+now+'.pt')
        return trainvalset, testset

    def split_trainset_and_save_dataloaders(self, trainvalset, testset, split='automatic'):
        '''
        This method splits the training set returned by save_trainval_test_data_sets
        into training and validation, and returns and saves three dataloaders
        to be given to a model.
        '''
        if split=='automatic':
            split=[self.hyperpars.train_size, self.hyperpars.val_size]
        trainset,valset = torch.utils.data.random_split(trainvalset, split)
        train_dataloader = torch.utils.data.DataLoader(trainset,
                            batch_size=self.hyperpars.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(valset, self.hyperpars.batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(testset, self.hyperpars.batch_size, shuffle=False)
        now=datetime.now().strftime("%d%m%Y_%H%M%S")
        torch.save({'train_dataloader': train_dataloader,
                    'val_dataloader': val_dataloader,
                    'test_dataloader': test_dataloader},
                    './Dataloaders/Dataloaders_'+now+'.pt')
        return train_dataloader, val_dataloader, test_dataloader


    class CustomDataset(torch.utils.data.Dataset):
        '''
        This class implements the necessary methods to use the pytorch Dataset functionality.
        '''
        # pass instance of DataPrep (=outer class) to access its variables
        def __init__(self, outer_class):
            self.img_dir=outer_class.img_dir
            self.labels_path=outer_class.labels_path
            self.transform=outer_class.transform
            self.target_transform=outer_class.target_transform
            self.skip_images=outer_class.skip_images
            self.flip_right=outer_class.flip_right
            self.hyperpars=outer_class.hyperpars
            self.keywords=outer_class.keywords
            self.label_multiplier=outer_class.label_multiplier

        def __len__(self):
            # No of images in folder - number of images to omit
            return len(os.listdir(self.img_dir))-len(self.skip_images)

        def __getitem__(self, idx): # idx != patientID
            # Prepare the image
            img_name=listdirexcept(self.img_dir, self.skip_images)[idx]
            img_path=os.path.join(self.img_dir,img_name)
            image=torchvision.io.read_image(img_path) # Output: Tensor uint8 [0, 255]
            side=img_name
            for x in '0123456789_()':
                side=side.replace(x,'') # Remove everything except 'left' or 'right'
            side=remove_suffix(side,'.jpg')
            # Flip right image
            if self.flip_right and side=='right':
                image=torchvision.transforms.functional.hflip(image)
            if self.transform: # if self.transform != None
                image=self.transform(image) # apply this transform

            # Prepare the label
            patientID=img_name.split('_')[0]# get the patient ID = initial digits of file_name
            if side=='left':
                disease_keywords=self.keywords.at[int(patientID),'Left-Diagnostic Keywords']
            elif side=='right':
                disease_keywords=self.keywords.at[int(patientID),'Right-Diagnostic Keywords']
            else:
                raise Exception('Side of the image (right or left) was not extracted correctly.')
            label=[i*self.label_multiplier for i in keywords2vec(disease_keywords)]
            label=torch.Tensor(label)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        