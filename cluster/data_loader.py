import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd
import h5py
import scipy.sparse as sp


def get_loader(data_path, batch_size, mode='train'):
    """Build and return data loader."""

    dataset = KDD99Loader(data_path, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

class MyDataset(Dataset):
    def __init__(self, path,SGGpath='avenue2_train_-SGG.h5',transform=None, target_transform=None,):
        roidb=h5py.File(path+SGGpath,'r')
        self.img_to_first_box=roidb['img_to_first_box']
        self.img_to_last_box=roidb['img_to_last_box']

        # vertdb=h5py.File(path+vertpath,'r')
        # self.vertfeature=vertdb['vert_feat']

        self.imgs = self.img_to_first_box
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):


        num_rois = math.ceil((self.img_to_last_box[index]+1-self.img_to_first_box[index])/10)
        first_img_to_first_box=self.img_to_first_box[index]
        first_img_to_last_box=int(self.img_to_first_box[index]+num_rois)
        return first_img_to_first_box, first_img_to_last_box,num_rois,index

    def __len__(self):
        return len(self.imgs)
class MyDataset_test(Dataset):
    def __init__(self, path,SGGpath='avenue_test_-SGG.h5',transform=None, target_transform=None,):
        roidb=h5py.File(path+SGGpath,'r')
        self.img_to_first_box=roidb['img_to_first_box']
        self.img_to_last_box=roidb['img_to_last_box']


        self.imgs = self.img_to_first_box
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):



        num_rois = math.ceil((self.img_to_last_box[index]+1-self.img_to_first_box[index])/10)
        first_img_to_first_box=self.img_to_first_box[index]
        first_img_to_last_box=int(self.img_to_first_box[index]+num_rois)
        return first_img_to_first_box, first_img_to_last_box,num_rois,index

    def __len__(self):
        return len(self.imgs)
class MyDataset_total(Dataset):
    def __init__(self, path,SGGpath='avenue_total_-SGG.h5',transform=None, target_transform=None,):
        roidb=h5py.File(path+SGGpath,'r')
        self.img_to_first_box=roidb['img_to_first_box']
        self.img_to_last_box=roidb['img_to_last_box']


        self.imgs = self.img_to_first_box
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        num_rois = math.ceil((self.img_to_last_box[index]+1-self.img_to_first_box[index])/10)
        first_img_to_first_box=self.img_to_first_box[index]
        first_img_to_last_box=int(self.img_to_first_box[index]+num_rois)
        return first_img_to_first_box, first_img_to_last_box,num_rois,index

    def __len__(self):
        return len(self.imgs)
def my_get_loader(data_path="../data/avenue/", batch_size, mode='train'):
    """Build and return data loader."""
    shuffle = False
    if mode == 'train':
        dataset = MyDataset()
        shuffle = True
    elif mode=='test':
        dataset = MyDataset_test(data_path)
    else:
        dataset = MyDataset_total(data_path)

    data_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)  
    return data_loader