#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 21:23:27 2022

@author: gerard
"""

import os
import torch
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_full_image(directory):

    images = os.listdir(directory)
    rows = [int(x.split('x')[-2].split('_')[-1]) for x in images]
    columns = [int(x.split('x')[-1].split('.')[0]) for x in images]

    max_i, max_j = max(columns)+1, max(rows)+1
    full_image = np.zeros((max_i, max_j)).tolist()

    for image in images:

        i, j = int(image.split('x')[-1].split('.')[0]), int(image.split('x')[-2].split('_')[-1])
        full_image[i][j] = cv2.imread(directory+image, cv2.IMREAD_COLOR)

    return np.vstack([np.hstack(x) for x in full_image]) #OMG this function is SICK

class SHDataset(Dataset):

    def __init__(self, csv_file = "../data/ftb_data_harassment.csv", img_dir = "../panoramics", transform = None):

        self.img_dir = os.path.abspath("../panoramics")
        self.img_files = sorted(os.listdir(self.img_dir))
        self.text_df = pd.read_csv(csv_file, index_col=0)
        self.text_df = self.text_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.img_files[idx]
        idx = int(img_name)

        # Image
        img_path = self.img_dir + '/' + img_name + '/0/'
        img = get_full_image(img_path)
        img = torch.tensor(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Text
        text = self.text_df.loc[idx, "description"]
        
        # Opposite text
        label = self.text_df.loc[idx, "target"]
        opposite_text = self.text_df[self.text_df.target != label].sample()["description"].item()


        return (img, text, opposite_text)
    

    
    
    
    
    
    
    
    
    
    