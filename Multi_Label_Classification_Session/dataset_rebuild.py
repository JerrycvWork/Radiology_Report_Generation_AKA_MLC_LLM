import os.path
from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
#from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
from pathlib import Path

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob

import pandas as pd




class CustomDataset_classification_train_iuxray(Dataset):

    def __init__(self, classes, level=1000, train_csv="",image_folder="IUXray/NLMCXR_png/"):

        #Finding_Anatomy_encoding.csv
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(train_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):
            case_folder=image_folder

            if int(level)>=10:
                temp_target = list(str(data_csv[str(level)+'_coding_str'][i1]))
                for s1 in range(len(temp_target)):
                    temp_target[s1] = int(temp_target[s1])
                if np.sum(np.array(temp_target)) > 0:
                    self.targets.append(np.array(temp_target))
                    self.images.append(case_folder + data_csv['Case_num'][i1])


    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target

    def __len__(self):
        return len(self.targets)





class CustomDataset_classification_test_iuxray(Dataset):

    def __init__(self, classes, level=1000, test_csv="",image_folder="IUXray/NLMCXR_png/"):

        #Finding_Anatomy_encoding.csv
        #self.image_paths = image_paths
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(test_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):
            case_folder=image_folder

            if os.path.exists(case_folder+data_csv['Case_num'][i1]+"-1001.png")>0:

                if int(level)>=10:
                    temp_target = list(str(data_csv[str(level)+'_coding_str'][i1]))
                    for s1 in range(len(temp_target)):
                        temp_target[s1] = int(temp_target[s1])
                    if np.sum(np.array(temp_target)) >= 0:
                        self.targets.append(np.array(temp_target))
                        self.images.append(case_folder + data_csv['Case_num'][i1] + "-1001.png")

            if os.path.exists(case_folder+data_csv['Case_num'][i1]+"-2001.png")>0:

                if int(level) >= 10:
                    temp_target = list(str(data_csv[str(level) + '_coding_str'][i1]))
                    for s1 in range(len(temp_target)):
                        temp_target[s1] = int(temp_target[s1])
                    if np.sum(np.array(temp_target)) >= 0:
                        self.targets.append(np.array(temp_target))
                        self.images.append(case_folder + data_csv['Case_num'][i1] + "-2001.png")

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target, self.images[index]

    def __len__(self):
        return len(self.targets)


class CustomDataset_classification_val_iuxray(Dataset):

    def __init__(self, classes, level=1000, val_csv="",image_folder="IUXray/NLMCXR_png/"):

        #Finding_Anatomy_encoding.csv
        #self.image_paths = image_paths
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(val_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):
            case_folder=image_folder

            if os.path.exists(case_folder+data_csv['Case_num'][i1]+"-1001.png")>0:

                if int(level)>=10:
                    temp_target = list(str(data_csv[str(level)+'_coding_str'][i1]))
                    for s1 in range(len(temp_target)):
                        temp_target[s1] = int(temp_target[s1])
                    if np.sum(np.array(temp_target)) >= 0:
                        self.targets.append(np.array(temp_target))
                        self.images.append(case_folder + data_csv['Case_num'][i1] + "-1001.png")

            if os.path.exists(case_folder+data_csv['Case_num'][i1]+"-2001.png")>0:

                if int(level) >= 10:
                    temp_target = list(str(data_csv[str(level) + '_coding_str'][i1]))
                    for s1 in range(len(temp_target)):
                        temp_target[s1] = int(temp_target[s1])
                    if np.sum(np.array(temp_target)) >= 0:
                        self.targets.append(np.array(temp_target))
                        self.images.append(case_folder + data_csv['Case_num'][i1] + "-2001.png")

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target, self.images[index]

    def __len__(self):
        return len(self.targets)


class CustomDataset_classification_train_mimic_cxr(Dataset):

    def __init__(self, classes, level=1000, train_csv="",image_folder=""):

        #Finding_Anatomy_encoding.csv
        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(train_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):

            if int(level)>=10:
                temp_target = list(str(data_csv[str(level) + '_coding_str'][i1]))
                for s1 in range(len(temp_target)):
                    temp_target[s1] = int(temp_target[s1])
                if np.sum(np.array(temp_target)) > 0:
                    self.targets.append(np.array(temp_target))
                    self.images.append(image_folder+r"/Train/" + data_csv['Case_num'][i1]+'.jpg')


    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target

    def __len__(self):
        return len(self.targets)




class CustomDataset_classification_test_mimic_cxr(Dataset):

    def __init__(self, classes, level=1000, test_csv="",image_folder=""):

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(test_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):

            if int(level) >= 10:
                temp_target = list(str(data_csv[str(level) + '_coding_str'][i1]))
                for s1 in range(len(temp_target)):
                    temp_target[s1] = int(temp_target[s1])
                if np.sum(np.array(temp_target)) >= 0:
                    self.targets.append(np.array(temp_target))
                    self.images.append(image_folder+r"/Test/" + data_csv['Case_num'][i1] + '.jpg')

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target, self.images[index]

    def __len__(self):
        return len(self.targets)



class CustomDataset_classification_val_mimic_cxr(Dataset):

    def __init__(self, classes, level=1000, val_csv="",image_folder=""):

        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(std=(0.5), mean=(0.5)),
        ])

        self.classes = classes

        self.images  = []
        self.targets = []

        data_csv = pd.read_csv(val_csv,dtype=str)

        for i1 in range(len(list(data_csv['Case_num']))):

            if int(level) >= 10:
                temp_target = list(str(data_csv[str(level) + '_coding_str'][i1]))
                for s1 in range(len(temp_target)):
                    temp_target[s1] = int(temp_target[s1])
                if np.sum(np.array(temp_target)) >= 0:
                    self.targets.append(np.array(temp_target))
                    self.images.append(image_folder+r"/Test/" + data_csv['Case_num'][i1] + '.jpg')

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)

        target = self.targets[index]

        return t_image, target, self.images[index]

    def __len__(self):
        return len(self.targets)









