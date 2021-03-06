import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

#Class 14 - Stop

class GTSRB(Dataset):
    """GTSRB Image Dataset"""

    def __init__(self, root_dir, training=True, transform=None):
        self.root_dir = root_dir
        self.files = glob.glob(self.root_dir + '/*/*.ppm')
        if transform == None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.training = training
        threshold = int(len(self.files)*0.80)
        self.training_set = self.files[:threshold]
        self.testing_set = self.files[threshold:]

    def __len__(self): 
        if self.training:
            return len(self.training_set)
        else:
            return len(self.testing_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.training:
            item = self.training_set[idx]
            class_id = int(item.split('/')[-2])
            
            img = Image.open(item)
            img = img.resize((32,32))
            # img = img.resize((64,64))
            
            img = self.transform(img)

            return img, class_id
        else:
            item = self.testing_set[idx]
            class_id = int(item.split('/')[-2])
            
            img = Image.open(item)
            img = img.resize((32,32))
            # img = img.resize((64,64))
            
            img = self.transform(img)

            return img, class_id

# class GTSRBImbalance(Dataset):
#     """GTSRB Image Dataset"""

#     def __init__(self, root_dir, minority=14,training=True, transform=None):
#         self.root_dir = root_dir
#         self.files = glob.glob(self.root_dir + '/*/*.ppm')
#         if transform == None:
#             self.transform = transforms.ToTensor()
#         else:
#             self.transform = transform
#         self.training = training
#         threshold = int(len(self.files)*0.80)
#         self.training_set = self.files[:threshold]
#         self.testing_set = self.files[threshold:]
#         self.minority = minority

#     def __len__(self): 
#         if self.training:
#             return len(self.training_set)
#         else:
#             return len(self.testing_set)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         if self.training:
#             item = self.training_set[idx]
#             class_id = int(item.split('/')[-2])
#             if class_id == self.minority:
#                 class_id = 0
#             if class_id != self.minority:
#                 class_id = 1
#             img = Image.open(item)
#             img = img.resize((225,225))
#             # img = img.resize((64,64))
            
#             img = self.transform(img)

#             return img, class_id
#         else:
#             item = self.testing_set[idx]
#             class_id = int(item.split('/')[-2])
#             if class_id == self.minority:
#                 class_id = 0
#             if class_id != self.minority:
#                 class_id = 1
#             img = Image.open(item)
#             img = img.resize((225,225))
#             # img = img.resize((64,64))
            
#             img = self.transform(img)

#             return img, class_id

class GTSRBImbalance(Dataset):
    """GTSRB Image Dataset"""

    def __init__(self, root_dir, minority=14,training=True, transform=None):
        self.root_dir = root_dir
        self.files = glob.glob(self.root_dir + '/*/*.ppm')
        if transform == None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.training = training
        self.minority = minority
        self.training_set, self.testing_set = self.get_data()

    def __len__(self): 
        if self.training:
            return len(self.training_set)
        else:
            return len(self.testing_set)

    def get_data(self):
        minority = []
        majority = []
        for data in self.files:
            class_id = int(data.split('/')[-2])
            if class_id == self.minority:
                class_id = 0
                minority.append((data, class_id))
            if class_id != self.minority:
                class_id = 1
                majority.append((data, class_id))
        minority_threshold = int(len(minority)*0.5)
        majority_threshold = int(len(majority)*0.8)
        training_set = minority[:minority_threshold] + majority[:majority_threshold]
        testing_set = minority[minority_threshold:] + majority[majority_threshold:]
        return training_set, testing_set

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.training:
            item = self.training_set[idx]
            class_id = item[1]
            img = Image.open(item[0])
            img = img.resize((225,225))
            # img = img.resize((64,64))
            img = self.transform(img)
            return img, class_id
        else:
            item = self.testing_set[idx]
            class_id = item[1]
            img = Image.open(item[0])
            img = img.resize((225,225))
            # img = img.resize((64,64))
            img = self.transform(img)
            return img, class_id