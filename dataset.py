import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class GTSRB(Dataset):
    """GTSRB Image Dataset"""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = glob.glob(self.root_dir + '/*/*.ppm')
        self.transform = transforms.ToTensor()

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.files[idx]
        class_id = int(item.split('/')[-2])
        
        img = Image.open(item)
        img = img.resize((225,225))
        
        img = self.transform(img)

        return img, class_id