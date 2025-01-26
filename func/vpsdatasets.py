import func.dataset.sam2_datasets
import func.dataset.utils
import func.dataset.vos_dataset
import func.dataset.vos_raw_dataset
import func.dataset.vos_sampler
import func.utils.data_utils
import func.dataset.transforms as T
import random


phases_per_epoch = 1
train_batch_size = 1
num_frames = 10
max_num_objects = 1
multiplier = 2
num_train_workers = 10
resolution = 1024



import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VPSDataset(Dataset):
    def __init__(self, args, data_path, mode="train", augmentations=False):
        self.data_path = data_path
        self.frame_path = os.path.join(data_path, mode, 'Frame')  
        self.gt_path = os.path.join(data_path, mode, 'GT')      
        self.frame_files, self.gt_files = self._load_files()   
        self.img_size = args.image_size
        self.mask_size = args.out_size
        self.augmentations = augmentations  

        if self.augmentations:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])
        else:
            print('No augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])

    def _load_files(self):
        frame_files = []
        gt_files = []
        for root, _, files in os.walk(self.frame_path):
            for file in sorted(files): 
                frame_files.append(os.path.join(root, file))
        
        for root, _, files in os.walk(self.gt_path):
            for file in sorted(files): 
                gt_files.append(os.path.join(root, file))
        
        assert len(frame_files) == len(gt_files), "Frame and GT files count mismatch!"
        
        return frame_files, gt_files

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, index):
        img_path = self.frame_files[index]
        mask_path = self.gt_files[index]
        

        img = Image.open(img_path).convert('RGB')  
        mask = Image.open(mask_path).convert('L')  

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        
        img = self.img_transform(img) 

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        
        mask = self.gt_transform(mask)  

        mask = (mask >= 0.5).float()

        return {
            'image': img,
            'mask': mask
        }







