#!/usr/bin/python
# author htkk1111
# 2024年12月05日


import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import cfg
from conf import settings
import func.function as function
from func.vpsdatasets import *
from func.helper import *

import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.model_single import ModelEmb

import torch
from tqdm import tqdm
from func.function import validation_sam

import os
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from natsort import natsorted 

def collate_fn(batch):
    images = []
    masks = []
    names = []
    shapes = []
    
    for item in batch:
        images.append(item['image'])
        masks.append(item['mask'])
        names.append(item['name'])
        shapes.append(item['shape'])
    
    return {
        'image': torch.stack(images),
        'mask': torch.stack(masks),
        'name': names,  
        'shape': shapes  
    }


class V(Dataset):
    def __init__(self, frame_path, gt_path, img_size, augmentations=False):
        self.frame_path = frame_path
        self.gt_path = gt_path
        self.img_size = img_size
        self.augmentations = augmentations

        self.frame_files, self.gt_files = self._load_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4732661 , 0.44874457, 0.3948762 ],
                                 [0.22674961, 0.22012031, 0.2238305 ])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def _load_files(self):
        frame_files = natsorted(os.listdir(self.frame_path)) 
        gt_files = natsorted(os.listdir(self.gt_path)) 

        frame_files = [os.path.join(self.frame_path, f) for f in frame_files]
        gt_files = [os.path.join(self.gt_path, f) for f in gt_files]

        assert len(frame_files) == len(gt_files), "Mismatch between frames and masks!"

        return frame_files, gt_files

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, index):
        img_path = self.frame_files[index]
        mask_path = self.gt_files[index]
        
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask_size = mask.size

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        
        img = self.img_transform(img)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        
        mask = self.gt_transform(mask)
        mask = (mask >= 0.5).float()

        return {'image': img, 'mask': mask, 'name': img_name, 'shape': mask_size}







# args = cfg.parse_args()
GPUdevice = torch.device('cuda', args.gpu_device)


net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)


model_emb = ModelEmb(args)  
model_emb = model_emb.to(GPUdevice)

checkpoint = torch.load('')
model_emb.load_state_dict(checkpoint['model_state_dict'])
net.load_state_dict(checkpoint['net_state_dict'])

mode = [ ]  


import time



for md in mode:

    total_inference_time = 0
    total_frames_processed = 0

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    total_iou = 0
    total_dice = 0
    total_num = 0

    frame_path = os.path.join(args.data_path, md, 'Frame')
    gt_path = os.path.join(args.data_path, md, 'GT')
    save_path = './result_map/myself/{}/'.format(md)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    subfolders = sorted(os.listdir(frame_path))

    
    start_time = time.perf_counter()
    
    for subfolder in subfolders:
        subfolder_frame_path = os.path.join(frame_path, subfolder)
        subfolder_gt_path = os.path.join(gt_path, subfolder)

        if not (os.path.isdir(subfolder_frame_path) and os.path.isdir(subfolder_gt_path)):
            continue

        print(f"Processing subfolder: {subfolder}")

        dataset = V(subfolder_frame_path, subfolder_gt_path, args.image_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        total_frames_processed += len(dataloader)


        _, (iou, dice) = validation_sam(
            args, dataloader, 0, model_emb, net, CreatMask=True, save_path=save_path
        )

        total_iou += iou * len(dataloader)
        total_dice += dice * len(dataloader)
        total_num += len(dataloader)

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    total_inference_time += inference_time


       
    if md == '':
        name = ''
    elif md == '':
        name = ''
    else:
        name = ''

    print(f'{name}  dice:{total_dice/total_num}, iou:{total_iou/total_num}')

    fps = total_frames_processed / total_inference_time
    ms_per_frame = (total_inference_time / total_frames_processed) * 1000  
    
    print(f"Total inference time: {total_inference_time:.2f} seconds")
    print(f"Frames processed: {total_frames_processed}")
    print(f"Inference speed: {fps:.2f} FPS")
    print(f"Time per frame: {ms_per_frame:.2f} ms") 













