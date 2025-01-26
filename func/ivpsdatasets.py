import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def split_dataset(data_path, train_ratio=0.7):
    frame_path = os.path.join(data_path, 'Frame')
    gt_path = os.path.join(data_path, 'GT')

    # 获取所有文件路径
    frame_files = sorted([f.path for f in os.scandir(frame_path) if f.is_file()])
    gt_files = sorted([f.path for f in os.scandir(gt_path) if f.is_file()])

    # 确保图像与掩码数量一致
    assert len(frame_files) == len(gt_files), "图像与掩码文件数量不一致"

    # 打乱并划分数据
    combined = list(zip(frame_files, gt_files))
    random.shuffle(combined)
    frame_files, gt_files = zip(*combined)

    split_idx = int(len(frame_files) * train_ratio)
    train_data = (frame_files[:split_idx], gt_files[:split_idx])
    val_data = (frame_files[split_idx:], gt_files[split_idx:])

    return train_data, val_data



class ivps(Dataset):
    def __init__(self, data, trainsize, augmentations=True):
        """
        :param data: 数据集划分后的文件路径元组 (frame_files, gt_files)
        :param trainsize: 图像和掩码的目标大小
        :param augmentations: 是否启用数据增强
        """
        self.frame_files, self.gt_files = data
        self.trainsize = trainsize
        self.augmentations = augmentations

        # 配置变换
        if self.augmentations:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([

                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
        else:
            print('No augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, index):
        """
        获取图像和掩码并应用变换
        """
        img_path = self.frame_files[index]
        mask_path = self.gt_files[index]

        # 加载图像和掩码
        img = Image.open(img_path).convert('RGB')  # 假设图像是RGB格式
        mask = Image.open(mask_path).convert('L')  # 假设掩码是灰度图

        # 应用变换
        img = self.img_transform(img)
        mask = self.gt_transform(mask)

        # 转化为二值化的掩码
        mask = (mask >= 0.5).float()

        return {'image': img, 'mask': mask}

