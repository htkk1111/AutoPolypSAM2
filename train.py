import os
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

import cfg
from conf import settings
import func.function as function
from func.ivpsdatasets import *
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
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms import ddp_comm_hooks


class Trainer:
    def __init__(self, args, autopromot, net, train_loader, test_loader, logger, model_emb):
        self.args = args
        self.net = net
        self.epoch = 100
        self.num = 0
        self.model_emb = autopromot
        # checkpoint = torch.load('./latest_epoch.pth')
        # self.model_emb.load_state_dict(checkpoint['model_state_dict'])
        # self.net.load_state_dict(checkpoint['net_state_dict'])

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger

        self.device = torch.device('cuda', args.gpu_device)


        
        self.optimizer = optim.Adam([
    {'params': self.model_emb.parameters(), 'lr': 3e-4},
    {'params': self.net.parameters(), 'lr': 1e-4}
], weight_decay=1e-5, amsgrad=False)

        self.checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.checkpoint_path = os.path.join(self.checkpoint_path, '{net}-{epoch}-{type}.pth')
        self.writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))


    def train(self):

        best_dice= 0.0
        save_path = os.path.join('./', 'latest_epoch.pth')
    
        for epoch in range(self.epoch):
            if epoch == 0:
                  tol, (eiou, edice) = function.validation_sam(args, self.test_loader, epoch, self.model_emb, self.net)
                  self.logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            self.net.train()
            self.model_emb.train()
            time_start = time.time()
            loss = function.train_sam(self.args, self.model_emb, self.net, self.optimizer, self.train_loader, epoch, self.writer)
            self.logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
            time_end = time.time()
            print('Time for training: ', time_end - time_start)
            # validation
            self.net.eval()
            self.model_emb.eval()
            if epoch % args.val_freq == 0 or epoch == self.epoch - 1:
    
                tol, (eiou, edice) = function.validation_sam(args, self.test_loader, epoch, self.model_emb, self.net, self.writer)
                self.logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    
                if edice > best_dice:
                    best_dice = edice
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model_emb.state_dict(),
                            'net_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_dice': best_dice,
                        }, save_path)


def single_proc_run(local_rank, main_port, world_size):
    """Single GPU process"""
    torch.multiprocessing.set_start_method("spawn")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:
        return None
    assert amp_type in ["bfloat16", "float16"], "Invalid Amp type."
    if amp_type == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float16

def custom_collate_fn(batch):
    from func_2d.utils.data_utils import collate_fn
    return collate_fn(batch, dict_key='all')

def custom_collate_fn2(batch):
    all_frames_data = []
    all_masks = []

    for datapoint in batch:
        for frame in datapoint.frames:
            data = frame.data if isinstance(frame.data, torch.Tensor) else torch.tensor(frame.data)
            all_frames_data.append(data)

            frame_masks = []
            for obj in frame.objects:
                frame_masks.append(obj.segment)

            all_masks.append(torch.stack(frame_masks) if frame_masks else torch.zeros_like(data[0], dtype=torch.uint8))

    all_frames_data = torch.stack(all_frames_data)
    all_masks = torch.stack(all_masks)

    return all_frames_data, all_masks




def main():
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)

    model_emb = ModelEmb(args)  
    model_emb = model_emb.to(GPUdevice)


    refuge_train_dataset = VPSDataset(args, args.data_path, mode='', augmentations=True)
    refuge_test_dataset = VPSDataset(args, args.data_path, mode='')

    nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
    nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    trainer = Trainer(args, model_emb, net, nice_train_loader, nice_test_loader, logger, model_emb)

    trainer.train()


if __name__ == '__main__':
    main()

