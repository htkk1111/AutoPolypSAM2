import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import cfg
from conf import settings
from func.helper import *
import pandas as pd
from sam2_train.loss_fns import MultiStepMultiMasksAndIous

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
loss_params = {
    'weight_dict': {
        'loss_mask': 20,
        'loss_dice': 1,
        'loss_iou': 1,
        'loss_class': 0
    },
    'supervise_all_iou': True,
    'iou_use_l1_loss': True,
    'pred_obj_scores': True,
    'focal_gamma_obj_score': 0.0,
    'focal_alpha_obj_score': -1.0
}


criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32
threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

#torch.backends.cudnn.benchmark = True

from models.model_single import ModelEmb


def structure_loss(pred, mask):  
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train_sam(args, model_emb, net: nn.Module, optimizer, train_loader, epoch, writer):

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    
    # train mode
    net.train()
    optimizer.zero_grad()

    # init
    epoch_loss = 0
    lossfunc = criterion_G
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    memory_bank_list = []

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for ind, pack in enumerate(train_loader):

            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            # input image and gt masks
            batch = pack['image'].to(dtype = mask_type, device = GPUdevice)
            mask = pack['mask'].to(dtype = mask_type, device = GPUdevice)
                

            '''Train image encoder'''
            backbone_out = net.forward_image(batch)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)

            '''Train memory attention to condition on meomory bank'''
            B = vision_feats[-1].size(1)  # batch size

            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(
                    device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                    torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

            else:
                for element in memory_bank_list:
                    to_cat_memory.append(
                        (element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1))  # maskmem_features
                    to_cat_memory_pos.append(
                        (element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1))  # maskmem_pos_enc
                    to_cat_image_embed.append((element[3]).cuda(non_blocking=True))  # image_embed

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                similarity_scores = F.softmax(similarity_scores, dim=1)
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(
                    1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2),
                                                          memory_stack_ori_new.size(3))
                

                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                )

            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

            image_embed = feats[-1]
            high_res_feats = feats[:-1]

            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed

            '''prompt encoder'''
            # No `torch.no_grad()` block, allow ModelEmb to be trained
            dense_embeddings = None
            if (ind%5)==0:
                input = batch
                dense_embeddings = model_emb(input)
                flag = True
            else:
                flag = False    
              
            sparse_embeddings_none, dense_embeddings_none = net.sam_prompt_encoder(points=None, boxes=None, masks=None, batch_size=B) 

            if dense_embeddings is not None:
                de = dense_embeddings
            else:
                de = dense_embeddings_none
            se = sparse_embeddings_none

            # Now `de` is the dense embeddings used in the prompt encoder.
            # Continue with the remaining forward pass, using `se` and `de` for decoding

            # mask decoder
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,  # Use the dense embeddings `de` here
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats
            )

            # Resize prediction
            pred = F.interpolate(low_res_multimasks, size=(mask.size()[2], mask.size()[2]))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)

            '''memory encoder'''
            # new calculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

            # Add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             image_embed[batch].reshape(-1).detach()])
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     image_embed[batch].reshape(-1).detach()])

    
            # backpropagation
            temp = eval_seg(pred, mask, threshold)
            print(temp[0], temp[1])
            loss = structure_loss(pred, mask)
            pbar.set_postfix(**{'loss (batch)': loss.item()})
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

            pbar.update()

    return epoch_loss/len(train_loader)





def validation_sam(args, val_loader, epoch, model_emb, net: nn.Module, CreatMask=False, save_path=None, clean_dir=True):

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    # eval mode
    net.eval()
    model_emb.eval()
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0


    with tqdm(total=len(val_loader), desc='Validation round', unit='batch') as pbar:
        for ind, pack in enumerate(val_loader):


            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            batch = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            mask = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
            if CreatMask==True:
                [name] = pack['name']
                [s] = pack['shape']
            

            '''test'''
            with torch.no_grad():

                """ image encoder """
                backbone_out = net.forward_image(batch)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1)
                
                """ memory condition """
                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(
                        device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(
                        torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True))  # image_embed

                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                    similarity_scores = F.softmax(similarity_scores, dim=1)
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(
                        1)  # Shape [batch_size, 16]

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2),
                                                          memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2),
                                                              memory_stack_ori_new.size(3))

                

                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                    )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size)
                         for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                '''prompt encoder'''
                # No `torch.no_grad()` block, allow ModelEmb to be trained
                dense_embeddings = None
                if (ind%5)==0:
                    input = batch
                    dense_embeddings = model_emb(input)
                    flag = True
                else:
                    flag = False    
                  
                sparse_embeddings_none, dense_embeddings_none = net.sam_prompt_encoder(points=None, boxes=None, masks=None, batch_size=B) 
    
                if dense_embeddings is not None:
                    de = dense_embeddings
                else:
                    de = dense_embeddings_none
                se = sparse_embeddings_none
    
                # Now `de` is the dense embeddings used in the prompt encoder.
                # Continue with the remaining forward pass, using `se` and `de` for decoding

                # mask decoder
                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                """ memory encoder """
                maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag)

                maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)

                """ memory bank """
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0],
                                                 image_embed[batch].reshape(-1).detach()])

                else:
                    for batch in range(maskmem_features.size(0)):

                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2,
                                                                        dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores)
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < \
                                current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index)
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0],
                                                         image_embed[batch].reshape(-1).detach()])

                  
                # binary mask and calculate loss, iou, dice
                total_loss += structure_loss(pred, mask)
                pred = (pred> 0.5).float()
                if CreatMask==True:
                    s = (s[1], s[0])
                    res = F.upsample(pred, size=s, mode='bilinear', align_corners=False)  

                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)  

                    cv2.imwrite(save_path+name, res*255)  
                    
                temp = eval_seg(pred, mask, threshold)
                #print(temp[0], temp[1])
                total_eiou += temp[0]
                total_dice += temp[1]


                            
            pbar.update()

    return total_loss/len(val_loader) , tuple([total_eiou/len(val_loader), total_dice/len(val_loader)])


