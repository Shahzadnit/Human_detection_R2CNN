# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
import time
from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from maskrcnn_benchmark.modeling.backbone import attention
from maskrcnn_benchmark.modeling.backbone import chan_attention
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import torchvision.transforms as torch_transform
from PIL import Image
import cv2
import numpy as np
import random



def returnCAM(feat):
    nc, h, w = feat.shape
    feat = torch.sum(feat,dim=0)
    cam = feat.reshape(h, w)
    cam = cam - torch.min(cam)
    cam_img = np.uint8((cam / torch.max(cam)).cpu().detach().numpy()*255)
    # print(cam_img)
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        # print(self.backbone)
        self.attention = attention.Pixel_atten(self.backbone.out_channels)
        self.chan_attention = chan_attention.chan_attention(self.backbone.out_channels)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, mask=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        features1=self.chan_attention(features)
        train_mode=self.training
        if self.training:
            sal_map,atten_losses=self.attention(train_mode,features,mask)
        else:
            sal_map,mapp =self.attention(train_mode,features,mask)
            # print(mapp_1.shape)
            cam_img = returnCAM(mapp[:,1,:,:])
            # cv2.imwrite("/home/saima/Downloads/MT_r2cnn_2/datasets/human/feat_result/result_%d.jpg"%random.randint(1,1000),cam_img)

        
        features1=torch.mul(features1,sal_map[:,1,:,:])
        ###############################################################
        # if not self.training:
        #     # print(features1.shape)
        #     upsample=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        #     feat_upsample = upsample(features.cpu())
        #     cam_img = returnCAM(feat_upsample)
        #     cv2.imwrite("/home/saima/Downloads/MT_r2cnn_2/datasets/human/feat_result/result_%d.jpg"%random.randint(1,1000),cam_img)
        #####################################################################
        
        
        proposals, proposal_losses = self.rpn(images, features1, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features1, proposals, targets)
        else:
            x = features1
            result = proposals
            detector_losses = {}
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update({"Atten_loss": atten_losses})
            return losses
        return result,cam_img
