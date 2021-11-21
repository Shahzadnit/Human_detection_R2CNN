# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

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
        # print(images.shape)
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # print(images.tensors.shape)
        features = self.backbone(images.tensors)
        # features11=features
        # m = nn.InstanceNorm2d(768)
        
        # features=m(features)
        features1=self.chan_attention(features)
        # feat=features

        train_mode=self.training
        sal_map,atten_losses=self.attention(train_mode,features,mask)
        features1=torch.mul(features1,(F.softmax(sal_map)[:,1,:,:]+1))
        # print(sal_map.shape)
        # print(features1.shape)
        proposals, proposal_losses = self.rpn(images, features1, targets)
        # rpn_result=proposals
        # print(proposals[0].bbox)
        # print((proposals[0].bbox).size())
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features1, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features1
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update({"Atten_loss": atten_losses})
            return losses
        # print(result)
        # rpn_result=proposals
        # return result, sal_map, feat
        return result ,None, None
