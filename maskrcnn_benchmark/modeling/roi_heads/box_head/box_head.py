# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import time

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator


class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        #tic=time.time()
        #print(targets)
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                # print((proposals[0].bbox).shape)#torch.Size([2027, 4])
                proposals = self.loss_evaluator.subsample(proposals, targets)
                #print(proposals)
                # print((proposals[0].bbox).shape)#torch.Size([512, 4])


        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        #tic=time.time()
        x = self.feature_extractor(features, proposals)
        #x1, x2 = self.feature_extractor(features, proposals)
        #print(x1.shape,x2.shape)
        #print("In ROI head",time.time()-tic)
        # print(x.shape)#torch.Size([512, 1024])
        # final classifier that converts the features into predictions
        class_logits, box_regression, quad_box_regression = self.predictor(x)
        #class_logits, box_regression, quad_box_regression = self.predictor(x1,x2)
        #print("Quad box regression---->",quad_box_regression.shape)#(512,16)
        #print("rect box regression---->",box_regression.shape)#torch.Size([512, 8])
        #print("Class logit------------>",class_logits.shape)#torch.Size([512, 2])
        #print("No. of proposals------->",proposals[0].bbox.shape)

        if not self.training:
            #print(proposals[0].bbox.shape)
            #tic=time.time()
            #print("In ROI Box head",time.time()-tic)
            result = self.post_processor((class_logits, box_regression, quad_box_regression), proposals)
            # print(result)
            #print("Time taken by post_processor  ROI Box head",time.time()-tic)
            return x,result, {}
        #print(quad_box_regression)

        loss_classifier, loss_box_reg, loss_quad_box_reg = self.loss_evaluator(
            [class_logits], [box_regression], [quad_box_regression]
        )
        #print(x.shape)#torch.Size([512, 1024])
        #print((proposals[0].bbox).shape)#torch.Size([512, 4])
        #print(proposals[0].bbox)
        #print("In ROI Box head",time.time()-tic)
        return (
            x,    
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg,
                 loss_quad_box_reg= loss_quad_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
