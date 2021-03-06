# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn
import time


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels
        #num_inputs = 256*7*7

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)
        self.quad_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 8)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        nn.init.normal_(self.quad_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.quad_pred.bias, 0)

    def forward(self, x):
        #print("x1 & x2 shapes in predictor",x1.shape,x2.shape)
        #print(self.in_channels)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x1 = x1.view(x1.size(0), -1)
        #x2 = x2.view(x2.size(0), -1)
        #print("in predictor befor fc",x.shape)
        #tic = time.time()
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        quad_deltas = self.quad_pred(x)
        #cls_logit = self.cls_score(x1)
        #bbox_pred = self.bbox_pred(x2)
        #quad_deltas = self.quad_pred(x2)
        #print("class logits score",cls_logit.shape)
        #print("time taken in predictor",time.time()-tic)
        return cls_logit, bbox_pred, quad_deltas


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.quad_pred = nn.Linear(representation_size, num_bbox_reg_classes * 8)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.normal_(self.quad_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred, self.quad_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        quad_deltas = self.quad_pred(x)

        return scores, bbox_deltas, quad_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
