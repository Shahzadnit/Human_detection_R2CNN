# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from .generalized_rcnn import GeneralizedRCNN
# from .generalized_rcnn_exp1 import GeneralizedRCNN
from .generalized_rcnn_exp2 import GeneralizedRCNN
# from .generalized_rcnn_exp3 import GeneralizedRCNN
# from .generalized_rcnn_exp4 import GeneralizedRCNN
# from .generalized_rcnn_exp5 import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
