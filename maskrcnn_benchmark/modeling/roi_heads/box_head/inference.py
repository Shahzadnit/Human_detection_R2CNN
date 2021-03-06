# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.quad_bounding_box import QuadBoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.quad_box_coder import QuadBoxCoder
import time
class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        quad_box_coder=None,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.quad_box_coder = quad_box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        #tic = time.time()
        # print("reach at inference##############################")
        class_logits, box_regression, quad_box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        quad_proposals = self.quad_box_coder.decode(
            quad_box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        quad_proposals = quad_proposals.split(boxes_per_image, dim=0)
        #print(quad_proposals)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        #print("In postprocc clas-prob shape",class_prob.shape,quad_proposals.shape,proposals.shape,len(image_shapes))
        #print("time in post processor 1st",time.time()-tic)

        results = []
        #tic =time.time()
        for prob, boxes_per_img, quad_boxes_per_img, image_shape in zip(
            class_prob, proposals, quad_proposals, image_shapes
        ): 
            #print("inside loop boxes per image shape",boxes_per_img.shape)
            #tic2 = time.time()
            boxlist = self.prepare_boxlist(boxes_per_img, quad_boxes_per_img, prob, image_shape)
            #print(" time inside for loop for prepare boxlist",time.time()-tic2)
            #tic3 = time.time()
            boxlist = boxlist.clip_to_image(remove_empty=False)
            #print(" time inside for loop for clip to image",time.time()-tic3)
            #print("boxlist inside loop in inference",boxlist)
            #tic1 = time.time()
            boxlist = self.filter_results(boxlist, num_classes)
            #print(" time inside for loop for filter result",time.time()-tic1)
            # print(boxlist)
            results.append(boxlist)
        #print("len of result in infrence",(results))
        #print("time in postprocessor 2nd",time.time()-tic)
        return results

    def prepare_boxlist(self, boxes, quad_boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        quad_boxes = quad_boxes.reshape(-1, 8)
        scores = scores.reshape(-1)
        boxlist = QuadBoxList(quad_boxes, image_shape, mode="xyxy", source="box_head/inference")
        boxlist.bbox = boxes
        boxlist.add_field("scores", scores)
        # print(boxlist.quad_bbox)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        quad_boxes = boxlist.quad_bbox.reshape(-1, num_classes * 8)
        # print(quad_boxes.quad_bbox)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            quad_boxes_j = quad_boxes[inds, j * 8 : (j + 1) * 8]
            boxlist_for_class = QuadBoxList(quad_boxes_j, boxlist.size, mode="xyxy",source="box_head/inference")
            boxlist_for_class.bbox = boxes_j
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    quad_bbox_reg_weights = cfg.MODEL.ROI_HEADS.QUAD_BBOX_REG_WEIGHTS
    quad_box_coder = QuadBoxCoder(weights=quad_bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        quad_box_coder,
        cls_agnostic_bbox_reg
    )
    return postprocessor
