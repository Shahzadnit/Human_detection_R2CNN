# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList
from .qbounding_box import QBoxList
from .quad_bounding_box import QuadBoxList
from maskrcnn_benchmark.layers import nms as _box_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    # xywh_boxes = boxlist.convert("xywh").bbox
    # _, _, ws, hs = xywh_boxes.unbind(dim=1)
    # keep = (
    #     (ws >= min_size) & (hs >= min_size)
    # ).nonzero().squeeze(1)
    # return boxlist[keep]

    bbox = boxlist.bbox
    ws = bbox[:, 2] - bbox[:, 0]
    hs = bbox[:, 3] - bbox[:, 1]
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)
    # print(N,M)

    area1 = boxlist1.area()
    area2 = boxlist2.area()
    # print(area1.shape,area2.shape)

    box1, box2 = boxlist1.bbox, boxlist2.bbox
    device = box1.device
    # cpu=torch.device("cpu")
    # box1=box1.to(cpu)
    # box2=box2.to(cpu)
    TO_REMOVE = 1
    inter=torch.zeros([box1.shape[0],box2.shape[0]]).float().to(device)
    for i in range(box1.shape[0]):
        lt = torch.max(box1[i, None, :2], box2[:, :2])
        rb = torch.min(box1[i, None, 2:], box2[:, 2:])
        wh = (rb - lt + TO_REMOVE).clamp(min=0)
        inter[i,:]= wh[ :, 0] * wh[ :, 1]
    

    # lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    # rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    # # print(lt.device, rb.device)

    # TO_REMOVE = 1

    # wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # inter=inter.to(device)
    # print(inter.shape)







    iou = inter / (area1[:, None] + area2 - inter)
    # print(iou.shape)# torch.Size([17, 816480])
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) or isinstance(bbox, QuadBoxList) or isinstance(bbox, QBoxList)  for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    if isinstance(bboxes[0], BoxList):
        # print("reached here1")
        cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    elif isinstance(bboxes[0], QuadBoxList):
        # print("reached here2")
        cat_boxes = QuadBoxList(_cat([bbox.quad_bbox for bbox in bboxes], dim=0), size, mode, source="boxlist_ops")

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes
