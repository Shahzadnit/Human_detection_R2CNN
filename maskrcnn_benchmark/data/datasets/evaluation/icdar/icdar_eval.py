# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
import os, json
from collections import defaultdict
import torch
import numpy as np
from shapely.geometry import Polygon
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.quad_bounding_box import QuadBoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def calculate_iou(bbox,gtbox):
    poly_1 = Polygon([[bbox[0],bbox[1]],[bbox[2],bbox[3]],[bbox[4],bbox[5]],[bbox[6],bbox[7]]])
    poly_2 = Polygon([[gtbox[0],gtbox[1]],[gtbox[2],gtbox[3]],[gtbox[4],gtbox[5]],[gtbox[6],gtbox[7]]])
    iou = poly_1.intersection(poly_2).area/poly_1.union(poly_2).area
    return iou
def Quad_iou(box,gt):
    iou = torch.zeros([box.shape[0],gt.shape[0]]).float()
    for i in range(box.shape[0]):
        for j in range(gt.shape[0]):
            iou[i,j] = calculate_iou(box[i,:],gt[j,:])
    return iou

def get_groundtruth(image_id, dataset, image_width, image_height):
    path = os.path.join(dataset.ann_file)
    with open(path, 'r') as f:
        ids = json.load(f)
    anno=ids[image_id]
    # print(anno)
    boxes = [obj["bbox"] for obj in anno['objs'] if 0 or not obj['isDifficult']]
    rboxes = torch.as_tensor(boxes).reshape(-1,8)
    # print(rboxes.shape)
    target = QuadBoxList(rboxes, [anno['width'], anno['height']], mode="xyxy", source="icdar")
    # target = BoxList(target.bbox, [image_width,image_height], mode="xyxy")
    classes = [obj["category_id"] for obj in anno['objs']]
    classes = torch.tensor(classes)
    target.add_field("labels", classes)
    difficult = [obj["isDifficult"] for obj in anno['objs']]
    difficult = torch.tensor(difficult)
    target.add_field("difficult", difficult)
    return target


def do_icdar_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(prediction) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        # print(prediction.quad_bbox.shape)#torch.Size([2, 8])
        pred_boxlists.append(prediction)

        gt_boxlist = get_groundtruth(image_id, dataset, image_width, image_height)
        gt_boxlists.append(gt_boxlist)
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    # print(gt_boxlists,pred_boxlist)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        # print("GT----->",gt_boxlist)
        # print("pred ---->",pred_boxlist)
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_bbox_quad = pred_boxlist.quad_bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_bbox_quad = gt_boxlist.quad_bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        # print(gt_label)
        # print(gt_bbox.shape)
        # print(pred_label)
        # print(pred_score)
        # print("GT bbox------>",gt_bbox.shape)
        # print("Pred bbox------>",pred_bbox.shape)
        gt_difficult = gt_boxlist.get_field("difficult").numpy()
        # print('###############################',gt_difficult)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # print(l)
            # print(gt_label)
            # print(gt_bbox.shape)
            # print(pred_label)
            pred_mask_l = pred_label == l
            # print(pred_mask_l)
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_bbox_l_quad = pred_bbox_quad[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_bbox_l_quad = pred_bbox_l_quad[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            # print(gt_mask_l)
            # print(gt_bbox)
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_bbox_l_quad = gt_bbox_quad[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l_quad = pred_bbox_l_quad.copy()
            pred_bbox_l[:, 2:] += 1
            pred_bbox_l_quad[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l_quad = gt_bbox_l_quad.copy()
            gt_bbox_l[:, 2:] += 1
            gt_bbox_l_quad[:, 2:] += 1
            # print("GT---->",gt_bbox_l_quad)
            # print("Pred --->",pred_bbox_l_quad)
            # iou_quad = Quad_iou(pred_bbox_l_quad,gt_bbox_l_quad).numpy()
            # print(iou_quad)
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            # print(iou)
            # print("#########################################",iou)
            # print("iou Shape #####################",iou.shape)
            # print("Quad iou Shape #####################",iou_quad.shape)
            #gt_index = iou_quad.argmax(axis=1)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            #gt_index[iou_quad.max(axis=1) < iou_thresh] = -1
            del iou
            #del iou,iou_quad

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
    # print(n_pos.keys())
    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
