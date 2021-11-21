# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm
from shapely import geometry
import numpy as np

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize


def poly_area(coor):
    coor=np.asarray(coor)

    poly = geometry.Polygon(((coor[0],coor[1]),(coor[2],coor[3]),(coor[4],coor[5]),(coor[6],coor[7])))
    return poly.area


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, masks,image_ids = batch
        mask=masks
        images = images.to(device)
        mask = mask.to(device)
        # print(mask.shape)
        with torch.no_grad():
            output = model(images,mask)
            # print(output[0])
            output = [o.to(cpu_device) for o in output[0]]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda:0",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # print(predictions)
    for item in predictions:
        # print(predictions[item].quad_bbox.shape)
        # l_coor=predictions[item].quad_bbox
        # print(l_coor )
        # print(predictions[item].bbox)
        # print(predictions[item].get_field("labels"))
        # print(predictions[item].extra_fields)
        # predictions[item].remove_field('labels')
        # print(predictions[item].extra_fields)
        # print(predictions[item].get_field("difficult"))
        for index, coor in enumerate(predictions[item].quad_bbox):
            
            # print(index,"------>",poly_area(coor))
            if poly_area(coor) < 1 and len(list(predictions[item].bbox))>index:
                # print(index,len(list(predictions[item].bbox)))
                # print(predictions[item].bbox)
                predictions[item].bbox= torch.cat([predictions[item].bbox[0:index,:], predictions[item].bbox[index+1:,:]])
                label =predictions[item].get_field("labels")
                score =predictions[item].get_field("scores")
                predictions[item].remove_field('labels')
                predictions[item].remove_field('scores')
                # print(score)
                label =torch.cat([label[0:index], label[index+1:]])
                score =torch.cat([score[0:index], score[index+1:]])
                predictions[item].add_field("labels",label)
                predictions[item].add_field("scores",score)
                predictions[item].quad_bbox= torch.cat([predictions[item].quad_bbox[0:index,:], predictions[item].quad_bbox[index+1:,:]])
                # print("After---",predictions[item].quad_bbox.shape)
                # detec_quadbbox=predictions[item].quad_bbox
                # print(detec_quadbbox)
                # detec_quadbbox.pop(index)
                # predictions[item].quad_bbox=torch.cat(detec_quadbbox)

        # print(predictions[item].quad_bbox[0])
    # print(predictions)
    
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
