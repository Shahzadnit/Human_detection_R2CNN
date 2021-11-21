# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
import torchvision.transforms as torch_transform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class inference_engine(object):
    # COCO categories for pretty print
    CATEGORIES =["__background__","Human",]
    # CATEGORIES =["__background__","fixture_1","fixture_1_slot", "fixture_2","fixture_2_slot", "fixture_3","fixture_3_slot", "fixture_4","fixture_4_slot", "fixture_6","fixture_6_slot",]
    #CATEGORIES = [
    #    "__background__",
    #    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane",]

    def __init__(
        self,
        cfg,
        weights,
        confidence_threshold=0.5,
        min_image_size=864,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(weights)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold


    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image,i,img_name):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions,cam_img, sal_map,feature= self.compute_prediction(image,i,img_name)
        # ind1=torch.where(sal_map[:,1,:,:] <0)
        # ind2=torch.where(sal_map[:,1,:,:] >=0)
        # print(ind2)
        # print(ind2[0][0].item(),ind2[1][0].item(),ind2[2][0].item())
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        return result,cam_img

    def compute_prediction(self, original_image,i,img_name):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        image = image.to(self.device)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # writer.add_graph(self.model, image_list.tensors)
        # compute predictions
        with torch.no_grad():
            predictions,cam_img = self.model(image_list.tensors, writer)
            # print(sal_map.shape)
            # predictions= self.model(image_list.tensors, writer)
        
        # smap=torch.argmax(sal_map,dim=1).cpu()
        # smap=smap.detach().numpy()
        # smap = smap.transpose(1,2,0)
        # smap=((smap*200)+55)
        # smap = smap.astype(np.uint8)
        

        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        # print(prediction.quad_bbox)

        # return prediction,smap,sal_map, feature
        return prediction,cam_img,None, None

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        # print(scores)
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        #print(boxes)
        quad_boxes = predictions.quad_bbox
        #print(quad_boxes.shape)
        # print(image.shape)

        colors = self.compute_colors_for_labels(labels).tolist()
        # colors=(128,128,255)

        for quad_box, box, color in zip(quad_boxes, boxes, colors):
            box = box.to(torch.int64)
            quad_box = quad_box.to(torch.int64)
            # print(quad_box)
            # print(box)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            #image = cv2.rectangle(
            #     image, tuple(top_left), tuple(bottom_right), tuple((0,0,255)), 3
            #)
            cv2.line(image, (quad_box[0], quad_box[1]), (quad_box[2], quad_box[3]), color, 3)
            cv2.line(image, (quad_box[2], quad_box[3]), (quad_box[4], quad_box[5]), color, 3)
            cv2.line(image, (quad_box[4], quad_box[5]), (quad_box[6], quad_box[7]), color, 3)
            cv2.line(image, (quad_box[6], quad_box[7]), (quad_box[0], quad_box[1]), color, 3)


        return image
    def overlay_boxes_rpn(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        # labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist()

        for box in zip(boxes):
            # print(box[0])
            box = box[0].to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple((0,255,0)), 2
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        # print(scores)
        labels = predictions.get_field("labels").tolist()
        # print(labels)
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        q_boxes = predictions.quad_bbox
        #print(predictions.quad_bbox)

        template = "{}: {:.2f}"
        for box, qbox, score, label in zip(boxes, q_boxes, scores, labels):
            #print(qbox[0])
            x0,y0,x1,y1,x2,y2,x3,y3 = qbox[0], qbox[1], qbox[2], qbox[3], qbox[4], qbox[5], qbox[6], qbox[7]
            x, y = box[:2]
            #print(label)
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2
            )
            #cv2.putText(image, '1st', (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 2)
            #cv2.putText(image, '2nd', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 2)
            #cv2.putText(image, '3rd', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 2)
            #cv2.putText(image, '4th', (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 2)


        return image

import os
import time
from maskrcnn_benchmark.config import cfg
if __name__ == '__main__':
    # test_img_dir = '../datasets/Machine_tending/test_final'
    test_img_dir = '../datasets/human/img'
    out_dir = '../datasets/human/result'
    # out_dir = '/home/cvlab/Desktop/test_img/take'
    config_file = '../configs/Human_det_test.yaml'
    # weights = '../output/human_2/model_0007000.pth'
    weights = '../output/human/model_final.pth'
    # weights = '/media/cvlab/90855c73-58e1-44e8-9f80-4b3e5100021e/MT_weights/MT_weights_new/MT_weights_new/model_0220000.pth'
    img_lists = os.listdir(test_img_dir)
    cfg.merge_from_file(config_file)
    detector = inference_engine(cfg, weights)

    for i, img_name in enumerate(img_lists):
        print(img_name)
        strt = time.time()
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        # print(img.shape)
        #writer.add_image('icdar_input_image', img)
        canvas,cam_img = detector.run_on_opencv_image(img,i,img_name)
        # print(canvas.shape[0],canvas.shape[1])
        smap = cv2.resize(cam_img, (canvas.shape[1],canvas.shape[0]))
        # print(smap.shape)
        # print(canvas.shape)
        # cv2.imwrite('temp/'+img_name, smap)
        end = time.time()
        print("time taken for detection:", end-strt)
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, np.hstack([canvas,smap]))
        writer.close()
        # break
