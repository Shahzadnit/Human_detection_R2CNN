import os
import numpy as np
import cv2
import torch
import torchvision
import os, json
from PIL import Image
import sys
import torchvision.transforms as torch_transform
from maskrcnn_benchmark.structures.quad_bounding_box import QuadBoxList
from maskrcnn_benchmark.structures.qbounding_box import QBoxList
from maskrcnn_benchmark.structures.bounding_box import BoxList

class DotaDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
        #'alpen_cereal','barley_muesli','cha4tea','clif_crunch','keto_coffee','guava_leaftea','joe_coffee','kashi_cereal','kind_bars','lucky_charms','maxwell_coffee','naturevalley_bars','nature_cereal','nescafe_coffee','pure_bars','rooibos_tea','specialk_fruit','specialk_berries','specialk_bars','thatsit','organic_coffee','twinings_tea','verena_coffee'
    )

    def __init__(self, data_dir, split, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "labelTxt", "%s.txt")
        self._imgpath = os.path.join(self.root, "images", "%s.png")
        self._imgsetpath = os.path.join(self.root, "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = DotaDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        # img = Image.open(self._imgpath % img_id).convert("RGB")
        img = cv2.imread(self._imgpath % img_id)
        h, w, _ = img.shape
        target = self.get_groundtruth(index, w,h)
        #print(target)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(Image.fromarray(img), target)
        return img, target, None, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index, width,height):
        img_id = self.ids[index]
        anno = open(self._annopath % img_id)
        # anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)
        # target = BoxList(anno["hboxes"], (width, height), mode="xyxy")
        target = QBoxList(anno["boxes"],anno["hboxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        hboxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        lines = target.readlines()
        for line in lines:
        	line = line.strip('\ufeff')
        	line = line.strip('\n')
        	box, name = line.split(' ')[:8], line.split(' ')[-2]
        	bndbox = tuple(map(lambda x: int(x - TO_REMOVE), list(map(float, box))))
        	# hbox = (min(bndbox[0],bndbox[2], bndbox[4], bndbox[6]), min(bndbox[1],bndbox[3], bndbox[5], bndbox[7]),
        	# 		max(bndbox[0],bndbox[2], bndbox[4], bndbox[6]), max(bndbox[1],bndbox[3], bndbox[5], bndbox[7]))
        	# print(bndbox, hbox)
        	# hboxes.append(hbox)
        	boxes.append(bndbox)
        	gt_classes.append(self.class_to_ind[name])
        # size = target.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "hboxes": [(0,0,0,0)],       #torch.tensor(hboxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id)
        h, w, _ = img.shape
        im_info = tuple(map(int, (h, w)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return DotaDataset.CLASSES[class_id]


