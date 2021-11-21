import os
import numpy as np
import cv2
import torch
import torch.utils.data
from PIL import Image
import sys
import torchvision.transforms as torch_transform

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList


def get_mask2( boxes,w,h ):
    box_list = []
    for i in range(len(boxes)):
        b=[boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][0],boxes[i][3]]
        box_list.append(b)
            # print(b)
    gtbox_label = np.array(box_list, dtype=np.int32)

    # w, h = img.size
    mask = np.zeros([h, w])
    for b in gtbox_label:
#         b = np.reshape(b[0:-1], [4, 2])
        b = np.reshape( b,[4, 2])
        # print(b)
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    mask = np.array(mask, np.float32)
    mask=cv2.resize(mask,(w,h))
    im=Image.fromarray(np.uint8(mask))
    return im


class PascalVOCDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
        #'alpen_cereal','barley_muesli','cha4tea','clif_crunch','keto_coffee','guava_leaftea','joe_coffee','kashi_cereal','kind_bars','lucky_charms','maxwell_coffee','naturevalley_bars','nature_cereal','nescafe_coffee','pure_bars','rooibos_tea','specialk_fruit','specialk_berries','specialk_bars','thatsit','organic_coffee','twinings_tea','verena_coffee'
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PascalVOCDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        #print(target)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        trans = torch_transform.ToPILImage()
        img1=trans(img)          
        
        # img1.save("/home/alien3/frameworks/maskrcnn-benchmark/datasets/voc/img_atten/%d_img.jpg"%idx,"JPEG")
        w=(img.shape)[2]
        h = (img.shape)[1]
        # print(img.shape)
        b = (target.bbox).numpy()
        # print(img.shape)#  torch.Size([3, 750, 1333])
        mask1=get_mask2(b,w,h)
        # print(mask.size)
        # mask1.save("/home/alien3/frameworks/maskrcnn-benchmark/datasets/voc/img_atten/%d_mask.jpg"%idx,"JPEG")
        pil2tensor = torch_transform.ToTensor()
        mask1 = pil2tensor(mask1)
        mask = mask1.long()
        # print(mask.shape) # torch.Size([1, 750, 1333])
        return img, target,mask, index
        # return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text, 
                bb.find("ymin").text, 
                bb.find("xmax").text, 
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return PascalVOCDataset.CLASSES[class_id]
