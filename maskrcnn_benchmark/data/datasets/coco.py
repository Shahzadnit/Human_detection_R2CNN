# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib
import cv2
import numpy as np
import torchvision.transforms as transforms


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.qbounding_box import QBoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False
# def get_mask(img, boxes ):
#     box_list = []
#     for i in range(len(boxes)):
#         b=[boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][0],boxes[i][3]]
#         box_list.append(b)
#             # print(b)
#     gtbox_label = np.array(box_list, dtype=np.int32)

#     w, h = img.size
#     mask = np.zeros([h, w])
#     for b in gtbox_label:
# #         b = np.reshape(b[0:-1], [4, 2])
#         b = np.reshape( b,[4, 2])
#         # print(b)
#         rect = np.array(b, np.int32)
#         cv2.fillConvexPoly(mask, rect, 1)
#     # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
#     mask = np.expand_dims(mask, axis=-1)
#     mask = np.array(mask, np.float32)
#     mask=cv2.resize(mask,(w,h))
#     im=Image.fromarray(np.uint8(mask))
#     return im


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

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        # print(idx)
        img, anno = super(COCODataset, self).__getitem__(idx)
        
        # print(img.size)
        # print(anno[0])

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        #print(anno)

        boxes = [obj["bbox"] for obj in anno]
        # print(len(boxes))
        # box_list = []
        # for i in range(len(boxes)):
        #     b=[boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][0],boxes[i][3]]
        #     box_list.append(b)
        #     # print(b)
        # gtbox_label = np.array(box_list, dtype=np.int32)
        # print(gtbox_label)
        # im=get_mask(img, boxes)
        # print(im.size)
        
        # imgplot=plt.imshow(im)
        # plt.show()
        # print(img.size)

        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # print(boxes.shape)# coordinate
        # target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")
        target = QBoxList(boxes, img.size, mode="xywh").convert("xyxy")
        # print(target)#boxlist type object

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        
        trans = transforms.ToPILImage()
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
        pil2tensor = transforms.ToTensor()
        mask1 = pil2tensor(mask1)
        mask = mask1.long()
        # print(mask.shape) # torch.Size([1, 750, 1333])
        return img, target,mask, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
