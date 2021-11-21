import os
import numpy as np
import cv2
import torch
import torchvision
import os, json
from PIL import Image
from shapely import geometry
import sys
import torchvision.transforms as torch_transform
from maskrcnn_benchmark.structures.qbounding_box import QBoxList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.quad_bounding_box import QuadBoxList

def get_mask2( boxes,w,h,img1 ):
    # print(img.shape)
    box_list = []
    for i in range(len(boxes)):
        b=[boxes[i][0], boxes[i][1],boxes[i][2], boxes[i][3], boxes[i][4], boxes[i][5], boxes[i][6],boxes[i][7]]
        box_list.append(b)
        # print(b)
    gtbox_label = np.array(box_list, dtype=np.int32)

    # w, h = img.size
    mask = np.zeros([h, w])
    for b in gtbox_label:
#         b = np.reshape(b[0:-1], [4, 2])
        b = np.reshape( b,[4, 2])
        # print(b)
        # pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
        pts = b.reshape((-1,1,2))
        # # print(pts)
        cv2.polylines(img1,[pts],True,(0,0,255),6)

        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    # mask = np.array(mask, np.float32)
    mask=cv2.resize(mask,(w,h))
    mask = np.uint8(mask)
    # print(mask[np.nonzero(mask)])
    # im=Image.fromarray(mask)
    return mask,img1
def poly_area(coor):
    coor=np.asarray(coor)

    poly = geometry.Polygon(((coor[0],coor[1]),(coor[2],coor[3]),(coor[4],coor[5]),(coor[6],coor[7])))
    return poly.area

class IcdarDataset(torch.utils.data.Dataset):
    # CLASSES = (
    #    "__background",
    #    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane",)
    # CLASSES = ("__background__","Text",)
    CLASSES = ("__background__","Human",)

    def __init__(self, root, ann_file, use_difficult=True, transforms=None):
        self.img_dir = root
        # self.mask_dir = root_mask
        self.ann_file = ann_file
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self.load_annotations()

    def load_annotations(self):
        path = os.path.join(self.ann_file)
        with open(path, 'r') as f:
            self.ids = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        anno = self.ids[idx]
        # print(idx)
        #print(anno['objs'])
        boxes = [obj["bbox"] for obj in anno['objs'] if self.keep_difficult or not obj['isDifficult']]
        #print(boxes)
        # rboxes = [obj["bbox"] for obj in anno['objs'] if self.keep_difficult or not obj['isDifficult']]
        # print(len(rboxes), len(boxes))
        # boxes = []
        # for rbox in rboxes:
        #     boxes.append([min(rbox[0],rbox[2], rbox[4], rbox[6]), min(rbox[1],rbox[3], rbox[5], rbox[7]),
        #             max(rbox[0],rbox[2], rbox[4], rbox[6]), max(rbox[1],rbox[3], rbox[5], rbox[7])])
        # boxes = torch.as_tensor(boxes).reshape(-1, 4)

        rboxes = torch.as_tensor(boxes).reshape(-1,8)
        # print(rboxes.dtype)
        # print(boxes.shape)#torch.Size([46, 8])
        # print([anno['width'], anno['height']])#[1280, 720]
        #print(anno['img_name'])
        img1 = Image.open(os.path.join(self.img_dir, anno['img_name'])).convert("RGB")
        # mask_gt = Image.open(os.path.join(self.mask_dir, anno['img_name'])).convert("RGB")
        # print(mask_gt.size)
        height = img1.size[1]
        width = img1.size[0]
        target = QuadBoxList(rboxes, [anno['width'], anno['height']], mode="xyxy", source="icdar")
        # target = QuadBoxList(rboxes, [width, height], mode="xyxy", source="icdar") #this chnge for WCCL data
        # target = QBoxList(rboxes, boxes, img.size, mode="xyxy")   #.convert("xyxy")
        # target = BoxList(boxes, img.size, mode="xyxy") 
        # print(target.device)
        # print((target.bbox).shape)#torch.Size([9, 4])

        classes = [obj["category_id"] for obj in anno['objs']]
        #print(classes)
        classes = torch.tensor(classes)
        # print(classes)
        target.add_field("labels", classes)
        target = target.clip_to_image(remove_empty=False)
        # print(target.device)
        # print("befor ###############",(target.quad_bbox[0]))#torch.Size([6, 4])

        # img1 = Image.open(os.path.join(self.img_dir, anno['img_name'])).convert("RGB")
        # print(img1.size[0])

        ####################################################################################################
        # ww = img1.size[0]
        # hh = img1.size[1]
        # img_c2 = np.asarray(img1)
        # img_c2=np.reshape(img_c2,(hh,ww,3))
        # bb = (target.quad_bbox).numpy()
        # mask_c, img_c2=get_mask2(bb,ww,hh,img_c2)
        
        # print(img2.shape)
        # cv2.imwrite("/home/rl/frameworks/R2CNN.pytorch/datasets/WCCL/test_anno/before/%d_img.jpg"%idx,img_c2)
        #####################################################################################################
        if self.transforms is not None:
            img, target = self.transforms(img1, target)


        
        # for index, coor in enumerate(target.quad_bbox):
        #     thres=(1)*(img.shape[2]*img.shape[1]/(img1.size[0]*img1.size[1]))
        #     # print(thres)
        #     if poly_area(coor)<thres and len(list(target.quad_bbox))>index and len(list(target.quad_bbox))>1:
        #         # print(index,len(list(predictions[item].bbox)))
        #         # print(predictions[item].bbox)
        #         target.bbox= torch.cat([target.bbox[0:index,:], target.bbox[index+1:,:]])
        #         label =target.get_field("labels")
        #         # score =predictions[item].get_field("scores")
        #         target.remove_field('labels')
        #         # predictions[item].remove_field('scores')
        #         # print(score)
        #         label =torch.cat([label[0:index], label[index+1:]])
        #         # score =torch.cat([score[0:index], score[index+1:]])
        #         target.add_field("labels",label)
        #         # predictions[item].add_field("scores",score)
        #         target.quad_bbox= torch.cat([target.quad_bbox[0:index,:], target.quad_bbox[index+1:,:]])
        

        trans = torch_transform.ToPILImage()
        img1=trans(img)          
        # img1.save("/home/dl/frameworks/R2CNN.pytorch/datasets/ICDAR2015/img_atten/%d_img.jpg"%idx,"JPEG")
        w=(img.shape)[2]
        h = (img.shape)[1]
        #print(img1.size,h,w)
        b = (target.quad_bbox).numpy()
        # # # print(img.shape)#  torch.Size([3, 750, 1333])
        img1 = np.asarray(img1)
        img1=np.reshape(img1,(h,w,3))
        mask1, img2=get_mask2(b,w,h,img1)
        # print(img2.shape)
        # cv2.imwrite("./datasets/after/%d_img.jpg"%idx,img2)
        # img1.save("/home/rl/frameworks/R2CNN.pytorch/datasets/ICDAR2015/img_atten/%d_img.jpg"%idx,"JPEG")
        # print(mask.size)
        # print(np.sum(mask1))
        # print(mask1)
        mask1=np.reshape(mask1,(1,h,w))
        mask1 = torch.from_numpy(mask1)
        # print(mask1.shape)
        img_mask=trans(mask1*255)
        mask1 = mask1.long()
        # img_mask.save("./datasets/mask/%d_mask.jpg"%idx,"JPEG")


        ################################ For mask ##############################
        # mask_gt = np.array(mask_gt)
        # mask_gt = ((mask_gt[:,:,0]*255)/255).astype(np.uint8)
        # # print(mask_gt.shape)
        # mask_gt = cv2.resize(mask_gt,(w,h))
        # mask_gt=np.reshape(mask_gt,(1,h,w))
        # # cv2.imwrite("./datasets/mask/%d_img.jpg"%idx,mask_gt*255)
        # mask_gt = torch.from_numpy(mask_gt)
        # # img_mask=trans(mask1*255)
        # mask_gt = mask_gt.long()
        ##################################################################

        return img, target,mask1, idx

        # return img, target, idx

    def get_img_info(self, idx):
        anno = self.ids[idx]
        # print(anno)
        return {"height": anno['height'], "width": anno['width']}
    def map_class_id_to_class_name(self, class_id):
        return IcdarDataset.CLASSES[class_id]
