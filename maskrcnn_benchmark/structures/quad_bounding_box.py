# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import time


class QuadBoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, quad_bbox, image_size, mode="xyxy", source="None"):
        device = quad_bbox.device if isinstance(quad_bbox, torch.Tensor) else torch.device("cpu")
        quad_bbox = torch.as_tensor(quad_bbox, dtype=torch.float32, device=device)
        #print("Device of quad box ",device)
        if quad_bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(quad_bbox.ndimension())
            )
        if quad_bbox.size(-1) != 8:
            raise ValueError(
                "last dimenion of bbox should have a "
                "size of 8, got {}".format(quad_bbox.size(-1))
            )
        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        self.device = device
        self.quad_bbox = quad_bbox  # 8 values
        # print("Source:", quad_bbox.shape)
        if not source=="box_head/inference":
            self.bbox = self.quad_bbox_to_bbox()
        #tic_qud = time.time()
        #self.bbox = self.quad_bbox_to_bbox()  # 4 values
        #print("time taken to convert quad box to bbox",time.time()-tic_qud)
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def quad_bbox_to_bbox(self):
        bbox = torch.zeros((self.quad_bbox.shape[0], 4))
        if self.quad_bbox.shape[0] == 0:
            return bbox.to(self.device)
        bbox[:, 0], _ = torch.min(self.quad_bbox[:, 0::2], 1)
        bbox[:, 1], _ = torch.min(self.quad_bbox[:, 1::2], 1)
        bbox[:, 2], _ = torch.max(self.quad_bbox[:, 0::2], 1)
        bbox[:, 3], _ = torch.max(self.quad_bbox[:, 1::2], 1)
        return bbox.to(self.device)

    def add_field(self, field, field_data):   #del test_dict['Mani'] 
        self.extra_fields[field] = field_data

    def remove_field(self, field):   
        del self.extra_fields[field]

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy"):
            raise ValueError("mode should be 'xyxy'")
        if mode == self.mode:
            return self


    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.quad_bbox * ratio
        else:
            ratio_width, ratio_height = ratios
            scaled_box = self.quad_bbox
            scaled_box[:, 0::2] *= ratio_width
            scaled_box[:, 1::2] *= ratio_height
        bbox = QuadBoxList(scaled_box, size, mode=self.mode,source="same file for resize functions")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox

    def to(self, device):
        bbox = QuadBoxList(self.quad_bbox.to(device), self.size, self.mode, source="same file for to functions")
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = QuadBoxList(self.quad_bbox[item], self.size, self.mode, source="same file for __getitem__ functions")
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        if isinstance(self.quad_bbox, torch.Tensor):
            return self.quad_bbox.shape[0]
        else:
            return len(self.quad_bbox)

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.quad_bbox[:, 0::2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.quad_bbox[:, 1::2].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = QuadBoxList(self.quad_bbox, self.size, self.mode, source="same file for copy_with_fields functions")
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    # a=([[0, 0, 10, 10, 0, 0, 5, 5]], (10, 10))
    # print(a[0])
    bbox = QuadBoxList([[0, 0, 10, 10, 0, 0, 5, 5]], (10, 10))
    # print(bbox.bbox)
    s_bbox = bbox.resize((5, 5))
    # print(s_bbox.shape)
    print(s_bbox.bbox)

    # t_bbox = bbox.transpose(0)
    # print(t_bbox)
    # print(t_bbox.bbox)
