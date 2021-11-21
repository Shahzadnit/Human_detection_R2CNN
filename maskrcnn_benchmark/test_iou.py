from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou


box_1 = [[511, 41], [577, 41], [577, 76], [511, 76]]
box_2 = [[544, 59], [610, 59], [610, 94], [544, 94]]

print(calculate_iou(box_1, box_2))