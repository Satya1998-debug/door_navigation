import numpy as np
import cv2
from utils.config import CX, FX, CY, FY

def crop_to_bbox_depth(img, door_box):
    h, w = img.shape
    x_min, y_min, x_max, y_max = door_box["bbox"]

    # croping safely within image bounds
    x_min = max(0, float(x_min))
    y_min = max(0, float(y_min))
    x_max = min(w-1, float(x_max))
    y_max = min(h-1, float(y_max))

    # extract ROI from RGB and depth
    roi_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    return roi_img

def crop_to_bbox_rgb(img, door_bbox):
    h, w, _ = img.shape
    x1, y1, x2, y2 = door_bbox

    # croping safely within image bounds
    x1 = max(0, float(x1))
    y1 = max(0, float(y1))
    x2 = min(w-1, float(x2))
    y2 = min(h-1, float(y2))

    # extract ROI from RGB and depth
    roi_img = img[int(y1):int(y2), int(x1):int(x2), :] # apply for all channels
    return roi_img

def clamp_bbox(x1, x2, y1, y2, img_width, img_height):
    # clamp bounding box coordinates to image dimensions, it basically ensures bbox is within image frame
    x1_clamped = max(0, min(x1, img_width - 1))
    x2_clamped = max(0, min(x2, img_width - 1))
    y1_clamped = max(0, min(y1, img_height - 1))
    y2_clamped = max(0, min(y2, img_height - 1))
    return int(x1_clamped), int(y1_clamped), int(x2_clamped), int(y2_clamped)

def expand_bbox(x1, x2, y1, y2, exp_ratio, img_width, img_height):
    # expand bounding box by a certain ratio, ensuring it stays within image dimensions
    bbox_width = x2 - x1
    # bbox_height = y2 - y1
    exp_width = int(bbox_width * exp_ratio) # expanded width (only width expansion for wall fitting)
    # exp_height = int(bbox_height * exp_ratio) # expanded height

    return clamp_bbox(x1 - exp_width/2, x2 + exp_width/2, 
                      y1, y2, 
                      img_width, img_height)

def divide_bbox(rgb_rs, x1, x2, y1, y2, exp_ratio, img_width, img_height, visualize_bbox=True):
    # divide double door bbox into two single door bboxes with some margin in between
    bbox_width = x2 - x1
    door_width = bbox_width / 2
    margin = door_width * exp_ratio  # margin between two doors
    mid_x = (x1 + x2) / 2
    left_x1 = x1
    left_x2 = mid_x - margin / 2
    right_x1 = mid_x + margin / 2
    right_x2 = x2

    left_bbox = clamp_bbox(left_x1, left_x2, y1, y2, img_width, img_height)
    right_bbox = clamp_bbox(right_x1, right_x2, y1, y2, img_width, img_height)

    # visualize divided bboxes on rgb image
    if visualize_bbox:
        cv2.rectangle(rgb_rs, (int(left_bbox[0]), int(left_bbox[1])), (int(left_bbox[2]), int(left_bbox[3])), (255, 0, 0), 2)
        cv2.rectangle(rgb_rs, (int(right_bbox[0]), int(right_bbox[1])), (int(right_bbox[2]), int(right_bbox[3])), (0, 255, 0), 2)
        cv2.imshow("Divided Door Bboxes", rgb_rs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return left_bbox, right_bbox

def ring_mask(img_width, img_height, inner_bbox, outer_bbox, visualize_mask=False):
    # create a ring mask given inner and outer bounding boxes, this is created to construct wall plane points
    mask = np.zeros((img_height, img_width), dtype=bool)
    x1, y1, x2, y2 = inner_bbox
    x1_o, y1_o, x2_o, y2_o = outer_bbox

    # indexes must be integers
    x1_o = int(x1_o)
    # y1_o = int(y1_o)
    x2_o = int(x2_o)
    # y2_o = int(y2_o)
    x1 = int(x1)
    # y1 = int(y1)
    x2 = int(x2)
    # y2 = int(y2)

    mask[y1_o:y2_o, x1_o:x2_o] = True  # outer box is set first
    mask[y1:y2, x1:x2] = False # inner box
    return mask

def project_to_3d(x1, y1, valid_mask=None, depth=None, FX=FX, FY=FY, CX=CX, CY=CY):
    # x1, y1: top-left corner of ROI in full image coordinates
    try:
        if valid_mask is None:
            # get valid depth points
            valid_mask = np.isfinite(depth) & (depth > 0)

        ys, xs = np.where(valid_mask) # get valid pixel coordinates in ROI (coordinates in terms of ROI local)
        Z = depth[ys, xs]  # 1D array of valid depth values in meters

        # convert to full image coordinates
        u = xs + x1
        v = ys + y1
        X = (u - CX) * Z / FX
        Y = (v - CY) * Z / FY
        points_3d = np.stack([X, Y, Z], axis=1)  # (N,3) meters
        return points_3d
    except Exception as e:
        print(f"3D projection failed: {e}")
        return np.array([])