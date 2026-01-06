import base64
from os import path
from ollama import chat
import cv2
import numpy as np
from collections import deque
from fit_plane import fit_plane, run_depth_anything_v2_on_image, get_corrected_depth_image, \
            run_yolo_model, crop_to_bbox_depth, crop_to_bbox_rgb, project_to_3d

EXPANSION_RATIO = 0.5  # ratio to expand door bbox for wall plane fitting

def estimate_door_state_ollama_vlm(image_path):
    # directly use ollama api to estimate door state
    try:
        ok, buf = cv2.imencode('.jpg', cv2.imread(image_path))
        if not ok:
            raise RuntimeError(f"Failed to encode image at {image_path}.")
        rgb_img_bytes = buf.tobytes() 
        img_b64 = base64.b64encode(rgb_img_bytes).decode('utf-8')

        prompt = "Look at this image. Classify the door state as 'open', 'partly-open', or 'closed'. Also tell if any human is present. " \
        "Classify door_type (single/double), door_material (normal/glass)" \
        "Response structure: {'door_state': <state>, 'human_present': <yes/no>, 'door_type': <type>, 'door_material': <material>}. Only respond with the JSON structure."

        response = chat(
            model='qwen3-vl:4b',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }
            ],
            format="json"
        )

        
        # print(f"Ollama API response: {response}")

        res = response.message.content.strip().lower()
        if res:
            door_state = res
            print(f"Estimated door state: {door_state}")
            return door_state
        else:
            print("No valid response received from Ollama API.")
            return None
    
    except Exception as e:
        print(f"Error during estimate_door_state_ollama_api: {e}")
        return None

def clamp_bbox(xmin, xmax, ymin, ymax, img_width, img_height):
    # clamp bounding box coordinates to image dimensions, it basically ensures bbox is within image frame
    xmin_clamped = max(0, min(xmin, img_width - 1))
    xmax_clamped = max(0, min(xmax, img_width - 1))
    ymin_clamped = max(0, min(ymin, img_height - 1))
    ymax_clamped = max(0, min(ymax, img_height - 1))
    return xmin_clamped, xmax_clamped, ymin_clamped, ymax_clamped

def expand_bbox(xmin, xmax, ymin, ymax, exp_ratio, img_width, img_height):
    # expand bounding box by a certain ratio, ensuring it stays within image dimensions
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    exp_width = int(bbox_width * exp_ratio) # expanded width
    exp_height = int(bbox_height * exp_ratio) # expanded height

    return clamp_bbox(xmin - exp_width/2, xmax + exp_width/2, 
                      ymin - exp_height/2, ymax + exp_height/2, 
                      img_width, img_height)

def ring_mask(img_width, img_height, inner_bbox, outer_bbox):
    # create a ring mask given inner and outer bounding boxes, this is created to construct wall plane points
    mask = np.zeros((img_height, img_width), dtype=bool)
    xmin_i, xmax_i, ymin_i, ymax_i = inner_bbox
    xmin_o, xmax_o, ymin_o, ymax_o = outer_bbox

    mask[ymin_o:ymax_o, xmin_o:xmax_o] = True  # outer box is set first
    mask[ymin_i:ymax_i, xmin_i:xmax_i] = False # inner box
    return mask

def estimate_wall_normal(points_3d):
    try:
        # fit plane to depth points in ROI
        inliers, normal_vector, plane_model = fit_plane(points_3d)
        
        if inliers is None or normal_vector is None or plane_model is None:
            print("Plane fitting failed, cannot compute door pose")
            return
    except Exception as e:
        print(f"Error in estimate_wall_normal: {e}")
        return None

def sequential_ransac_plane_fitting(points_3d, door_bbox, img_width, img_height):
    pass

def estimate_door_state(img_path, depth_path, visualize_roi=True):
    try:
        # loads RGB 
        rgb_rs = cv2.imread(img_path) # numpy array HWC
        
        # loads depth map
        depth_rs = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters

        # get RAW depth from DA model (in meters)
        depth_da = run_depth_anything_v2_on_image(rgb_image=rgb_rs)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = get_corrected_depth_image(depth_da=depth_da, model="quad")

        # get bounding box, make detection object
        detections = run_yolo_model(rgb_image=rgb_rs) # runs YOLO model and returns detections
        door_boxes = [item for item in detections if item["cls_id"] == 0]  # assuming class_id 0 is door
        if len(door_boxes) == 0:
            print("No door detected in the image.")
            return
        
        door_box = door_boxes[0] # we only have one door in the scene (safe assumption for now)
        # actual bbox coordinates from detection
        x_min, x_max, y_min, y_max = (int(door_box["xmin"]), int(door_box["xmax"]), int(door_box["ymin"]), int(door_box["ymax"]))
        
        # crop ROI for depth
        roi_depth = crop_to_bbox_depth(depth_da_corr, door_box)

        # expand ROI for wall plane fitting, formulate ring mask
        img_height, img_width = rgb_rs.shape[:2]
        outer_bbox = expand_bbox(x_min, x_max, y_min, y_max, exp_ratio=EXPANSION_RATIO, img_width=img_width, img_height=img_height)
        inner_bbox = (x_min, x_max, y_min, y_max)
        mask = ring_mask(img_width, img_height, inner_bbox, outer_bbox)

        if visualize_roi: # visualize ROI
            roi_rgb = crop_to_bbox_rgb(rgb_rs, door_box) # crop RGB for visualization
            roi_depth_clean = np.nan_to_num(roi_depth, nan=0.0) # replace nan with 0 for visualization
            depth_max = np.max(roi_depth_clean)
            if depth_max > 0:
                depth_normalized = np.clip(roi_depth_clean / depth_max, 0, 1)
                depth_viz = (depth_normalized * 255).astype(np.uint8)
                depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            else:
                depth_viz_color = np.zeros_like(roi_rgb)
            
            try:
                cv2.imshow("ROI RGB", roi_rgb)
                cv2.imshow("ROI Depth", depth_viz_color)
                cv2.imshow("Wall Mask", mask.astype(np.uint8)*255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as viz_error:
                print(f"Visualization skipped (no GUI support): {viz_error}")

        # subsequent plane fitting and door state estimation logic

    except Exception as e:
        print(f"Error during door_state_estimate: {e}")
        return None

if __name__ == "__main__":
    img_id = 27
    img_path = f"/home/RUS_CIP/st184744/codebase/door_navigation/scripts/data_new/latest_image_color_lab_{img_id}.jpg"
    depth_path = f"/home/RUS_CIP/st184744/codebase/door_navigation/scripts/data_new/latest_image_depth_lab_{img_id}.png"
    estimate_door_state(img_path, depth_path)