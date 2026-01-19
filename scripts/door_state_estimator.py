import base64
from ollama import chat
import cv2
import numpy as np
from door_pose_estimator import fit_plane, project_to_3d, visualize_plane_with_normal
from utils.visualization import visualize_door_passability, visualize_roi
from utils.utils import crop_to_bbox_depth, expand_bbox, divide_bbox, ring_mask

FX = 385.88861083984375
FY = 385.3906555175781
CX = 317.80999755859375
CY = 243.65032958984375

import ctypes
# Fix for PyTorch libgomp TLS allocation issue - preload libgomp before torch imports
try:
    ctypes.CDLL("libgomp.so.1", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass  # libgomp already loaded or not found

ROBOT_WIDTH = 0.5  # in meters, robot width for door pass check
EXPANSION_RATIO = 0.2  # ratio to expand door bbox for wall plane fitting
LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}
S_DOOR_OPEN_THRESHOLD = 60
S_DOOR_CLOSED_THRESHOLD = 15
S_DOOR_MAX_ANGLE = 90
D_DOOR_OPEN_THRESHOLD = 60
D_DOOR_CLOSED_THRESHOLD = 30
SAFETY_MARGIN_WIDTH = 0.1  # in meters, additional margin for door pass check
SLAB_THICKNESS = 0.3  # in meters, thickness of depth slab for door pass check
MIN_POINTS_PASS_CHECK = 30  # minimum valid depth points in slab region for door pass check

def estimate_door_state_ollama_vlm(rgb_img, is_passable="", door_open_percent="", door_wall_angle="", left_right_door_angle="", door_type=""):
    # directly use ollama api to estimate door state
    try:
        # Encode OpenCV image (BGR) as JPEG
        ok, buf = cv2.imencode('.jpg', rgb_img)
        if not ok:
            raise RuntimeError(f"Failed to encode image.")
        rgb_img_bytes = buf.tobytes() 
        img_b64 = base64.b64encode(rgb_img_bytes).decode('utf-8')

        prompt = f"""
            You are a robot perception assistant.

            Your task is to VISUALLY VERIFY the door state in the image.
            You are NOT the primary decision maker.

            Classify the door state using ONLY one of:
            - "open"
            - "semi_open"
            - "closed"
            - "unknown"

            Also detect whether a human is clearly visible.

            You may use the following preliminary information (may be noisy):
            - is door passable: {is_passable}
            - door_open_percent (only in single door): {door_open_percent}
            - door_wall_angle (only in single door): {door_wall_angle}
            - left_right_door_angle (only in double door): {left_right_door_angle}
            - door_type: {door_type}

            Rules:
            - If the door is clearly closed, choose "closed".
            - If the door is fully open or no door leaf blocks the opening, choose "open".
            - If the door is partially open, choose "semi_open".
            - If the image is ambiguous or occluded, choose "unknown".
            - If a human is visible near the door, set human_present = "yes", otherwise "no".
            - If someone is in the path of the door, consider them present and request them to clear the way.
            - Do NOT guess human intent.
            - Do NOT explain your reasoning.
            - Do NOT output anything except the JSON object.

            Then generate a SHORT, polite, single-sentence spoken request appropriate for the situation:
            - If closed and human present → request to open the door.
            - If partly_open and human present → request to open the door more.
            - If open and human present → request to keep the door open.
            - If no human present → polite neutral message or empty string.

            Output STRICTLY in the following JSON format:
            {{
            "door_state": "<open|semi_open|closed|unknown>",
            "human_present": "<yes|no>",
            "conversation": "<single short sentence or empty string>"
            }}
            """


        response = chat(
            # model='qwen3-vl:4b',
            model='qwen3-vl:2b-instruct-q8_0',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [img_b64],
                }
            ],
            format="json"
        )

        """
        res = {
            'door_state': 'open',
            'human_present': 'no',
            'conversation': 'please open the door'
        }
        """
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

def calculate_door_opening_angle(n1, n2):
    ang = np.arccos(np.clip(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)), -1.0, 1.0))
    angle_deg = np.degrees(ang)
    return angle_deg

def calculate_door_state_single(angle_deg, 
                                open_threshold=S_DOOR_OPEN_THRESHOLD, 
                                closed_threshold=S_DOOR_CLOSED_THRESHOLD):
    door_open_percent = angle_deg / S_DOOR_MAX_ANGLE * 100.0
    if door_open_percent > 100:
        door_open_percent = -1.0 # this means invalid value

    if angle_deg >= open_threshold:
        return 'open', door_open_percent
    elif angle_deg <= closed_threshold:
        return 'closed', door_open_percent
    else:
        return 'semi_open', door_open_percent
    
def calculate_door_state_double(angle_deg, 
                                open_threshold=D_DOOR_OPEN_THRESHOLD, 
                                closed_threshold=D_DOOR_CLOSED_THRESHOLD,
                                is_passable=None):
    if is_passable:
        if angle_deg >= open_threshold:
            return 'open'
        else:
            return 'semi_open'
    else:  #  not passable 
        if angle_deg <= closed_threshold:
            return 'closed'
        else:
            return 'semi_open'

def is_door_passable(depth, bbox, FX, CX, 
                     robot_width=ROBOT_WIDTH, safety_margin=SAFETY_MARGIN_WIDTH, 
                     depth_slab_thickness=SLAB_THICKNESS, stride=2, min_points=MIN_POINTS_PASS_CHECK,
                     visualize=False, visualize_3d=False):
    
    x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    # depth ponts inside bbox in the slab region
    xs = np.arange(x1, x2, stride) # interms of full image coordinates
    ys = np.arange(y1, y2, stride) # interms of full image coordinates
    xv, yv = np.meshgrid(xs, ys)

    z = depth[yv, xv]  # depth values (obtained for the 2D meshgrid points) from Full depth
    
    valid_mask = np.isfinite(z) & (z > 0)
    if np.sum(valid_mask) < min_points:
        print("Not enough valid depth points in slab region for door pass check.")
        return 'unknown', 0.0
    
    xv_valid = xv[valid_mask] # keeping only those meshgrid (horizontal) x points (which has valid depths)
    yv_valid = yv[valid_mask] # keeping only those meshgrid (vertical) y points (which has valid depths)
    z = z[valid_mask] # depth values for the 2d meshgrid points

    # door depth points in meters
    z_center = np.median(z) # centre depth for slab placement
    print(f"Slab center depth: {z_center:.2f} meters")
    print(f"Slab thickness: {depth_slab_thickness:.2f} meters (±{depth_slab_thickness/2:.2f}m)")

    # slab mask (about depth centre)
    slab_mask = (z >= (z_center - depth_slab_thickness/2)) & (z <= (z_center + depth_slab_thickness/2))
    slab_xv = xv_valid[slab_mask]
    slab_yv = yv_valid[slab_mask]
    slab_z = z[slab_mask]

    print(f"Valid points in slab: {len(slab_xv)} / {len(z)} total valid points")

    # convert to 3D points, slab 3D points
    X = (slab_xv - CX) * slab_z / FX
    Y = (slab_yv - CY) * slab_z / FY  # Using CY and FY
    
    # All valid points in 3D
    X_all = (xv_valid - CX) * z / FX
    Y_all = (yv_valid - CY) * z / FY

    # gap based passibility
    num_bins = 30  # tuneable
    x_min, x_max = np.min(X), np.max(X) # to check horizontal width
    bins = np.linspace(x_min, x_max, num_bins + 1)

    occupied = np.zeros(num_bins, dtype=bool)

    for x in X:
        idx = np.searchsorted(bins, x) - 1
        if 0 <= idx < num_bins:
            occupied[idx] = True

    # largest continuous free gap
    # https://websites.umich.edu/~ykoren/uploads/The_Vector_Field_HistogramuFast_Obstacle_Avoidance.pdf
    max_free_bins = 0
    current_free = 0

    for occ in occupied:
        if not occ:
            current_free += 1
            max_free_bins = max(max_free_bins, current_free)
        else:
            current_free = 0

    bin_width = (x_max - x_min) / num_bins
    max_free_width = max_free_bins * bin_width

    print(f"Max free opening width: {max_free_width:.2f} meters")
    robot_effective_width = robot_width + safety_margin  # add margin
    passable = max_free_width >= robot_effective_width
    print(f"Robot effective width (with margin): {robot_effective_width:.2f} meters")
    
    # door is passable if the measured opening width is >= robot's required width
    print(f"Passability result: {passable}")

    # Visualization
    if visualize or visualize_3d:
        visualize_door_passability(depth, bbox, xv_valid, yv_valid, slab_xv, slab_yv,
                                   X, Y, X_all, Y_all, z, slab_z, z_center, depth_slab_thickness, robot_effective_width,
                                   passable, num_bins=30, visualize_2d=True, visualize_3d=True)
    
    return passable

def estimate_single_door_state(door_bbox, rgb_rs, roi_depth, full_depth, visualize=True, use_vlm=False):
    try:
        # get bbox coordinates (also the inner bbox for wall fitting)
        if len(door_bbox) == 0:
            print("Empty door box provided for single door state estimation.")
            return None
        
        x1, y1, x2, y2 = (int(door_bbox[0]), int(door_bbox[1]), int(door_bbox[2]), int(door_bbox[3]))
        inner_bbox = (x1, y1, x2, y2)
        print(f"Inner bbox for wall fitting: {inner_bbox}")

        # expand ROI for wall plane fitting, formulate ring mask (only 2 sided wall region)
        img_height, img_width = rgb_rs.shape[:2]
        print(f"Image dimensions: width={img_width}, height={img_height}")

        # get outer bbox (sideways expansion only)
        outer_bbox = expand_bbox(x1, x2, y1, y2, exp_ratio=EXPANSION_RATIO, img_width=img_width, img_height=img_height)
        print(f"Outer bbox for wall fitting: {outer_bbox}")
        
        exp_mask = ring_mask(img_width, img_height, inner_bbox, outer_bbox)
        print(f"Ring mask shape: {exp_mask.shape}")

        if visualize: # visualize ROI
            visualize_roi(rgb_rs, door_bbox, roi_depth)
        
        # fit plane for door
        points_3d_door = project_to_3d(x1, y1, valid_mask=None, depth=roi_depth)
        door_inliers, door_n, _ = fit_plane(points_3d_door, "singledoor_roi_plane-door")
        # visualize door plane with normal
        if visualize:
            visualize_plane_with_normal(door_inliers, normal_vector=door_n)

        # fit wall plane
        x1_o, y1_o, _, _ = outer_bbox
        points_3d_wall = project_to_3d(x1_o, y1_o, valid_mask=exp_mask, depth=full_depth)
        wall_inliers, wall_n, _ = fit_plane(points_3d_wall, "singledoor_roi_plane-wall")
        # visualize wall plane with normal
        if visualize:
            visualize_plane_with_normal(wall_inliers, normal_vector=wall_n)

        # calculate door opening angle
        door_opening_angle = calculate_door_opening_angle(door_n, wall_n)
        print(f"Estimated door opening angle: {door_opening_angle} degrees")

        # door pass check
        is_passable = is_door_passable(full_depth, door_bbox, FX, CX, visualize=visualize, visualize_3d=visualize)

        # door state, open percent (NOTE: geometrically to take decision)
        door_state, door_open_percent = calculate_door_state_single(door_opening_angle)
        print(f"Door state based on angle thresholds: {door_state}, open percent: {door_open_percent:.2f}%")

        # VLM based door state estimation (falls back to geometric)

        if use_vlm:
            door_state_vlm = estimate_door_state_ollama_vlm(rgb_rs, is_passable=is_passable, 
                                                            door_open_percent=door_open_percent,
                                                            door_wall_angle=door_opening_angle,
                                                            door_type="single")
            return door_state_vlm

        # TODO: Audio generation based on door state and human presence (door_state_vlm output)

        # calculate post door pose
        return door_state

    except Exception as e:
        print(f"Error in estimate_single_door_state: {e}")
        return None

def estimate_double_door_state(door_bbox, rgb_rs, roi_depth, full_depth, visualize=True, use_vlm=False):
    try:

        if len(door_bbox) == 0:
            print("Empty door box provided for double door state estimation.")
            return None
        
        # get bbox coordinates
        x1, y1, x2, y2 = (int(door_bbox[0]), int(door_bbox[1]), int(door_bbox[2]), int(door_bbox[3]))

        img_height, img_width = rgb_rs.shape[:2]
        print(f"Image dimensions: width={img_width}, height={img_height}")

        # divide the double door bbox into two single door bboxes
        left_bbox, right_bbox = divide_bbox(rgb_rs, x1, x2, y1, y2, exp_ratio=EXPANSION_RATIO, img_width=img_width, img_height=img_height)
        print(f"Left door bbox: {left_bbox}, Right door bbox: {right_bbox}")
        
        if visualize: # visualize ROI
            visualize_roi(rgb_rs, door_bbox, roi_depth)

        # fit plane for left door
        points_3d_door_left = project_to_3d(left_bbox[0], left_bbox[1], valid_mask=None, depth=roi_depth)
        door_l_inliners, door_l_n, _ = fit_plane(points_3d_door_left)
        if visualize:
            visualize_plane_with_normal(door_l_inliners, normal_vector=door_l_n)

        # fit plane for right door
        points_3d_door_right = project_to_3d(right_bbox[0], right_bbox[1], valid_mask=None, depth=roi_depth)
        door_r_inliners, door_r_n, _ = fit_plane(points_3d_door_right)
        if visualize:
            visualize_plane_with_normal(door_r_inliners, normal_vector=door_r_n)
        
        # calculate angle between two door normals
        side_doors_angle = calculate_door_opening_angle(door_l_n, door_r_n)
        print(f"Estimated door opening angle: {side_doors_angle} degrees")

        # door pass check
        is_passable = is_door_passable(full_depth, door_bbox, FX, CX, visualize=visualize, visualize_3d=visualize)

        # door state, open percent (NOTE: geometrically to take decision)
        door_state = calculate_door_state_double(side_doors_angle, is_passable=is_passable)
        
        # VLM based door state estimation
        if use_vlm:
             door_state_vlm = estimate_door_state_ollama_vlm(rgb_rs, is_passable=is_passable, 
                                                            left_right_door_angle=side_doors_angle,
                                                            door_type="double")
             return door_state_vlm
       
        return door_state

    except Exception as e:
        print(f"Error in estimate_double_door_state: {e}")
        return None

def estimate_door_state(img_path, depth_path, visualize=True):
    # NOT used currently (only for testing)
    # NOTE: this is executed at the Pre-Pose stage, before robot moves through the door
    try:
        # loads RGB 
        rgb_rs = cv2.imread(img_path) # numpy array HWC
        
        # loads depth map
        depth_rs = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters

        # get RAW depth from DA model (in meters)
        depth_da = run_depth_anything_v2_on_image(rgb_image=rgb_rs)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = get_corrected_depth_image(depth_da=depth_da, model="quad")
        # depth_da_corr = depth_rs

        # get bounding box, make detection object
        detections = run_yolo_model(rgb_image=rgb_rs) # runs YOLO model and returns detections

        # decide the door type based on detection (single/double)
        # since door state estimation will run infront of the door, we assume only one door is present in the scene
        door_detections = [(item, LABEL_MAP[item["cls_id"]]) for item in detections if item["cls_id"] in [0, 1]]  # class_id 0 is door_double, class_id 1 is door_single
        if len(door_detections) == 0:
            print("No door detected in the image.")
            return
        
        door_box = door_detections[0][0] # we only have one door in the scene (safe assumption for now)
        door_type = door_detections[0][1]
        print(f"Detected door type: {door_type} with confidence {door_box['conf']:.2f}")
        
        # crop ROI for depth, based on actual bbox
        roi_depth = crop_to_bbox_depth(depth_da_corr, door_box)
        full_depth = depth_da_corr

        # NOTE: door estimation for Single Door
        if door_type == 'door_single':
            door_state = estimate_single_door_state(door_box.get("bbox", []), rgb_rs, roi_depth, full_depth, visualize=visualize, use_vlm=False)
            print(f"Estimated single door state: {door_state}")
            return door_state

        # NOTE: door estimation for Double Door
        elif door_type == 'door_double':
            door_state = estimate_double_door_state(door_box.get("bbox", []), rgb_rs, roi_depth, full_depth, visualize=visualize, use_vlm=False)
            print(f"Estimated double door state: {door_state}")
            return door_state
        else:
            print(f"Unknown door type: {door_type}, cannot estimate door state.")
            return None
        # subsequent plane fitting and door state estimation logic

    except Exception as e:
        print(f"Error during door_state_estimate: {e}")
        return None

        
if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # single door: 19(normal-closed), 63(normal-semi-open), 66(glass-closed)
    # double door: 27(glass-closed), 30(glass-closed), 35(glass-semi-open)
    img_id = 19
    img_path = os.path.join(script_dir, f"data_new/latest_image_color_lab_{img_id}.jpg")
    depth_path = os.path.join(script_dir, f"data_new/latest_image_depth_lab_{img_id}.png")
    estimate_door_state(img_path, depth_path)