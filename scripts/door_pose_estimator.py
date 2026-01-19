import time
import rospkg
import sys, os
import numpy as np
import cv2
import json
import open3d as o3d

# ------ path setup -----
try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"[door-pose-estimator] rospkg not available, using relative path: {PACKAGE_PATH}")

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.config import LABEL_MAP
from utils.utils import project_to_3d, expand_bbox, ring_mask, divide_bbox
from utils.visualization import visualize_plane_with_normal, visualize_roi

EXPANSION_RATIO = 0.2  # ratio to expand door bbox for wall plane fitting
MAX_FORWARD_DEVIATION_DEG = 45  # max deviation from camera forward direction (safety constraint)


def get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0):
    # not used currently
    door_centre_x, door_centre_y, door_centre_z = door_centre
    pre_x = door_centre_x + normal_vector[0] * offset_distance
    pre_y = door_centre_y + normal_vector[1] * offset_distance
    pre_z = door_centre_z + normal_vector[2] * offset_distance
    pre_yaw = np.arctan2(door_centre_y - pre_y, door_centre_x - pre_x) # yaw angle in radians
    return pre_x, pre_y, pre_z, pre_yaw

def compute_angle_bisector(normal1, normal2):
    bisector = (normal1 + normal2) / 2.0
    bisector = bisector / np.linalg.norm(bisector)
    return bisector


def fit_plane(points_3d, ply_file_name=""):
    try:
        pcd = o3d.geometry.PointCloud() # empty point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d) # assign points to point cloud

        # save point cloud for visualization/debugging
        if ply_file_name:
            o3d.io.write_point_cloud(f"{ply_file_name}_full.ply", pcd)

        # plane model: ax + by + cz + d = 0, inliers: list of point indices that are inliers to the plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, # 2 cm, as depth are in meters
                                                 ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model

        INLINER_THRESHOLD = 0.3  # at least 30% of points should be inliers to consider valid plane

        # validate plane 
        if len(points_3d) == 0:
            print("No points for plane fitting")
            return None, None, None
                
        inliner_ratio = len(inliers) / len(points_3d)
        
        if inliner_ratio < INLINER_THRESHOLD:
            print("Plane rejected: low inlier ratio")
            return None, None, None
        
        # calculate normal vector and distance
        normal_vector = np.array([a, b, c]) # with zhis we can get any normal, can be either pointing towards or away from camera/origin
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # normalize for unit vector/direction

        # extract actual 3D points from inlier indices
        inlier_points = points_3d[inliers]
        
        # save inlier point cloud for visualization
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
        if ply_file_name:
            o3d.io.write_point_cloud(f"{ply_file_name}_inliers.ply", inlier_pcd)
        
        # camera_fordward vector (pointing outwards from camera/origin)
        camera_forward = np.array([0, 0, 1])  # assuming camera looks along +Z axis (front as per ROS optical frame)
        dot_product = np.dot(normal_vector, camera_forward) # +ve if both in same direction, -ve if opposite direction
        if dot_product > 0: 
            # If normal · +Z > 0: Normal points away from camera
            # If normal · +Z < 0: Normal points towards camera
            # we need to flip normal vector to point towards camera, i.e., outwards from door and towards robot
            # this is need to calculate pre-pose pose infront of door
            normal_vector = -normal_vector
        
        print(f"Normal vector: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
        return inlier_points, normal_vector, plane_model

    except Exception as e:
        print(f"Plane fitting failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def get_normal_vector_single_door(x1, y1, x2, y2, rgb_image, full_depth, visualize=False):
    try:
        print("Fitting wall plane for safer normal vector...")
            
        img_height, img_width = rgb_image.shape[:2]
        inner_bbox = (x1, y1, x2, y2)
            
        # expand ROI for wall plane fitting (sideways expansion only)
        outer_bbox = expand_bbox(x1, x2, y1, y2, exp_ratio=EXPANSION_RATIO, 
                                    img_width=img_width, img_height=img_height)
            
        # create ring mask (only 2 sided wall region)
        exp_mask = ring_mask(img_width, img_height, inner_bbox, outer_bbox)
            
        # fit wall plane
        x1_o, y1_o, _, _ = outer_bbox
        points_3d_wall = project_to_3d(x1_o, y1_o, valid_mask=exp_mask, depth=full_depth)
        wall_inliers, wall_normal, _ = fit_plane(points_3d_wall, "")
        
        if wall_normal is not None:
            print(f"[Single Door] Using wall normal for pre-pose: [{wall_normal[0]:.3f}, {wall_normal[1]:.3f}, {wall_normal[2]:.3f}]")
            normal_vector = wall_normal
            # compute door center from wall inliers (median is robust to outliers)
            door_centre = np.median(wall_inliers, axis=0).astype(np.float32)
            # visualize normal vector
            if visualize:
                visualize_plane_with_normal(wall_inliers, normal_vector)
        else:
            # fit plane for full door region as fallback
            print("Wall plane fit failed, fitting full door region as fallback...")
            width = x2 - x1
            height = y2 - y1
            roi_depth = full_depth[int(y1):int(y1+height), int(x1):int(x1+width)]
            points_3d_door = project_to_3d(x1, y1, roi_depth)
            door_inliers, door_normal, _ = fit_plane(points_3d_door, "")
            if door_normal is not None:
                print(f"[Single Door] Using door normal for pre-pose: [{door_normal[0]:.3f}, {door_normal[1]:.3f}, {door_normal[2]:.3f}]")
                normal_vector = door_normal
                # compute door center from door inliers
                door_centre = np.median(door_inliers, axis=0).astype(np.float32)
                # visualize normal vector
                if visualize:
                    visualize_plane_with_normal(door_inliers, normal_vector)
            else:
                print("[Single Door] Full door plane fit also failed, cannot estimate normal vector")
                return None, None
    
        return normal_vector, door_centre

    except Exception as e:
        print(f"Error in get_normal_vector_single_door: {e}")
        None

def get_normal_vector_double_door(x1, y1, x2, y2, rgb_image, full_depth, visualize=False):
    try:
        # For double doors, use angle bisector of left and right door normals
        print("[Double Door] Computing angle bisector of left and right door normals...")
            
        img_height, img_width = rgb_image.shape[:2]
            
        # Divide double door bbox into left and right halves
        left_bbox, right_bbox = divide_bbox(rgb_image, x1, x2, y1, y2, 
                                               exp_ratio=EXPANSION_RATIO, 
                                               img_width=img_width, 
                                               img_height=img_height,
                                               visualize_bbox=False)
        print(f"[Double Door] Left bbox: {left_bbox}, Right bbox: {right_bbox}")
            
        # Fit plane for left door leaf
        l_x1, l_y1, l_x2, l_y2 = left_bbox
        l_width = l_x2 - l_x1
        l_height = l_y2 - l_y1
        # Crop directly from full depth using full image coordinates
        roi_depth_l = full_depth[int(l_y1):int(l_y2), int(l_x1):int(l_x2)]

        if visualize: # visualize ROI
            visualize_roi(rgb_image, left_bbox, roi_depth_l)

        points_3d_l = project_to_3d(int(l_x1), int(l_y1), depth=roi_depth_l)
        l_inliers, l_door_n, _ = fit_plane(points_3d_l, "")

        if visualize:
            if l_door_n is not None:
                visualize_plane_with_normal(l_inliers, l_door_n)
            
        # Fit plane for right door leaf
        r_x1, r_y1, r_x2, r_y2 = right_bbox
        r_width = r_x2 - r_x1
        r_height = r_y2 - r_y1
        # Crop directly from full depth using full image coordinates
        roi_depth_r = full_depth[int(r_y1):int(r_y2), int(r_x1):int(r_x2)]

        if visualize: # visualize ROI
            visualize_roi(rgb_image, right_bbox, roi_depth_r)

        points_3d_r = project_to_3d(int(r_x1), int(r_y1), depth=roi_depth_r)
        r_inliers, r_door_n, _ = fit_plane(points_3d_r, "")

        if visualize:
            if r_door_n is not None:
                visualize_plane_with_normal(r_inliers, r_door_n)
            
        # Compute angle bisector if both planes fit successfully
        if l_door_n is not None and r_door_n is not None:
            print(f"Left-door-normal: [{l_door_n[0]:.3f}, {l_door_n[1]:.3f}, {l_door_n[2]:.3f}]")
            print(f"Right-door-normal: [{r_door_n[0]:.3f}, {r_door_n[1]:.3f}, {r_door_n[2]:.3f}]")
                
            normal_vector = compute_angle_bisector(l_door_n, r_door_n)
            print(f"[Double Door] Angle bisector: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")

            # Compute door center from combined left and right inliers
            combined_inliers = np.vstack((l_inliers, r_inliers))
            door_centre = np.median(combined_inliers, axis=0).astype(np.float32)
            
            if visualize:
                visualize_plane_with_normal(combined_inliers, normal_vector)
                
            # Apply forward direction constraint for safety
            return normal_vector, door_centre
        elif l_door_n is not None:
            print("Right plane fit failed, using left door normal with constraint")
            normal_vector = l_door_n
            door_centre = np.median(l_inliers, axis=0).astype(np.float32)
            return normal_vector, door_centre
        elif r_door_n is not None:
            print("Left plane fit failed, using right door normal with constraint")
            normal_vector = r_door_n
            door_centre = np.median(r_inliers, axis=0).astype(np.float32)
            return normal_vector, door_centre
        else:
            print("Both left and right plane fits failed, cannot compute door pose")
            return None, None

    except Exception as e:
        print(f"Error in get_normal_vector_double_door: {e}")
        return None, None

def compute_door_3d_pose_from_detection(rgb_image, depth_image, door_box, door_detector, 
                                        door_type='door_single', visualize=True):
    try:
        # get RAW depth from DepthAnything model (in meters)
        depth_da = door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_image)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")
        
        # actual bbox coordinates from detection
        bbox = door_box["bbox"]
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

        # For single doors, use wall normal instead of door normal for safer navigation
        # This prevents robot from going into walls when door is partly open
        if door_type == 'door_single':
            normal_vector, door_centre = get_normal_vector_single_door(x1, y1, x2, y2, rgb_image, depth_da_corr, visualize=visualize)
        else:
            normal_vector, door_centre = get_normal_vector_double_door(x1, y1, x2, y2, rgb_image, depth_da_corr, visualize=visualize)
        
        return door_centre, normal_vector

    except Exception as e:
        print(f"Error computing door 3D pose: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_pose_estimator(img_path, visualize=True):
    # loads RGB 
    from door_ros_interfaces import DoorDetector
    rgb_rs = cv2.imread(img_path) # numpy array HWC

    door_detector = DoorDetector()
    # get RAW depth from DepthAnything model (in meters)
    s_time = time.time()
    depth_da = door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_rs)
    print(f"DAv2 inference time: {time.time() - s_time:.3f} seconds")
    # apply correction to depth_da_raw using pre-computed calibration coefficients
    depth_da_corr = door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")

    # get bounding box, make detection object
    detections = door_detector.run_yolo_model(rgb_image=rgb_rs, visualize=visualize) # runs YOLO model and returns detections

    # decide the door type based on detection (single/double)
    # since door state estimation will run infront of the door, we assume only one door is present in the scene
    door_detections = [(item, LABEL_MAP[item["cls_id"]]) for item in detections if item["cls_id"] in [0, 1]]  # class_id 0 is door_double, class_id 1 is door_single
    if len(door_detections) == 0:
        print("No door detected in the image.")
        return
        
    door_box = door_detections[0][0] # we only have one door in the scene (safe assumption for now)
    door_type = door_detections[0][1]
    print(f"Detected door type: {door_type}, bbox: {door_box['bbox']}")

    s_time = time.time()
    compute_door_3d_pose_from_detection(rgb_image=rgb_rs, 
                                        depth_image=depth_da_corr, 
                                        door_box=door_box,
                                        door_detector=door_detector,
                                        door_type=door_type,
                                        visualize=visualize)
    print(f"Door pose estimation time: {time.time() - s_time:.3f} seconds")


if __name__ == "__main__":

    img_id = 63
    # loads RGB 
    IMAGE_PATH = f"/home/ias/satya/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_{img_id}.jpg"
    test_pose_estimator(IMAGE_PATH, visualize=False)