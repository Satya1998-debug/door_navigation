#!/home/satya/MT/uv_ros_py38/bin python3

import sys, os
import numpy as np
import cv2
import json
import open3d as o3d

from door_navigation.scripts.door_ros_interfaces import DoorDetector
from door_navigation.scripts.utils.depth_calibration import run_depth_anything_v2_on_image, get_corrected_depth_image
from door_navigation.scripts.utils.config import IMG_SIZE, CONFIDENCE_THRESHOLD, DETECTION_JSON_PATH, MODEL_PATH, LABEL_MAP
from door_navigation.scripts.utils.utils import crop_to_bbox_depth, crop_to_bbox_rgb
from door_navigation.scripts.utils.visualization import visualize_plane_with_normal
from door_navigation.scripts.utils.utils import project_to_3d

def get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0):
    door_centre_x, door_centre_y, door_centre_z = door_centre

    pre_x = door_centre_x + normal_vector[0] * offset_distance
    pre_y = door_centre_y + normal_vector[1] * offset_distance
    pre_z = door_centre_z + normal_vector[2] * offset_distance

    pre_yaw = np.arctan2(door_centre_y - pre_y, door_centre_x - pre_x) # yaw angle in radians

    return pre_x, pre_y, pre_z, pre_yaw

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

def compute_door_pose_goal(visualize_roi=True, 
                           visualize_3d_pcd=True, 
                           visualize_plane=True, 
                           visualize_normal=True):
    try:

        # loads RGB 
        IMAGE_PATH = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_35.jpg"
        rgb_rs = cv2.imread(IMAGE_PATH) # numpy array HWC
        
        # loads depth map
        RAW_RS_DEPTH_PATH = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_depth_lab_35.png"
        depth_rs = cv2.imread(RAW_RS_DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # convert mm to meters

        door_detector = DoorDetector()  # initialize door detector
        # get RAW depth from DepthAnything model (in meters)
        depth_da = door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_rs)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")

        # get bounding box, make detection object
        detections = door_detector.run_yolo_model(rgb_image=rgb_rs) # runs YOLO model and returns detections
        door_boxes = [item for item in detections if item["cls_id"] == 0]  # assuming class_id 0 is door
        if len(door_boxes) == 0:
            print("No door detected in the image.")
            return
        
        # take all door and run plane fitting, for now take first door only
        door_box = door_boxes[0]
        bbox = door_box["bbox"]
        # actual bbox coordinates from detection
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        
        # shape check
        roi_depth = crop_to_bbox_depth(depth_da_corr, door_box)

        if visualize_roi:
            roi_rgb = crop_to_bbox_rgb(rgb_rs, door_box["bbox"]) # crop RGB for visualization
            roi_depth_clean = np.nan_to_num(roi_depth, nan=0.0) # replace nan with 0 for visualization
            depth_max = np.max(roi_depth_clean)
            if depth_max > 0:
                depth_normalized = np.clip(roi_depth_clean / depth_max, 0, 1)
                depth_viz = (depth_normalized * 255).astype(np.uint8)
                depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
            else:
                depth_viz_color = np.zeros_like(roi_rgb)
            
            cv2.imshow("ROI RGB", roi_rgb)
            cv2.imshow("ROI Depth", depth_viz_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # project valid depth points to 3D
        # convert depth to mm before projection, use camera intrinsics from config (all in mm), from aligned depth to color
        points_3d = project_to_3d(x1, y1, roi_depth)        

        # fit plane to depth points in ROI
        inliers, normal_vector, plane_model = fit_plane(points_3d, "roi_door_plane_points_full")
        
        if inliers is None or normal_vector is None or plane_model is None:
            print("Plane fitting failed, cannot compute door pose")
            return

        # get door centre
        # door_centre = np.mean(inliers, axis=0).astype(np.float32)  # mean of all 3D points in meters
        
        visualize_plane_with_normal(inliers, 
                                    normal_vector=normal_vector)
        # calculate a point infront of door along normal vector (pre-door pose)
        door_centre = np.median(inliers, axis=0).astype(np.float32)  # median of all 3D points in meters
        pre_x, pre_y, pre_z, pre_yaw = get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0)  # 1m infront of door 

        print(f"Pre-door pose: X={pre_x:.3f}m, Y={pre_y:.3f}m, Z={pre_z:.3f}m, Yaw={np.degrees(pre_yaw):.2f}deg")

        # TODO: publish pre-door pose and door normal vector for door navigation
        # TODO: transform from camera frame to robot base frame if needed

    except Exception as e:
        print(f"Error in fit_roi_plane: {e}")
        return


if __name__ == "__main__":
    compute_door_pose_goal()