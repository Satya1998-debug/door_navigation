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

from utils.utils import crop_to_bbox_depth, crop_to_bbox_rgb
from utils.utils import project_to_3d


def get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0):
    # not used currently
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

def compute_door_3d_pose_from_detection(rgb_image, depth_image, door_box, door_detector, visualize_roi=False):
    """
    Compute 3D door pose (center and normal) from RGB-D images and door detection.
    
    Args:
        rgb_image: RGB image (numpy array HWC)
        depth_image: Depth image in meters (numpy array HW)
        door_box: Door detection dict with 'bbox' key [x1, y1, x2, y2]
        door_detector: DoorDetector instance for depth processing
        visualize_roi: Whether to visualize cropped regions
    
    Returns:
        tuple: (door_centre, normal_vector, inliers) or (None, None, None) if failed
            - door_centre: 3D position in camera frame (numpy array [x, y, z])
            - normal_vector: 3D normal vector (numpy array [x, y, z])
            - inliers: 3D points that fit the plane (numpy array Nx3)
    """
    try:
        # get RAW depth from DepthAnything model (in meters)
        depth_da = door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_image)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")
        
        # actual bbox coordinates from detection
        bbox = door_box["bbox"]
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        
        # Crop depth to door region
        roi_depth = crop_to_bbox_depth(depth_da_corr, door_box)

        if visualize_roi:
            roi_rgb = crop_to_bbox_rgb(rgb_image, door_box["bbox"]) # crop RGB for visualization
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

        # project valid depth points to 3D in camera frame
        points_3d = project_to_3d(x1, y1, roi_depth)        

        # fit plane to depth points in ROI
        inliers, normal_vector, plane_model = fit_plane(points_3d, "")
        
        if inliers is None or normal_vector is None or plane_model is None:
            print("Plane fitting failed, cannot compute door pose")
            return None, None, None

        # get door centre in camera frame (median is more robust to outliers than mean)
        door_centre = np.median(inliers, axis=0).astype(np.float32)  # median of all 3D points in meters
        
        return door_centre, normal_vector, inliers

    except Exception as e:
        print(f"Error computing door 3D pose: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    compute_door_3d_pose_from_detection()