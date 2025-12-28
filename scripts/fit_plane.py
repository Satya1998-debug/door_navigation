import numpy as np
import cv2
import json
import open3d as o3d
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft

def project_to_3d(xmin, xmax, ymin, ymax, depth, fx, fy, cx, cy):
    # u,v are pixel coordinates
    # fx, fy are effective focal lengths in x and y directions (in camera frame)
    # cx, cy are principal point coordinates (in camera frame)
    points_3d = []
    for v in range(ymin, ymax):
        for u in range(xmin, xmax):
            Z = depth[v, u] * 1000.0  # convert to mm
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points_3d.append([X, Y, Z])
    return np.array(points_3d)

def crop_to_bbox(img, door_box):
    h, w = img.shape
    x_min, y_min, x_max, y_max = door_box["bbox"]

    # croping safely within image bounds
    x_min = max(0, float(x_min))
    y_min = max(0, float(y_min))
    x_max = min(w-1, float(x_max))
    y_max = min(h-1, float(y_max))

    # extract ROI from RGB and depth
    roi_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
    return roi_img, (int(x_min), int(x_max), int(y_min), int(y_max))

def transform_door_center_vector(door_centre, normal_vector, offset_distance=300.0):
    pass

def yaw_to_quaternion(yaw):
    q = tft.quaternion_from_euler(0, 0, yaw)  # roll=0, pitch=0, yaw=yaw
    return q  # returns (x, y, z, w)

def get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0):
    door_centre_x, door_centre_y, door_centre_z = door_centre

    pre_x = door_centre_x + normal_vector[0] * offset_distance
    pre_y = door_centre_y + normal_vector[1] * offset_distance
    pre_z = door_centre_z + normal_vector[2] * offset_distance

    pre_yaw = np.arctan2(door_centre_y - pre_y, door_centre_x - pre_x) # yaw angle in radians

    return pre_x, pre_y, pre_z, pre_yaw

def fit_plane(points_3d):
    try:
        pcd = o3d.geometry.PointCloud() # empty point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d) # assign points to point cloud

        # save point cloud for visualization/debugging
        o3d.io.write_point_cloud("roi_door_plane_points.ply", pcd)

        # plane model: ax + by + cz + d = 0
        plane_model, inliers = pcd.segment_plane(distance_threshold=10.0, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model

        INLINER_THRESHOLD = 0.6  # at least 60% of points should be inliers to consider valid plane

        # validate plane 
        if inliers and len(inliers) / len(points_3d) < INLINER_THRESHOLD:
            print("Fitted plane has insufficient inliers, likely invalid.")
            return
        
        inliner_ratio = len(inliers) / len(points_3d)
        print(f"Fitted plane with inlier ratio: {inliner_ratio:.2f}")
        
        # calculate normal vector and distance
        normal_vector = np.array([a, b, c]) # with zhis we can get any normal, can be either pointing towards or away from camera/origin
        normal_vector = normal_vector / np.linalg.norm(normal_vector)  # normalize for unit vector/direction

        # d < 0 means normal pointing away from camera/origin (normal vector towards +ve half)
        # d > 0 means normal pointing towards camera/origin (normal vector towards -ve half)
        # d = 0 means plane passes through origin

        # camera_fordward vector (pointing outwards from camera/origin)
        camera_forward = np.array([0, 0, 1])  # assuming camera looks along +Z axis
        dot_product = np.dot(normal_vector, camera_forward)
        if dot_product > 0: # normal vector pointing away from camera
            # we need to flip normal vector to point towards camera, i.e., outwards from door and towards robot
            # this is need to calculate pre-pose pose infront of door
            normal_vector = -normal_vector  

    except Exception as e:
        print(f"Plane fitting failed: {e}")
        return

def compute_door_pose_goal():
    try:

        # loads RGB 
        IMAGE_PATH = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data/latest_image_color_11.jpg"
        rgb_img = cv2.imread(IMAGE_PATH) # numpy array HWC
        
        # loads depth map
        DEPTH_PATH = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data/latest_image_color_11_da2_metric_depth.npy"
        depth_img = np.load(DEPTH_PATH) # numpy arrays, Units: meters, shape HW

        # get xyxy bbox coordinates
        detection_json_path = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data/detections.json"
        detections = json.load(open(detection_json_path, 'r'))
        img_detection = detections[f"{IMAGE_PATH.split('/')[-1].replace('.jpg', '')}"]
        boxes = img_detection['boxes']  # list of [x1, y1, x2, y2]
        door_boxes = [box for box, cls_id in zip(boxes, img_detection['class_ids']) if cls_id == 0]  # assuming class_id 0 is door
        if len(door_boxes) == 0:
            print("No door detected in the image.")
            return
        door_box = door_boxes[0]  # take first detected door
        
        roi_depth, (xmin, xmax, ymin, ymax) = crop_to_bbox(depth_img, door_box)
        # roi_rgb, _ = crop_to_bbox(rgb_img, door_box)

        # get valid depth points
        valid_mask = np.isfinite(roi_depth) & (roi_depth > 0)
        valid_mask &= (roi_depth < 6.0)  # filter out depths beyond 6 meters, as it gives infinite plane
        valid_mask &= (roi_depth > 0.2)  # filter out depths below 0.2 meters (20 cm)
        valid_depth_img = roi_depth[valid_mask] # 1D array for valid == True

        # get intrinsic parameters (in mm)
        fx, fy = 646.21, 645.38  # example focal lengths
        cx, cy = 636.35, 240.0  # example principal point

        # project valid depth points to 3D
        points_3d = project_to_3d(xmin, xmax, ymin, ymax, valid_depth_img, fx=646.21, fy=645.38, cx=636.35, cy=240.0)

        # fit plane to depth points in ROI
        inliers, normal_vector = fit_plane(points_3d)

        # get door centre
        door_centre_mean = np.mean(inliers, axis=0).astype(np.float16)  # mean of all 3D points, all in mm
        door_centre_median = np.median(inliers, axis=0).astype(np.float16)  # median of all 3D points, all in mm

        # calculate a point infront of door along normal vector (pre-door pose)
        pre_x, pre_y, pre_z, pre_yaw = get_pre_door_pose(door_centre_mean, normal_vector, offset_distance=1.0)  # 1m infront of door 

        # TODO: publish pre-door pose and door normal vector for door navigation
        # TODO: transform from camera frame to robot base frame if needed

    except Exception as e:
        print(f"Error in fit_roi_plane: {e}")
        return

if __name__ == "__main__":
    compute_door_pose_goal()