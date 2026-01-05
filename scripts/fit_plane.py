import sys, os
import numpy as np
import cv2
import json
import open3d as o3d
from geometry_msgs.msg import PoseStamped
import tf.transformations as tft
from depth_calibration import run_depth_anything_v2_on_image, get_corrected_depth_image
from config import IMG_SIZE, CONFIDENCE_THRESHOLD, DETECTION_JSON_PATH, MODEL_PATH, LABEL_MAP, FX, FY, CX, CY

def project_to_3d(xmin, ymin, valid_mask, roi_depth, FX=FX, FY=FY, CX=CX, CY=CY):
    try:
        # filter out depth outliers using percentiles (this is because the doors, often contains background, holes, weird monocular artifacts.)
        z = roi_depth[valid_mask]
        # z_lo, z_hi = np.percentile(z, [10, 90])
        # valid_mask = valid_mask & (roi_depth >= z_lo) & (roi_depth <= z_hi)

        ys, xs = np.where(valid_mask) # get valid pixel coordinates in ROI (coordinates in terms of ROI local)
        Z = roi_depth[ys, xs]  # 1D array of valid depth values in meters

        # convert to full image coordinates
        u = xs + xmin
        v = ys + ymin
        X = (u - CX) * Z / FX
        Y = (v - CY) * Z / FY
        points_3d = np.stack([X, Y, Z], axis=1)  # (N,3) meters
        return points_3d
    except Exception as e:
        print(f"3D projection failed: {e}")
        return np.array([])

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
    return roi_img, (int(x_min), int(x_max), int(y_min), int(y_max))

def crop_to_bbox_rgb(img, door_box):
    h, w, _ = img.shape
    x_min, y_min, x_max, y_max = door_box["bbox"]

    # croping safely within image bounds
    x_min = max(0, float(x_min))
    y_min = max(0, float(y_min))
    x_max = min(w-1, float(x_max))
    y_max = min(h-1, float(y_max))

    # extract ROI from RGB and depth
    roi_img = img[int(y_min):int(y_max), int(x_min):int(x_max), :] # apply for all channels
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

def run_yolo_model(model_path=MODEL_PATH, 
                   rgb_image=None, 
                   img_size=IMG_SIZE, 
                   confidence_threshold=CONFIDENCE_THRESHOLD):
    try:
        from ultralytics import YOLO

        if rgb_image is None:
            print("No RGB image provided for YOLO model inference.")
            return

        # load the model
        model = YOLO(model_path)

        valid_boxes = []
        jsonable_valid_boxes = []

        detections = dict()

        # run inference
        results = model(source=rgb_image, imgsz=img_size, conf=confidence_threshold)

        # print results
        for result in results:
            print(f"got {len(result.boxes)} boxes in test image")
            for i, box in enumerate(result.boxes):
                print("for box-{}:".format(i))
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # get bbox coordinates
                print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, BBox: {bbox}")

                # filter boxes above confidence threshold
                if conf >= confidence_threshold:
                    print(f"Detected door with confidence {conf:.2f} at bbox {bbox}")
                    valid_boxes.append({
                        'cls_id': cls_id,  # 0 for door, 1 for handle
                        'conf': conf,
                        'bbox': bbox
                    })
                    jsonable_valid_boxes.append({
                        'cls_id': cls_id,  # 0 for door, 1 for handle
                        'conf': conf,
                        'bbox': bbox.tolist()  # convert numpy array to list for json serialization
                    })
            detections.update({
                'valid_detections': jsonable_valid_boxes
            })

        # write detections to json file
        with open(os.path.join(DETECTION_JSON_PATH), 'w') as f:
            import json
            json.dump(detections, f, indent=4)

        color_image = rgb_image.copy()
        for vb in valid_boxes:
            x1, y1, x2, y2 = map(int, vb['bbox'])
            conf = vb['conf']
            cls_id = vb['cls_id']
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{LABEL_MAP.get(cls_id, 'Unknown')} {conf:.2f}"
            cv2.putText(color_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("Test Image Detections", color_image)
        # handle cv2 events and check for ESC key to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return valid_boxes

    except Exception as e:
        print("Error testing model: %s", e)

def fit_plane(points_3d):
    try:
        pcd = o3d.geometry.PointCloud() # empty point cloud
        pcd.points = o3d.utility.Vector3dVector(points_3d) # assign points to point cloud

        # save point cloud for visualization/debugging
        o3d.io.write_point_cloud("roi_door_plane_points_full.ply", pcd)

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

        # Extract actual 3D points from inlier indices
        inlier_points = points_3d[inliers]
        
        # Save inlier point cloud for visualization
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_points)
        o3d.io.write_point_cloud("roi_door_plane_inliers.ply", inlier_pcd)
        print(f"Saved {len(inlier_points)} inlier points to roi_door_plane_inliers.ply")
        
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

        # get RAW depth from DepthAnything model (in meters)
        depth_da = run_depth_anything_v2_on_image(rgb_image=rgb_rs)
        # apply correction to depth_da_raw using pre-computed calibration coefficients
        depth_da_corr = get_corrected_depth_image(depth_da=depth_da, model="quad")

        # get bounding box, make detection object
        detections = run_yolo_model(rgb_image=rgb_rs) # runs YOLO model and returns detections
        door_boxes = [item for item in detections if item["cls_id"] == 0]  # assuming class_id 0 is door
        if len(door_boxes) == 0:
            print("No door detected in the image.")
            return
        
        # take all door and run plane fitting, for now take first door only
        door_box = door_boxes[0] 
        
        # shape check
        roi_depth, (xmin, xmax, ymin, ymax) = crop_to_bbox_depth(depth_da_corr, door_box)

        if visualize_roi:
            roi_rgb, _ = crop_to_bbox_rgb(rgb_rs, door_box) # crop RGB for visualization
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

        # get valid depth points
        valid_mask = np.isfinite(roi_depth) & (roi_depth > 0)
        # valid_mask &= (roi_depth < 6.0)  # filter out depths beyond 6 meters, as it gives infinite plane
        # valid_mask &= (roi_depth > 0.2)  # filter out depths below 0.2 meters (20 cm)
        
        # project valid depth points to 3D
        # convert depth to mm before projection, use camera intrinsics from config (all in mm), from aligned depth to color
        points_3d = project_to_3d(xmin, ymin, valid_mask, roi_depth)        

        # fit plane to depth points in ROI
        inliers, normal_vector, plane_model = fit_plane(points_3d)
        
        if inliers is None or normal_vector is None or plane_model is None:
            print("Plane fitting failed, cannot compute door pose")
            return

        # get door centre
        # door_centre = np.mean(inliers, axis=0).astype(np.float32)  # mean of all 3D points in meters
        door_centre = np.median(inliers, axis=0).astype(np.float32)  # median of all 3D points in meters
        
        print(f"Door center: X={door_centre[0]:.3f}m, Y={door_centre[1]:.3f}m, Z={door_centre[2]:.3f}m")

        visualize_plane_with_normal(inliers, 
                                    plane_model=plane_model,
                                    door_center=door_centre,
                                    normal_vector=normal_vector)
        # calculate a point infront of door along normal vector (pre-door pose)
        pre_x, pre_y, pre_z, pre_yaw = get_pre_door_pose(door_centre, normal_vector, offset_distance=1.0)  # 1m infront of door 

        print(f"Pre-door pose: X={pre_x:.3f}m, Y={pre_y:.3f}m, Z={pre_z:.3f}m, Yaw={np.degrees(pre_yaw):.2f}deg")

        # TODO: publish pre-door pose and door normal vector for door navigation
        # TODO: transform from camera frame to robot base frame if needed

    except Exception as e:
        print(f"Error in fit_roi_plane: {e}")
        return

def visualize_plane_with_normal(inlier_points, plane_model, door_center, normal_vector):
    a, b, c, d = plane_model
    
    # point cloud from inliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inlier_points)
    pcd.paint_uniform_color([0.0, 0.7, 0.0])  # green for visualization of inlier points
    
    # plane mesh (large rectangle aligned with plane)
    # extent of points to size the plane
    extent = np.max(np.abs(inlier_points - door_center), axis=0)
    size = np.max(extent) * 1.5 # make plane larger than point extent
    
    # generating plane mesh vertices in plane coordinate system
    plane_mesh = o3d.geometry.TriangleMesh()
    vertices = []
    # tangent vectors perpendicular to normal for plane rectangle
    if abs(normal_vector[0]) < 0.9:
        tangent1 = np.cross(normal_vector, [1, 0, 0])
    else:
        tangent1 = np.cross(normal_vector, [0, 1, 0])
    tangent1 = tangent1 / np.linalg.norm(tangent1) # tangent unit vector
    tangent2 = np.cross(normal_vector, tangent1)  # second tangent vector for plane
    
    # 4 corners of plane rectangle
    vertices = [
        door_center + size * tangent1 + size * tangent2,
        door_center - size * tangent1 + size * tangent2,
        door_center - size * tangent1 - size * tangent2,
        door_center + size * tangent1 - size * tangent2,
    ]
    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector([[0,1,2], [0,2,3]])
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.0])  # yellow plane for visualization of door plane
    plane_mesh.compute_vertex_normals()
    
    # normal vector arrow
    arrow_length = 0.5  # 50cm arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.01,
        cone_radius=0.02,
        cylinder_height=arrow_length * 0.7,
        cone_height=arrow_length * 0.3
    )
    arrow.paint_uniform_color([1.0, 0.0, 0.0])  # red arrow for normal vector
    
    # rotating arrow to align with normal vector
    # default arrow points along +Z, rotate to normal direction
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal_vector)
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.arccos(np.clip(np.dot(z_axis, normal_vector), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(door_center)
    
    # coordinate frame at door center
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=door_center)
    
    # visualization
    o3d.visualization.draw_geometries(
        [pcd, plane_mesh, arrow, coord_frame],
        window_name="Door Plane & Normal Visualization",
        width=1280, height=720,
        point_show_normal=False
    )

if __name__ == "__main__":
    compute_door_pose_goal()