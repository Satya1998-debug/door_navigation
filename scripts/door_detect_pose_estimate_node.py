#!/home/ias/satya/venv38/bin/python3
"""
Combined Door Detection and Pose Estimation Node

This node subscribes to synchronized RGB-D images and performs:
1. Door detection using YOLO
2. 3D pose estimation from detections using depth

Publishes:
- /door/detections: Individual door detections with bounding boxes
- /door/poses: Door 3D poses in map frame

Benefits:
- Perfect RGB-D synchronization (same images for detection and pose)
- Lower latency (no cross-node message passing)
- More efficient (single image processing pipeline)
"""

import os
import os
import sys
import time
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3Stamped, Point
from door_navigation.msg import DoorDetection, DoorPose, DoorPoseArray
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import numpy as np
import message_filters
import cv2
import threading

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
from door_pose_estimator_utils import compute_door_3d_pose_from_detection, compute_da_depth
from door_ros_interfaces import get_door_detector_instance
from utils.config import (
    DOOR_DETECTION_TOPIC,
    DOOR_POSE_TOPIC, 
    LABEL_DOORS, 
    LABEL_MAP,
    MODEL_PATH, 
    CONFIDENCE_THRESHOLD, 
    DETECTION_RATE,
    IMG_SIZE,
    RGB_TOPIC,
    DEPTH_TOPIC,
    CAMERA_INFO_TOPIC,
    CAM_OPTICAL_FRAME,
    MAP_FRAME,
)
from utils.utils import crop_to_bbox_depth

POSE_PUB_QSIZE = 1
RGB_SUB_QSIZE = 1
DEPTH_SUB_QSIZE = 1
MAX_POSE_RADIUS_M = 5  # max distance to consider for pose estimation (to filter out far away false positives)

class DoorDetectionAndPoseNode:
    """Combined node for door detection and 3D pose estimation"""
    
    def __init__(self):
        self.visualize = False
        self.use_da = True
        self.max_pose_radius_m = MAX_POSE_RADIUS_M
        rospy.init_node("door_detection_and_pose_node")

        # get Intrinsics
        self.camera_info_received = False
        self.camera_frame_id = CAM_OPTICAL_FRAME
        self.camera_info_sub = rospy.Subscriber(CAMERA_INFO_TOPIC,CameraInfo, self.camera_info_callback,queue_size=1)
        
        # Door detector (YOLO + depth processing)
        self.door_detector = get_door_detector_instance() # preloaded models are already handled
        self.bridge = CvBridge()
        
        # TF for camera->map transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.is_processing = False
        
        self.latest_stamp = None
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.frame_lock = threading.Lock()
        
        # Detection rate control
        self.last_detection_time = rospy.Time.now()
        
        # standard 640×480 RGB image is about 0.9 MB., so buff_size is set to 4 MB to avoid dropping frames (to hold 1-2 images only)
        self.rgb_sub = message_filters.Subscriber(RGB_TOPIC, Image, queue_size=RGB_SUB_QSIZE, buff_size=2**22)
        self.depth_sub = message_filters.Subscriber(DEPTH_TOPIC, Image, queue_size=DEPTH_SUB_QSIZE, buff_size=2**22)
        
        # sync RGB and Depth with ApproximateTimeSynchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=5,
            slop=0.05  # 50ms tolerance
        )
        self.ts.registerCallback(self.rgbd_callback)  # use the new callback that processes both images together

        # debugging frames
        self.debug = True
        if self.debug:
            rospy.loginfo("DEBUG MODE: Frames will be visualized with detections and normals (if enabled)")
            self.RGB_WINDOW = "Door Live RGB"
            self.DEPTH_WINDOW = "Door Live Depth"    
            cv2.namedWindow(self.RGB_WINDOW, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.DEPTH_WINDOW, cv2.WINDOW_NORMAL)
            rospy.on_shutdown(cv2.destroyAllWindows)
        
        # Publishers
        # self.detection_pub = rospy.Publisher(DOOR_DETECTION_TOPIC, DoorDetection, queue_size=10) # NOT NEEDED now
        self.pose_pub = rospy.Publisher(DOOR_POSE_TOPIC, DoorPoseArray, queue_size=POSE_PUB_QSIZE) # send snapshot of poses per frame
        self.marker_pub = rospy.Publisher("/door_pose_markers", MarkerArray, queue_size=1)
        
        rospy.loginfo("=" * 60)
        rospy.loginfo("Door Detection and Pose Node Initialized")
        rospy.loginfo("=" * 60)
        rospy.loginfo(f"  Detection rate: {DETECTION_RATE} Hz")
        rospy.loginfo(f"  Model: {MODEL_PATH}")
        rospy.loginfo(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
        rospy.loginfo(f"  Publishing detections to: {DOOR_DETECTION_TOPIC}")
        rospy.loginfo(f"  Publishing poses to: {DOOR_POSE_TOPIC}")
        rospy.loginfo("  RGB-D synchronization: ENABLED (slop=50ms)")
        rospy.loginfo("=" * 60)

    def camera_info_callback(self, msg):
        if self.camera_info_received:
            return

        self.intrinsics = {
            'FX': msg.K[0],
            'FY': msg.K[4],
            'CX': msg.K[2],
            'CY': msg.K[5]
        }

        self.camera_info_received = True
        rospy.loginfo("Camera intrinsics received")
        self.camera_info_sub.unregister()

    def _make_point(self, x, y, z):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def publish_pose_markers(self, pose_array):
        if pose_array is None or len(pose_array.doors) == 0:
            return

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for index, door in enumerate(pose_array.doors):
            width = float(door.width) if float(door.width) > 0.0 else 0.9

            center_marker = Marker()
            center_marker.header.frame_id = "map"
            center_marker.header.stamp = pose_array.header.stamp
            center_marker.ns = "detect_center"
            center_marker.id = index * 3
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD
            center_marker.pose.position.x = float(door.position.x)
            center_marker.pose.position.y = float(door.position.y)
            center_marker.pose.position.z = float(door.position.z)
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = 0.18
            center_marker.scale.y = 0.18
            center_marker.scale.z = 0.18
            center_marker.color.r = 0.1
            center_marker.scale.x = 0.40
            center_marker.scale.y = 0.40
            center_marker.scale.z = 0.40
            center_marker.lifetime = rospy.Duration(0)
            center_marker.color.b = 0.1
            center_marker.color.a = 1.0
            center_marker.lifetime = rospy.Duration(0.75)
            marker_array.markers.append(center_marker)

            normal_marker = Marker()
            normal_marker.header.frame_id = "map"
            normal_marker.header.stamp = pose_array.header.stamp
            normal_marker.ns = "detect_normal"
            normal_marker.id = index * 3 + 1
            normal_marker.type = Marker.LINE_STRIP
            normal_marker.action = Marker.ADD
            normal_marker.scale.x = 0.05
            normal_marker.color.r = 1.0
            normal_marker.color.g = 0.2
            normal_marker.color.b = 0.2
            normal_marker.color.a = 1.0
            normal_marker.scale.x = 0.08
            normal_marker.lifetime = rospy.Duration(0)
            normal_marker.points = [
                self._make_point(door.position.x, door.position.y, door.position.z),
                self._make_point(
                    door.position.x + door.normal.x * max(width, 1.0),
                    door.position.y + door.normal.y * max(width, 1.0),
                    door.position.z + door.normal.z * max(width, 1.0),
                ),
            ]
            marker_array.markers.append(normal_marker)

            pre_goal_x = door.position.x + door.normal.x * self.max_pose_radius_m
            pre_goal_y = door.position.y + door.normal.y * self.max_pose_radius_m
            pre_goal_z = door.position.z + door.normal.z * self.max_pose_radius_m

            pre_marker = Marker()
            pre_marker.header.frame_id = "map"
            pre_marker.header.stamp = pose_array.header.stamp
            pre_marker.ns = "detect_pre_goal"
            pre_marker.id = index * 3 + 2
            pre_marker.type = Marker.SPHERE
            pre_marker.action = Marker.ADD
            pre_marker.pose.position.x = float(pre_goal_x)
            pre_marker.pose.position.y = float(pre_goal_y)
            pre_marker.pose.position.z = float(pre_goal_z)
            pre_marker.pose.orientation.w = 1.0
            pre_marker.scale.x = 0.14
            pre_marker.scale.y = 0.14
            pre_marker.scale.z = 0.14
            pre_marker.scale.x = 0.30
            pre_marker.scale.y = 0.30
            pre_marker.scale.z = 0.30
            pre_marker.color.a = 1.0
            pre_marker.lifetime = rospy.Duration(0.75)
            marker_array.markers.append(pre_marker)

        self.marker_pub.publish(marker_array)
    
    def rgbd_callback(self, rgb_msg, depth_msg):
        time_diff = abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec())
        rospy.loginfo_throttle(5.0, f"RGB-Depth sync dt={time_diff:.3f}s")
            
        # convert ROS Image messages to OpenCV images (BGR images)
        cv_color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1') # might need to change to pass through actual depth format from camera, but for testing we assume 16UC1 in mm

        # always save depth in 'mm' as uint16, as it is standard for depth images
        # if cv_depth_image.dtype == np.float32 or cv_depth_image.dtype == np.float64:
        cv_depth_image = cv_depth_image.astype(np.float32) / 1000.0 # convert to meters (always need to be in meters)
            
        # Verify synchronization
        if time_diff > 0.05:  # 50ms threshold
            rospy.logwarn(f"RGB-Depth time mismatch: {time_diff*1000:.1f}ms")

        with self.frame_lock:
            self.latest_rgb_frame = cv_color_image
            self.latest_depth_frame = cv_depth_image
            self.latest_stamp = rgb_msg.header.stamp
            
        rospy.loginfo_throttle(5.0, f"Processing synchronized RGBD at {rgb_msg.header.stamp.to_sec():.3f}")
            
    def rgbd_callback_old(self, rgb_msg, depth_msg):
        """
        Callback for synchronized RGB and Depth images.
        Performs detection and pose estimation on the same image pair.
        """
        current_time = rospy.Time.now()
        time_since_last = (current_time - self.last_detection_time).to_sec()
        if self.is_processing or time_since_last < (1.0 / DETECTION_RATE):
            rospy.loginfo("Already processing or below detection rate, skipping frame")
            return  # skip frame
        
        try:
            t_pose0 = time.time()
            self.is_processing = True
            time_diff = abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec())
            rospy.loginfo_throttle(5.0, f"RGB-Depth sync dt={time_diff:.3f}s")
            
            # convert ROS Image messages to OpenCV images (BGR images)
            cv_color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1') # might need to change to pass through actual depth format from camera, but for testing we assume 16UC1 in mm

            # always save depth in 'mm' as uint16, as it is standard for depth images
            # if cv_depth_image.dtype == np.float32 or cv_depth_image.dtype == np.float64:
            cv_depth_image = cv_depth_image.astype(np.float32) / 1000.0 # convert to meters (always need to be in meters)
            
            # Verify synchronization
            if time_diff > 0.05:  # 50ms threshold
                rospy.logwarn(f"RGB-Depth time mismatch: {time_diff*1000:.1f}ms")
            
            self.latest_rgb_frame = cv_color_image
            self.latest_depth_frame = cv_depth_image
            self.latest_stamp = rgb_msg.header.stamp

            if self.debug:
                with self.frame_lock:
                    self.latest_rgb_frame_debug = cv_color_image.copy()
                    self.latest_depth_frame_debug = cv_depth_image.copy()
            
            rospy.loginfo_throttle(5.0, f"Processing synchronized RGBD at {rgb_msg.header.stamp.to_sec():.3f}")
            
            # YOLO detection on RGB image
            t0 = time.time()
            detections = self.door_detector.run_yolo_model(
                model_path=MODEL_PATH,
                rgb_image=self.latest_rgb_frame,
                img_size=IMG_SIZE,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                visualize=False
            )
            t1 = time.time()
            rospy.loginfo(f"YOLO inference complete (dt={t1 - t0:.3f}s)")
            rospy.loginfo(f"Detected {len(detections)} door(s)")
            
            # Convert depth to meters for pose estimation
            # depth_image_m = self.latest_depth_frame.astype(np.float32) / 1000.0
            
            # compute DA depth once per frame (depth frame is in meters)
            depth_final = compute_da_depth(self.use_da, self.door_detector, 
                                           self.latest_rgb_frame, 
                                           self.latest_depth_frame)

            if self.debug:
                self.visualize_debug(detections, depth_final)

            if len(detections) == 0:
                rospy.loginfo("No doors detected in this frame")
                self.last_detection_time = current_time
                return
            
            door_pose_msgs = []
            for det in detections:
                # Publish detection (NOT NEEDED now, as we are publishing poses directly)
                # self.publish_detection(det, rgb_msg.header.stamp)
                
                # Compute 3D pose (do not publish per-door)
                if det.get('cls_id') in LABEL_DOORS:
                    pose_msg = self.compute_pose_message(det, self.latest_rgb_frame, depth_final, rgb_msg.header.stamp)
                    if pose_msg is not None:
                        door_pose_msgs.append(pose_msg)
                
            t_pose1 = time.time()
            rospy.loginfo(f"Pose estimation complete TOTAL TIME: {t_pose1 - t_pose0:.3f}s)")

            if len(door_pose_msgs) > 0:
                pose_array = DoorPoseArray()
                pose_array.header.stamp = rgb_msg.header.stamp
                pose_array.header.frame_id = "map" # as all poses are in map frame
                pose_array.doors = door_pose_msgs
                self.pose_pub.publish(pose_array)
                rospy.loginfo(f"Published pose array: {len(door_pose_msgs)} doors")
            
            self.last_detection_time = current_time
            
        except Exception as e:
            rospy.logerr(f"RGBD processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_processing = False
    
    def publish_detection(self, detection, stamp):
        """Publish door detection message"""
        try:
            msg = DoorDetection()
            msg.header.stamp = stamp
            msg.header.frame_id = self.camera_frame_id
            msg.bbox = detection['bbox']  # [x1, y1, x2, y2]
            msg.class_id = int(detection['cls_id'])
            msg.confidence = float(detection['conf'])
            
            self.detection_pub.publish(msg)
            
            door_type = LABEL_MAP.get(msg.class_id, 'unknown')
            rospy.loginfo(f"Published detection: {door_type}, conf={msg.confidence:.2f}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish detection: {e}")
    
    def visualize_debug(self, rgb, detections, depth_final):
        
        def normalize_depth_for_display(depth_m):
            if depth_m is None:
                return None

            depth = np.array(depth_m, dtype=np.float32)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            valid = depth > 0
            if not np.any(valid):
                return np.zeros_like(depth, dtype=np.uint8)

            clipped = np.clip(depth, 0.0, 8.0)
            depth_vis = np.zeros_like(clipped, dtype=np.uint8)
            depth_vis[valid] = np.clip((clipped[valid] / 8.0) * 255.0, 0, 255).astype(np.uint8)
            return depth_vis
        
        def annotate_frame(self, rgb, detections, depth_final):
            
            annotated = rgb.copy()

            def mean_valid_depth(roi_depth):
                if roi_depth is None:
                    return None
                roi = np.array(roi_depth, dtype=np.float32)
                valid = np.isfinite(roi) & (roi > 0)
                if not np.any(valid):
                    return None
                return float(np.median(roi[valid]))


            for det in detections:
                cls_id = int(det.get('cls_id', -1))
                if cls_id not in LABEL_DOORS:
                    continue

                x1, y1, x2, y2 = map(int, det['bbox'])
                label = LABEL_MAP.get(cls_id, 'door')
                conf = float(det.get('conf', 0.0))
                roi_depth = crop_to_bbox_depth(depth_final, det['bbox'])
                depth_m = mean_valid_depth(roi_depth)

                text = f"{label} {conf:.2f}"
                if depth_m is not None:
                    text += f" | z={depth_m:.2f}m"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                y_text = y1 - 10 if y1 > 20 else y1 + 20
                cv2.putText(annotated, text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            return annotated
        
        annotated = annotate_frame(self, rgb, detections, depth_final)
        
        depth_vis = normalize_depth_for_display(depth_final)
        if depth_vis is not None:
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow(self.RGB_WINDOW, annotated)
        if depth_vis is not None:
            cv2.imshow(self.DEPTH_WINDOW, depth_vis)
        cv2.waitKey(1)
    
    def compute_pose_message(self, detection, rgb_image, depth_final, stamp):
        """
        Compute 3D pose from detection and return DoorPose message.
        Uses the SAME rgb_image and depth_image that detection was performed on.
        """
        try:
            # Get door type
            cls_id = int(detection['cls_id'])
            door_type = LABEL_MAP.get(cls_id, 'door_single')
            
            # Create door box dict (required format for pose estimator)
            door_box_dict = {"bbox": detection['bbox']}
            
            # Compute 3D pose in camera frame, does all plane fitting, RANSAC, normal calculations, internally
            # door centre (x, y, z) in camera frame (camera_color_optical_frame), normal vector (x, y, z), door width in meters
            t0 = time.time()
            door_centre_cam, normal_vector_cam, door_width = compute_door_3d_pose_from_detection(
                rgb_image,
                depth_final,
                door_box_dict,
                door_type=door_type,
                visualize=False,
                use_da=self.use_da, # not used for pose estimationas its relatively slow
                intrinsics=self.intrinsics
            )
            t1 = time.time()
            rospy.loginfo(f"3D pose computation time: {t1 - t0:.3f}s")
            if door_centre_cam is None or normal_vector_cam is None:
                rospy.logwarn(f"Failed to compute 3D pose for {door_type}")
                return None

            distance = (door_centre_cam[0] ** 2 + door_centre_cam[1] ** 2 + door_centre_cam[2] ** 2) ** 0.5
            if distance > self.max_pose_radius_m: # if door is far away then not considered
                rospy.loginfo(f"Skipping pose at {distance:.2f}m (> {self.max_pose_radius_m:.2f}m)")
                return None
            
            # Transform to map frame
            door_pose_map = self.transform_camera_to_map(door_centre_cam, normal_vector_cam, stamp)
            
            if door_pose_map is None:
                rospy.logwarn("Transform to map frame failed")
                return None
            
            pose_msg = DoorPose()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = "map"
            pose_msg.position.x = door_pose_map["position"][0]
            pose_msg.position.y = door_pose_map["position"][1]
            pose_msg.position.z = door_pose_map["position"][2]
            pose_msg.normal.x = door_pose_map["normal"][0]
            pose_msg.normal.y = door_pose_map["normal"][1]
            pose_msg.normal.z = door_pose_map["normal"][2]
            pose_msg.width = float(door_width)
            pose_msg.door_type = door_type
            pose_msg.confidence = float(detection['conf'])

            return pose_msg

        except Exception as e:
            rospy.logerr(f"Pose computation/publishing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transform_camera_to_map(self, door_centre_cam, door_normal_cam, stamp):
        """Transform door pose from camera frame to map frame"""
        try:
            # Lookup transform: camera optical frame -> map
            tf_cam_to_map = self.tf_buffer.lookup_transform(
                MAP_FRAME, # map
                self.camera_frame_id, # camera_color_optical_frame
                stamp,
                rospy.Duration(1.0)
            )
            
            # Transform position
            door_point_cam = PointStamped()
            door_point_cam.header.frame_id = self.camera_frame_id
            door_point_cam.header.stamp = stamp
            door_point_cam.point.x = door_centre_cam[0]
            door_point_cam.point.y = door_centre_cam[1]
            door_point_cam.point.z = door_centre_cam[2]
            
            door_point_map = tf2_geometry_msgs.do_transform_point(
                door_point_cam, 
                tf_cam_to_map
            )
            
            # Transform normal vector
            normal_stamped = Vector3Stamped()
            normal_stamped.header.frame_id = self.camera_frame_id
            normal_stamped.vector.x = door_normal_cam[0]
            normal_stamped.vector.y = door_normal_cam[1]
            normal_stamped.vector.z = door_normal_cam[2]
            
            normal_map = tf2_geometry_msgs.do_transform_vector3(
                normal_stamped, 
                tf_cam_to_map
            )
            
            # Normalize
            normal_vec = np.array([
                normal_map.vector.x, 
                normal_map.vector.y, 
                normal_map.vector.z
            ], dtype=np.float64)
            
            norm = np.linalg.norm(normal_vec)
            if norm < 1e-6:
                rospy.logwarn("Transformed normal has near-zero length")
                return None
            normal_vec = normal_vec / norm
            
            door_pose_map = {
                "position": [
                    door_point_map.point.x, 
                    door_point_map.point.y, 
                    door_point_map.point.z
                ],
                "normal": [
                    float(normal_vec[0]), 
                    float(normal_vec[1]), 
                    float(normal_vec[2])
                ]
            }
            
            return door_pose_map
            
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None

    def spin(self):
        rate = rospy.Rate(5)  # 5 Hz, keep identical to detection rate to avoid unnecessary processing the same frames multiple times
        while not rospy.is_shutdown():
            try:
                with self.frame_lock:
                    rgb = None if self.latest_rgb_frame is None else self.latest_rgb_frame.copy()
                    depth = None if self.latest_depth_frame is None else self.latest_depth_frame.copy()
                    stamp = self.latest_stamp
                
                rospy.logwarn(f"Now: {rospy.Time.now().to_sec():.3f}")
                rospy.logwarn(f"Image: {stamp.to_sec():.3f}")
                rospy.logwarn(f"Image age: {(rospy.Time.now() - stamp).to_sec():.3f}s")

                if rgb is None or depth is None:
                    rate.sleep()
                    continue
                
                t_pose0 = time.time()
                # YOLO detection on RGB image
                t0 = time.time()
                detections = self.door_detector.run_yolo_model(
                    model_path=MODEL_PATH,
                    rgb_image=rgb,
                    img_size=IMG_SIZE,
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    visualize=False
                )
                t1 = time.time()
                rospy.loginfo(f"YOLO inference complete (dt={t1 - t0:.3f}s)")
                rospy.loginfo(f"Detected {len(detections)} door(s)")
                
                # Convert depth to meters for pose estimation
                # depth_image_m = self.latest_depth_frame.astype(np.float32) / 1000.0
                
                # compute DA depth once per frame (depth frame is in meters)
                depth_final = compute_da_depth(self.use_da, self.door_detector, rgb, depth)

                if self.debug:
                    self.visualize_debug(rgb, detections, depth_final)

                if len(detections) == 0:
                    rospy.loginfo("No doors detected in this frame")
                    continue
                
                door_pose_msgs = []
                for det in detections:
                    # Publish detection (NOT NEEDED now, as we are publishing poses directly)
                    # self.publish_detection(det, rgb_msg.header.stamp)
                    
                    # Compute 3D pose (do not publish per-door)
                    if det.get('cls_id') in LABEL_DOORS:
                        pose_msg = self.compute_pose_message(det, rgb, depth_final, stamp)
                        if pose_msg is not None:
                            door_pose_msgs.append(pose_msg)
                    
                t_pose1 = time.time()
                rospy.loginfo(f"Pose estimation complete TOTAL TIME: {t_pose1 - t_pose0:.3f}s)")

                if len(door_pose_msgs) > 0:
                    pose_array = DoorPoseArray()
                    pose_array.header.stamp = stamp
                    pose_array.header.frame_id = "map" # as all poses are in map frame
                    pose_array.doors = door_pose_msgs
                    self.pose_pub.publish(pose_array)
                    self.publish_pose_markers(pose_array)
                    rospy.loginfo(f"Published pose array: {len(door_pose_msgs)} doors")
            
            except Exception as e:
                rospy.logerr(f"Error in spin loop: {e}")
                import traceback
                traceback.print_exc()
            rate.sleep()

if __name__ == "__main__":
    import debugpy

    # Inside __init__ or at the start of main
    # debugpy.listen(('localhost', 5678))
    # rospy.loginfo("IDLE: Waiting for VSCode debugger to attach on port 5678...")
    # debugpy.wait_for_client()
    try:
        node = DoorDetectionAndPoseNode()
        rospy.loginfo("Node spinning...")
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")
        import traceback
        traceback.print_exc()
