#!/usr/bin/env python3
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

import sys

import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3Stamped
from door_navigation.msg import DoorDetection, DoorPose, DoorPoseArray
from cv_bridge import CvBridge
import numpy as np
import message_filters

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
from door_pose_estimator_utils import compute_door_3d_pose_from_detection
from door_ros_interfaces import DoorDetector
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
    DEPTH_TOPIC
)

POSE_PUB_QSIZE = 1
RGB_SUB_QSIZE = 1
DEPTH_SUB_QSIZE = 1
MAX_POSE_RADIUS_M = 3.0  # max distance to consider for pose estimation (to filter out far away false positives)

class DoorDetectionAndPoseNode:
    """Combined node for door detection and 3D pose estimation"""
    
    def __init__(self):
        self.visualize = True
        self.use_da = True
        self.max_pose_radius_m = MAX_POSE_RADIUS_M
        rospy.init_node("door_detection_and_pose_node")
        
        # Door detector (YOLO + depth processing)
        self.door_detector = DoorDetector()
        self.bridge = CvBridge()
        
        # TF for camera->map transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.is_processing = False
        
        self.latest_stamp = None
        
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
        self.ts.registerCallback(self.rgbd_callback)
        
        # Publishers
        # self.detection_pub = rospy.Publisher(DOOR_DETECTION_TOPIC, DoorDetection, queue_size=10) # NOT NEEDED now
        self.pose_pub = rospy.Publisher(DOOR_POSE_TOPIC, DoorPoseArray, queue_size=POSE_PUB_QSIZE) # send snapshot of poses per frame
        
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
    
    def rgbd_callback(self, rgb_msg, depth_msg):
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
            self.is_processing = True
            
            # convert ROS Image messages to OpenCV images
            cv_color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1') # might need to change to pass through actual depth format from camera, but for testing we assume 16UC1 in mm

            # always save depth in 'mm' as uint16, as it is standard for depth images
            # if cv_depth_image.dtype == np.float32 or cv_depth_image.dtype == np.float64:
            cv_depth_image = cv_depth_image.astype(np.float32) / 1000.0 # convert to meters (always need to be in meters)
            
            # Verify synchronization
            time_diff = abs((rgb_msg.header.stamp - depth_msg.header.stamp).to_sec())
            if time_diff > 0.05:  # 50ms threshold
                rospy.logwarn(f"RGB-Depth time mismatch: {time_diff*1000:.1f}ms")
            
            self.latest_rgb_frame = cv_color_image
            self.latest_depth_frame = cv_depth_image
            self.latest_stamp = rgb_msg.header.stamp
            
            rospy.logdebug_throttle(5.0, f"Processing synchronized RGBD at {rgb_msg.header.stamp.to_sec():.3f}")
            
            # YOLO detection on RGB image
            detections = self.door_detector.run_yolo_model(
                model_path=MODEL_PATH,
                rgb_image=self.latest_rgb_frame,
                img_size=IMG_SIZE,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                visualize=True
            )
            
            if len(detections) == 0:
                rospy.logdebug("No doors detected in this frame")
                self.last_detection_time = current_time
                return
            
            rospy.loginfo(f"Detected {len(detections)} door(s)")
            
            # Convert depth to meters for pose estimation
            # depth_image_m = self.latest_depth_frame.astype(np.float32) / 1000.0
            
            
            door_pose_msgs = []
            for det in detections:
                # Publish detection (NOT NEEDED now, as we are publishing poses directly)
                # self.publish_detection(det, rgb_msg.header.stamp)
                
                # Compute 3D pose (do not publish per-door)
                if det.get('cls_id') in LABEL_DOORS:
                    pose_msg = self.compute_pose_message(det, self.latest_rgb_frame, cv_depth_image, rgb_msg.header.stamp)
                    if pose_msg is not None:
                        door_pose_msgs.append(pose_msg)
                
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
            msg.header.frame_id = "camera_color_optical_frame"
            msg.bbox = detection['bbox']  # [x1, y1, x2, y2]
            msg.class_id = int(detection['cls_id'])
            msg.confidence = float(detection['conf'])
            
            self.detection_pub.publish(msg)
            
            door_type = LABEL_MAP.get(msg.class_id, 'unknown')
            rospy.logdebug(f"Published detection: {door_type}, conf={msg.confidence:.2f}")
            
        except Exception as e:
            rospy.logwarn(f"Failed to publish detection: {e}")
    
    def compute_pose_message(self, detection, rgb_image, depth_image_m, stamp):
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
            # door centre (x, y, z) in camera frame, normal vector (x, y, z), door width in meters
            door_centre_cam, normal_vector_cam, door_width = compute_door_3d_pose_from_detection(
                rgb_image,
                depth_image_m, # actual rs depth
                door_box_dict,
                self.door_detector,
                door_type=door_type,
                visualize=self.visualize,
                use_da=self.use_da # not used for pose estimationas its relatively slow
            )
            
            if door_centre_cam is None or normal_vector_cam is None:
                rospy.logwarn(f"Failed to compute 3D pose for {door_type}")
                return None

            distance = (door_centre_cam[0] ** 2 + door_centre_cam[1] ** 2 + door_centre_cam[2] ** 2) ** 0.5
            if distance > self.max_pose_radius_m: # if door is far away then not considered
                rospy.logdebug(f"Skipping pose at {distance:.2f}m (> {self.max_pose_radius_m:.2f}m)")
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
            # Lookup transform: camera_link -> map
            tf_cam_to_map = self.tf_buffer.lookup_transform(
                "map", 
                "camera_link", # NOTE: need to create a static transform from camera_link to base_link, and ensure tf tree is correct
                rospy.Time(0),  # Use latest available
                rospy.Duration(1.0)
            )
            
            # Transform position
            door_point_cam = PointStamped()
            door_point_cam.header.frame_id = "camera_link"
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
            normal_stamped.header.frame_id = "camera_link"
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


if __name__ == "__main__":
    import debugpy

    # Inside __init__ or at the start of main
    # debugpy.listen(('localhost', 5678))
    # rospy.loginfo("IDLE: Waiting for VSCode debugger to attach on port 5678...")
    # debugpy.wait_for_client()
    try:
        node = DoorDetectionAndPoseNode()
        rospy.loginfo("Node spinning...")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr(f"Node failed: {e}")
        import traceback
        traceback.print_exc()
