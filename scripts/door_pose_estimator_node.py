#!/usr/bin/env python3
"""
Door Pose Estimator Node
Subscribes to door detections, computes 3D poses using depth, and publishes poses in map frame.
"""

import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3Stamped
from door_navigation.msg import DoorDetection, DoorPose
from cv_bridge import CvBridge
import numpy as np
import message_filters

# Import pose estimator functions
from door_pose_estimator import compute_door_3d_pose_from_detection
from door_ros_interfaces import DoorDetector


class DoorPoseEstimatorNode:
    def __init__(self):
        rospy.init_node("door_pose_estimator_node")
        
        # Initialize components
        self.door_detector = DoorDetector()  # for depth estimation
        self.bridge = CvBridge()
        
        # TF for camera->map transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Latest images (for depth estimation)
        self.latest_rgb = None
        self.latest_depth = None
        self.image_lock = rospy.Lock()
        
        # Subscribe to images
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw",
            Image,
            self.rgb_callback,
            queue_size=1,
            buff_size=2**24
        )
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24
        )
        
        # Subscribe to detections
        self.detection_sub = rospy.Subscriber(
            "/door/detections",
            DoorDetection,
            self.detection_callback,
            queue_size=10
        )
        
        # Publisher
        self.pose_pub = rospy.Publisher(
            "/door/poses",
            DoorPose,
            queue_size=10
        )
        
        rospy.loginfo("DoorPoseEstimatorNode initialized")
    
    def rgb_callback(self, msg):
        try:
            with self.image_lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"RGB conversion failed: {e}")
    
    def depth_callback(self, msg):
        try:
            with self.image_lock:
                # Depth in mm (16UC1)
                self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logwarn(f"Depth conversion failed: {e}")
    
    def detection_callback(self, msg):
        """Process door detection and compute 3D pose"""
        with self.image_lock:
            rgb_image = self.latest_rgb
            depth_image_mm = self.latest_depth
        
        if rgb_image is None or depth_image_mm is None:
            rospy.logwarn("No images available for pose estimation")
            return
        
        try:
            # Convert depth to meters
            depth_image = depth_image_mm.astype(np.float32) / 1000.0
            
            # Determine door type
            LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}
            door_type = LABEL_MAP.get(msg.class_id, 'door_single')
            
            # Create door box dict
            door_box_dict = {"bbox": msg.bbox}
            
            # Compute 3D pose in camera frame
            door_centre_cam, normal_vector_cam = compute_door_3d_pose_from_detection(
                rgb_image,
                depth_image,
                door_box_dict,
                self.door_detector,
                door_type=door_type,
                visualize_roi=False
            )
            
            if door_centre_cam is None or normal_vector_cam is None:
                rospy.logwarn("Failed to compute 3D door pose")
                return
            
            # Transform to map frame
            door_pose_map = self.transform_camera_to_map(
                door_centre_cam,
                normal_vector_cam,
                msg.header.stamp
            )
            
            if door_pose_map is None:
                return
            
            # Publish pose
            pose_msg = DoorPose()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = "map"
            pose_msg.position.x = door_pose_map["position"][0]
            pose_msg.position.y = door_pose_map["position"][1]
            pose_msg.position.z = door_pose_map["position"][2]
            pose_msg.normal.x = door_pose_map["normal"][0]
            pose_msg.normal.y = door_pose_map["normal"][1]
            pose_msg.normal.z = door_pose_map["normal"][2]
            pose_msg.width = door_pose_map.get("width", 0.9)  # default width
            pose_msg.door_type = door_type
            pose_msg.confidence = msg.confidence
            
            self.pose_pub.publish(pose_msg)
            rospy.logdebug(f"Published door pose: {door_type}, pos=({pose_msg.position.x:.2f}, {pose_msg.position.y:.2f})")
            
        except Exception as e:
            rospy.logerr(f"Pose estimation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def transform_camera_to_map(self, door_centre_cam, door_normal_cam, stamp):
        """Transform door pose from camera frame to map frame"""
        try:
            # Lookup composed transform: camera_link -> map
            tf_cam_to_map = self.tf_buffer.lookup_transform(
                "map", "camera_link", rospy.Time(0), rospy.Duration(0.5)
            )
            
            # Transform point
            door_point_cam = PointStamped()
            door_point_cam.header.frame_id = "camera_link"
            door_point_cam.header.stamp = stamp
            door_point_cam.point.x = door_centre_cam[0]
            door_point_cam.point.y = door_centre_cam[1]
            door_point_cam.point.z = door_centre_cam[2]
            
            door_point_map = tf2_geometry_msgs.do_transform_point(door_point_cam, tf_cam_to_map)
            
            # Transform normal vector
            normal_stamped = Vector3Stamped()
            normal_stamped.header.frame_id = "camera_link"
            normal_stamped.vector.x = door_normal_cam[0]
            normal_stamped.vector.y = door_normal_cam[1]
            normal_stamped.vector.z = door_normal_cam[2]
            
            normal_map = tf2_geometry_msgs.do_transform_vector3(normal_stamped, tf_cam_to_map)
            
            # Normalize
            normal_vec = np.array([normal_map.vector.x, normal_map.vector.y, normal_map.vector.z], dtype=np.float64)
            norm = np.linalg.norm(normal_vec)
            if norm < 1e-6:
                rospy.logwarn("Transformed normal has near-zero length")
                return None
            normal_vec = normal_vec / norm
            
            return {
                "position": [door_point_map.point.x, door_point_map.point.y, door_point_map.point.z],
                "normal": [float(normal_vec[0]), float(normal_vec[1]), float(normal_vec[2])]
            }
            
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None


if __name__ == "__main__":
    try:
        node = DoorPoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
