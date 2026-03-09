#!/usr/bin/env python3
"""
Door State Estimator Service Node
Provides on-demand door state estimation (open/closed/passable) when called by coordinator.
"""

import os
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_navigation.srv import EstimateDoorState, EstimateDoorStateResponse
from door_navigation.msg import DoorPose
from door_navigation.scripts.door_state_estimator_utils import estimate_single_door_state, estimate_double_door_state
from door_ros_interfaces import RGBDImageReceiver


class DoorStateEstimatorService:
    def __init__(self):
        rospy.init_node("door_state_estimator_service")
        
        # Parameters
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.use_vlm = rospy.get_param("~use_vlm", False)
        self.visualize = rospy.get_param("~visualize", False)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # RGBD receiver for synced images
        self.rgbd_receiver = RGBDImageReceiver(self.rgb_topic, self.depth_topic)
        
        # Cache for latest door pose (from coordinator or topic)
        self.latest_door_pose = None
        rospy.Subscriber("/door/poses", DoorPose, self.door_pose_callback, queue_size=1)
        
        # Service
        self.service = rospy.Service(
            "/door/estimate_state",
            EstimateDoorState,
            self.handle_estimate_door_state
        )
        
        rospy.loginfo(f"Door State Estimator Service initialized")
        rospy.loginfo(f"  RGB topic: {self.rgb_topic}")
        rospy.loginfo(f"  Depth topic: {self.depth_topic}")
        rospy.loginfo(f"  Use VLM: {self.use_vlm}")
        rospy.loginfo(f"  Visualize: {self.visualize}")
        rospy.loginfo(f"Service available at: /door/estimate_state")
    
    def door_pose_callback(self, msg):
        """Cache latest door pose"""
        self.latest_door_pose = msg
    
    def handle_estimate_door_state(self, req):
        """Service callback for door state estimation"""
        response = EstimateDoorStateResponse()
        
        try:
            rospy.loginfo("Door state estimation service called")
            
            # Check if we have a door pose
            if self.latest_door_pose is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.confidence = 0.0
                response.error_message = "No door pose available"
                rospy.logwarn(response.error_message)
                return response
            
            # Get latest RGBD images
            rgb_img, depth_img = self.rgbd_receiver.get_rgbd()
            
            if rgb_img is None or depth_img is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.confidence = 0.0
                response.error_message = "No RGB-D images available"
                rospy.logwarn(response.error_message)
                return response
            
            # Convert depth to float32 meters if needed
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0  # mm to meters
            
            # Reconstruct bbox from door pose (we need to look up the detection)
            # For now, use the center point and estimate a reasonable bbox
            # In production, you might want to pass the bbox explicitly or cache it
            door_bbox = self.estimate_bbox_from_pose(self.latest_door_pose, rgb_img.shape)
            
            if door_bbox is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.confidence = 0.0
                response.error_message = "Could not determine door bbox"
                rospy.logwarn(response.error_message)
                return response
            
            # Determine door type
            door_type = self.latest_door_pose.door_type
            
            # Call appropriate estimation function
            if door_type == "double_door" or door_type == "0":
                rospy.loginfo("Estimating double door state")
                result = estimate_double_door_state(
                    door_bbox,
                    rgb_img,
                    depth_img,
                    depth_img,  # full_depth
                    visualize=self.visualize,
                    use_vlm=self.use_vlm
                )
            else:
                rospy.loginfo("Estimating single door state")
                result = estimate_single_door_state(
                    door_bbox,
                    rgb_img,
                    depth_img,
                    depth_img,  # full_depth
                    visualize=self.visualize,
                    use_vlm=self.use_vlm
                )
            
            # Parse result
            if result is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.confidence = 0.0
                response.error_message = "Door state estimation failed"
                rospy.logwarn(response.error_message)
                return response
            
            # Handle VLM output (dict) vs geometric output (string)
            if isinstance(result, dict):
                # VLM result
                door_state = result.get("door_state", "unknown")
                response.door_state = door_state
                response.is_passable = door_state in ["open", "semi_open"]
                response.confidence = self.latest_door_pose.confidence
                response.error_message = ""
                rospy.loginfo(f"VLM result: {door_state}, passable: {response.is_passable}")
            else:
                # Geometric result (string like "open", "closed", "semi_open")
                response.door_state = str(result)
                response.is_passable = result in ["open", "semi_open"]
                response.confidence = self.latest_door_pose.confidence
                response.error_message = ""
                rospy.loginfo(f"Geometric result: {result}, passable: {response.is_passable}")
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error in door state estimation service: {e}")
            response.door_state = "error"
            response.is_passable = False
            response.confidence = 0.0
            response.error_message = str(e)
            return response
    
    def estimate_bbox_from_pose(self, door_pose_msg, img_shape):
        """
        Estimate a reasonable bbox from door pose.
        This is a fallback - ideally, the coordinator should cache the detection bbox.
        """
        try:
            # Camera intrinsics (should match door_pose_estimator)
            FX = 385.88861083984375
            FY = 385.3906555175781
            CX = 317.80999755859375
            CY = 243.65032958984375
            
            # Door center in camera frame (need to transform from map frame)
            # For simplicity, assume door is in front and project center + width
            # This is a rough estimate - better to cache the original detection bbox
            
            # Use a fixed bbox size based on typical door dimensions
            # This is a workaround - proper implementation should track the bbox
            img_h, img_w = img_shape[:2]
            
            # Estimate bbox as center 1/3 of image with standard door proportions
            w = img_w // 3
            h = int(img_h * 0.6)  # Doors are typically tall
            x1 = (img_w - w) // 2
            y1 = (img_h - h) // 2
            x2 = x1 + w
            y2 = y1 + h
            
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            rospy.logwarn(f"Using estimated bbox (not from detection): {bbox}")
            rospy.logwarn("For better accuracy, pass the original detection bbox")
            
            return bbox
            
        except Exception as e:
            rospy.logerr(f"Error estimating bbox: {e}")
            return None
    
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        service_node = DoorStateEstimatorService()
        service_node.spin()
    except rospy.ROSInterruptException:
        pass
