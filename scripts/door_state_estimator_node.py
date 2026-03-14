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
import message_filters
import threading

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_navigation.srv import EstimateDoorState, EstimateDoorStateResponse
from door_navigation.scripts.door_state_estimator_utils import estimate_single_door_state, estimate_double_door_state
from door_navigation.scripts.door_pose_estimator_utils import get_final_depth
from door_ros_interfaces import DoorDetector
from utils.utils import crop_to_bbox_depth
from utils.config import (
    LABEL_MAP, 
    MODEL_PATH, 
    CONFIDENCE_THRESHOLD, 
    IMG_SIZE,
    RGB_TOPIC,
    DEPTH_TOPIC
)

POSE_PUB_QSIZE = 1
RGB_SUB_QSIZE = 1
DEPTH_SUB_QSIZE = 1

WAIT_BEFORE_ESTIMATE = 2.0 # secs waited after reaching pre-goal and before running state estimation

class DoorStateEstimatorService:
    def __init__(self):
        rospy.init_node("door_state_estimator_service")
        
        # Parameters
        self.rgb_topic = RGB_TOPIC
        self.depth_topic = DEPTH_TOPIC
        self.use_vlm = True
        self.use_da = True
        self.visualize = False
        self.wait_before_estimate = WAIT_BEFORE_ESTIMATE
        
        # CV Bridge
        self.bridge = CvBridge()

        # YOLO detector for on-demand re-detection at pre-door
        self.door_detector = DoorDetector()

        # RGB-D sync (similar to DoorDetectionAndPoseNode)
        self.frame_lock = threading.Lock()
        self.latest_rgb_frame = None
        self.latest_depth_frame = None
        self.latest_stamp = None

        self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image, queue_size=1, buff_size=2**22)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1, buff_size=2**22)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=5,
            slop=0.05
        )
        self.ts.registerCallback(self.rgbd_callback)
        
        # cache for latest door pose (from coordinator or topic)
        # self.latest_door_pose = None
        # rospy.Subscriber("/door/poses", DoorPoseArray, self.door_pose_callback, queue_size=1)
        
        # service definition
        self.service = rospy.Service("/door/estimate_state", EstimateDoorState, self.handle_estimate_door_state)
        
        rospy.loginfo(f"Door State Estimator Service initialized")
        rospy.loginfo(f"  RGB topic: {self.rgb_topic}")
        rospy.loginfo(f"  Depth topic: {self.depth_topic}")
        rospy.loginfo(f"  Use VLM: {self.use_vlm}")
        rospy.loginfo(f"  Visualize: {self.visualize}")
        rospy.loginfo(f"  Wait before estimate: {self.wait_before_estimate} s")
        rospy.loginfo(f"Service available at: /door/estimate_state")

    def rgbd_callback(self, rgb_msg, depth_msg):
        """Cache latest synchronized RGB-D frames"""
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            if cv_depth_image.dtype == np.float32 or cv_depth_image.dtype == np.float64:
                cv_depth_image = (cv_depth_image * 1000).astype(np.uint16)

            with self.frame_lock:
                self.latest_rgb_frame = cv_color_image
                self.latest_depth_frame = cv_depth_image
                self.latest_stamp = rgb_msg.header.stamp
        except Exception as e:
            rospy.logwarn(f"Failed to cache RGB-D frame: {e}")
    
    def handle_estimate_door_state(self, req): # empty request
        """Service callback for door state estimation"""
        response = EstimateDoorStateResponse()
        
        try:
            rospy.loginfo("Door state estimation service called")
            
            if self.wait_before_estimate > 0.0:
                rospy.sleep(self.wait_before_estimate)
            
            with self.frame_lock: # lock the frame during transfer/copy latest frame to local variable for processing
                rgb_img = self.latest_rgb_frame.copy() if self.latest_rgb_frame is not None else None
                depth_img = self.latest_depth_frame.copy() if self.latest_depth_frame is not None else None
            
            # after with block, the lock release (rgb_img & depth_img are local copies, so safe to process without lock)

            if rgb_img is None or depth_img is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.confidence = 0.0
                response.error_message = "No synchronized RGB-D images available"
                response.success = False
                rospy.logwarn(response.error_message)
                return response
            
            # Convert depth to float32 meters if needed
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0  # mm to meters

            result = self.estimate_door_state_from_rgbd(rgb_img, depth_img)
            
            """
            result = {
                'door_state': 'open',
                'human_present': 'no',
                'conversation': 'please open the door'
                }
            """
            
            # Parse result
            if result is None:
                response.door_state = "unknown"
                response.is_passable = False
                response.error_message = "Door state estimation failed"
                rospy.logwarn(response.error_message)
                return response
            
            door_state = result.get("door_state", "unknown")
            response.door_state = door_state
            response.conversation = result.get("conversation", "NA")
            response.is_passable = result.get("is_passable", False)
            response.error_message = ""
            rospy.loginfo(f"Door state: {door_state}, passable: {response.is_passable}")
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
        
    def estimate_door_state_from_rgbd(self, rgb_img, depth_img_rs):
        """
        Run YOLO on current RGB frame, choose best door, then estimate door state.
        Returns a dict with door_state, confidence when successful, or None on failure.
        """
        try:
            detections = self.door_detector.run_yolo_model(
                model_path=MODEL_PATH,
                rgb_image=rgb_img,
                img_size=IMG_SIZE,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                visualize=False
            )

            if len(detections) == 0:
                rospy.logwarn("No doors detected for state estimation")
                return None

            best_det = max(detections, key=lambda d: d.get("conf", 0.0))
            door_bbox = best_det.get("bbox")
            if not door_bbox:
                rospy.logwarn("Detection missing bbox, cannot estimate door state")
                return None

            cls_id = int(best_det.get("cls_id", -1))
            door_type = LABEL_MAP.get(cls_id, "door_single")
            
            # process depth image
            if self.use_da:
                # get RAW depth from DepthAnything model (in meters)
                depth_da = self.door_detector.run_depth_anything_v2_on_image(rgb_image=rgb_img)
                # apply correction to depth_da_raw using pre-computed calibration coefficients
                depth_da_corr = self.door_detector.get_corrected_depth_image(depth_da=depth_da, model="quad")
                # get final depth image (corrected + scaled)
                depth_da_final = get_final_depth(depth_img_rs, depth_da_corr)  # TODO need to check
            else: # while navigation, we can directly use the RS depth as it's more real-time and accurate for non-glass regions, and the robot will be close enough to the door for better depth readings
                depth_da_final = depth_img_rs
                
            # crop ROI for depth, based on actual bbox
            roi_depth = crop_to_bbox_depth(depth_da_final, door_bbox)
            full_depth = depth_da_final

            if door_type == "door_double":
                rospy.loginfo("Estimating double door state (re-detect)")
                door_state_res = estimate_double_door_state(door_bbox, rgb_img, roi_depth, full_depth, 
                                                        visualize=self.visualize, use_vlm=self.use_vlm)
            else:
                rospy.loginfo("Estimating single door state (re-detect)")
                door_state_res = estimate_single_door_state(door_bbox, rgb_img, roi_depth, full_depth, 
                                                        visualize=self.visualize, use_vlm=self.use_vlm)
                
            """
            res = {
                'door_state': 'open',
                'human_present': 'no',
                'conversation': 'please open the door',
                'is_passable': True,
                }
            """
            return door_state_res

        except Exception as e:
            rospy.logerr(f"Door state estimation failed: {e}")
            return None
    
    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        service_node = DoorStateEstimatorService()
        service_node.spin()
    except rospy.ROSInterruptException:
        pass
