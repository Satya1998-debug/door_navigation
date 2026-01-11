#!/home/satya/MT/uv_ros_py38/bin python3

import os
import sys

import rospkg
import rospy
import tf2_ros
import tf2_geometry_msgs
import math
from enum import Enum

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import sensor_msgs.msg
import geometry_msgs.msg

# ------ path setup -----
# Get package path using rospkg (works with rosrun)
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils.config import *
from door_navigation.scripts.door_pose_estimator import get_pre_door_pose, compute_door_3d_pose_from_detection
from door_ros_interfaces import DoorDetector, RGBDImageReciever
import cv2
import numpy as np

TEB_GLOBAL_PLAN_TOPIC = "/move_base/TebLocalPlannerROS/global_plan"

class DoorState(Enum):
    # states for door navigation
    NAVIGATING = 0
    APPROACHING_DOOR = 1
    AT_PRE_DOOR = 2
    WAIT_HUMAN = 3
    TRAVERSING = 4
    AT_POST_DOOR = 5


class DoorCoordinator:
    def __init__(self):
        rospy.init_node("door_coordinator")

        # robot params for door navigation
        self.pre_door_distance = PRE_DOOR_DISTANCE    # before door
        self.post_door_distance = POST_DOOR_DISTANCE   # after door
        self.door_trigger_distance = DOOR_TRIGGER_DISTANCE  # start door logic when closer than this

        # states
        self.state = DoorState.NAVIGATING
        self.current_plan = None # latest navigation plan
        self.current_door_pose_map = None
        self.door_handled = False
        
        # vision-based door detection - run YOLO directly, no external subscription
        self.door_detector = DoorDetector()
        self.rgbd_receiver = RGBDImageReciever()
        
        # Detection control
        self.frame_count = 0
        self.detection_interval = 5  # Run YOLO every 5 frames to save computation
        self.cached_detections = []  # Cache detections between intervals
        self.last_detection_time = rospy.Time.now()

        # buffer to receive TF transforms
        # internally subscribes to /tf (published by LIO-SLAM everytime) /tf_static (published by robot_state_publisher once)
        # map -> odom -> base_link -> (LIO-SLAM)
        # base_link -> camera_link (fixed, from URDF robot state publisher)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 

        # subscribe to global plan from move_base (TEB publishes to this topic)
        rospy.Subscriber(TEB_GLOBAL_PLAN_TOPIC, Path, self.plan_callback, queue_size=1)

        # move base client
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base")

        rospy.loginfo("DoorCoordinator initialized")

    def plan_callback(self, msg):
        # msg format: nav_msgs/Path
        self.current_plan = msg

    def get_robot_pose_in_map(self):
        try:
            # target=map, source=base_link, where is robot base_link in map frame/world frame
            # simply, where is source in target frame
            tf = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0),rospy.Duration(0.2))
            pose = PoseStamped() # has position (x, y, z) and orientation (quaternion x, y, z, w)
            pose.header.frame_id = "map"
            pose.pose.position.x = tf.transform.translation.x
            pose.pose.position.y = tf.transform.translation.y
            pose.pose.orientation = tf.transform.rotation
            return pose
        except Exception as e:
            rospy.logwarn("TF lookup failed: %s", str(e))
            return None
        
    def is_door_on_path(self):
        """Run door detection on current frame and check if door is on planned path."""
        if self.current_plan is None:
            return False

        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            return False
        
        # Get current RGB-D frame
        rgb_image = self.rgbd_receiver.latest_frame_color
        depth_image = self.rgbd_receiver.latest_frame_depth
        
        if rgb_image is None or depth_image is None:
            return False
        
        # Run YOLO detection at intervals to save computation
        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            rospy.loginfo("Running door detection on current frame")
            self.cached_detections = self.door_detector.run_yolo_model(
                rgb_image=rgb_image,
                confidence_threshold=0.5,
                visualize=False
            )
            self.last_detection_time = rospy.Time.now()
        
        # Use cached detections if available
        if len(self.cached_detections) == 0:
            return False
        
        # Get best door detection (highest confidence)
        best_door = max(self.cached_detections, key=lambda d: d['conf'])
        
        rospy.loginfo(f"Best door detection: class={best_door['cls_id']}, conf={best_door['conf']:.2f}")
        
        # Compute door 3D pose in map frame using SAME frame
        door_pose_map = self.compute_door_pose_in_map_frame(best_door, rgb_image, depth_image)
        if door_pose_map is None:
            rospy.logwarn("Failed to compute door pose in map frame")
            return False
        
        self.current_door_pose_map = door_pose_map
        
        # Check if door is on the planned path
        return self.check_door_intersects_path(door_pose_map)
    
    def compute_door_pose_in_map_frame(self, door_detection_dict, rgb_image, depth_image_mm):
        """Compute 3D door pose in map frame using vision + depth.
        
        Args:
            door_detection_dict: YOLO detection dict with 'bbox', 'conf', 'cls_id'
            rgb_image: RGB image (same frame as detection)
            depth_image_mm: Depth image in mm (same frame as detection)
        """
        try:
            # Convert depth from mm to meters
            depth_image = depth_image_mm.astype(np.float32) / 1000.0
            # Get door bounding box from YOLO detection dict
            bbox = door_detection_dict['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Create door box dict for compute_door_3d_pose_from_detection
            door_box_dict = {"bbox": [x1, y1, x2, y2]}
            
            door_centre_camera, normal_vector, inliers = compute_door_3d_pose_from_detection(
                rgb_image, 
                depth_image, 
                door_box_dict, 
                self.door_detector,
                visualize_roi=False
            )
            
            if door_centre_camera is None or normal_vector is None:
                rospy.logwarn("Failed to compute door 3D pose from vision")
                return None
            
            rospy.loginfo(f"Door detected in camera frame: center={door_centre_camera}, normal={normal_vector}")
            
            # Transform from camera frame to map frame
            door_pose_map = self.transform_camera_to_map(door_centre_camera, normal_vector)
            
            return door_pose_map
            
        except Exception as e:
            rospy.logerr(f"Error computing door pose: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transform_camera_to_map(self, point_camera, normal_camera):
        """Transform door pose from camera frame to map frame using TF."""
        try:
            # Get transform from camera to base_link
            tf_camera_to_base = self.tf_buffer.lookup_transform(
                "base_link", "camera_link", rospy.Time(0), rospy.Duration(0.5))
            
            # Transform point
            point_stamped = geometry_msgs.msg.PointStamped()
            point_stamped.header.frame_id = "camera_link"
            point_stamped.point.x = point_camera[0]
            point_stamped.point.y = point_camera[1]
            point_stamped.point.z = point_camera[2]
            
            point_base = tf2_geometry_msgs.do_transform_point(point_stamped, tf_camera_to_base)
            
            # Get transform from base_link to map
            tf_base_to_map = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.5))
            
            point_map = tf2_geometry_msgs.do_transform_point(
                geometry_msgs.msg.PointStamped(
                    header=geometry_msgs.msg.Header(frame_id="base_link"),
                    point=point_base.point),
                tf_base_to_map)
            
            # Transform normal vector (rotation only)
            # TODO: Properly transform normal vector using rotation matrix
            
            return {
                "position": [point_map.point.x, point_map.point.y, point_map.point.z],
                "normal": normal_camera  # Simplified - should be transformed
            }
            
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    
    def check_door_intersects_path(self, door_pose_map):
        """Check if door position is close to any point on the planned path."""
        if self.current_plan is None or len(self.current_plan.poses) == 0:
            return False
        
        door_x = door_pose_map["position"][0]
        door_y = door_pose_map["position"][1]
        
        # Check distance from door to each point on path
        min_distance = float('inf')
        for pose in self.current_plan.poses:
            path_x = pose.pose.position.x
            path_y = pose.pose.position.y
            
            dist = math.sqrt((door_x - path_x)**2 + (door_y - path_y)**2)
            min_distance = min(min_distance, dist)
        
        # Door is considered "on path" if within threshold
        DOOR_PATH_THRESHOLD = 1.5  # meters
        is_on_path = min_distance < DOOR_PATH_THRESHOLD
        
        if is_on_path:
            rospy.loginfo(f"Door detected on path (distance: {min_distance:.2f}m)")
        
        return is_on_path

    def compute_pre_door_goal(self):
        """Compute pre-door goal using vision-based door pose estimation."""
        if self.current_door_pose_map is None:
            rospy.logwarn("No door pose available, using path-based fallback")
            # Fallback to path-based computation
            path = self.current_plan.poses
            if len(path) < 2:
                return None
            idx = max(0, len(path) - 5)
            return path[idx]
        
        # Use vision-based door pose to compute pre-door position
        door_pos = self.current_door_pose_map["position"]
        door_normal = self.current_door_pose_map["normal"]
        
        # Compute pre-door pose (1m in front along normal)
        door_centre = np.array(door_pos)
        normal_vector = np.array(door_normal)
        
        pre_x, pre_y, pre_z, pre_yaw = get_pre_door_pose(
            door_centre, normal_vector, offset_distance=self.pre_door_distance)
        
        # Create PoseStamped goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = pre_x
        goal.pose.position.y = pre_y
        goal.pose.position.z = 0.0  # Keep on ground plane
        
        # Convert yaw to quaternion
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, pre_yaw)
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        
        rospy.loginfo(f"Vision-based pre-door goal: x={pre_x:.2f}, y={pre_y:.2f}, yaw={np.degrees(pre_yaw):.1f}°")
        
        return goal

    def compute_post_door_goal(self):
        pass

    def send_goal(self, pose_stamped):
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        self.move_base_client.send_goal(goal)
        rospy.loginfo("Sent navigation goal")

    def trigger_pre_door(self):
        rospy.loginfo("Triggering pre-door pose")
        pre_goal = self.compute_pre_door_goal()
        if pre_goal:
            self.send_goal(pre_goal)
            self.state = DoorState.APPROACHING_DOOR
            
    def check_pre_door_reached(self):
        if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached pre-door pose")
            self.state = DoorState.AT_PRE_DOOR
            rospy.loginfo("Trigger DoorStateEstimation (run once here)")
            self.state = DoorState.WAIT_HUMAN

    def send_post_door_goal(self):
        rospy.loginfo("Sending post-door goal")
        post_goal = self.compute_post_door_goal()
        if post_goal:
            self.send_goal(post_goal)
            self.state = DoorState.TRAVERSING

    def spin(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.state == DoorState.NAVIGATING:
                if self.is_door_on_path():
                    rospy.loginfo("Door detected ahead on path")
                    self.trigger_pre_door()

            elif self.state == DoorState.APPROACHING_DOOR:
                self.check_pre_door_reached()

            elif self.state == DoorState.TRAVERSING:
                if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Door traversal complete")
                    self.state = DoorState.NAVIGATING
            rate.sleep()

if __name__ == "__main__":
    coordinator = DoorCoordinator()
    coordinator.spin()









