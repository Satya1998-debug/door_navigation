#!/home/satya/MT/uv_ros_py38/bin python3

import os
import sys
import traceback

import rospkg
import rospy
import tf2_ros
import tf2_geometry_msgs
import math
from enum import Enum

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped
from std_msgs.msg import Bool

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import sensor_msgs.msg

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
from door_state_estimator import estimate_single_door_state, estimate_double_door_state, is_door_passable
import cv2
import numpy as np

TEB_GLOBAL_PLAN_TOPIC = "/move_base/TebLocalPlannerROS/global_plan"
# DOOR navigation parameters
POST_DOOR_DISTANCE = 1.5  # meters after door
PRE_DOOR_DISTANCE = 1.2   # meters before door
DOOR_TRIGGER_DISTANCE = 2.0  # start door logic when closer than this
LOOKAHEAD_POINTS = 80  # tune based on plan density



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
        
        self.test_mode = True  # Set test mode to True for debugging, all testing related code
        

        # robot params for door navigation
        self.pre_door_distance = PRE_DOOR_DISTANCE    # before door
        self.post_door_distance = POST_DOOR_DISTANCE   # after door
        self.door_trigger_distance = DOOR_TRIGGER_DISTANCE  # start door logic when closer than this

        # states
        self.state = DoorState.NAVIGATING # initial state is always navigating, waiting for door on path
        self.current_plan = None # latest navigation plan
        # self.current_door_pose_map = None
        self.door_handled = False
        self.original_goal = None  # need to save original destination
        
        # vision-based door detection - run YOLO directly, no external subscription
        self.door_detector = DoorDetector()
        self.rgbd_receiver = RGBDImageReciever()
        
        # detection control
        self.frame_count = 0
        self.detection_interval = 10  # Run YOLO every 5 frames to save computation
        self.cached_detections = []  # Cache detections between intervals
        self.last_detection_time = rospy.Time.now()

        # buffer to receive TF transforms
        # internally subscribes to /tf (published by TEB everytime) /tf_static (published by robot_state_publisher once)
        # map -> odom -> base_link -> (TEB planner)
        # base_link -> camera_link (fixed, from URDF robot state publisher)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) 

        # subscribe to global plan from move_base (TEB publishes to this topic)
        rospy.Subscriber(TEB_GLOBAL_PLAN_TOPIC, Path, self.plan_callback, queue_size=1)
        
        # subscribe to human confirmation for door traversal
        # rospy.Subscriber("/door/human_confirm", Bool, self.human_confirm_callback, queue_size=1)

        # move base client
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base")

        rospy.loginfo("DoorCoordinator initialized")

    def plan_callback(self, msg): # gets current global plan
        # msg format: nav_msgs/Path
        self.current_plan = msg

    def get_robot_pose_in_map(self):
        try:
            # target=map, source=base_link, where is robot base_link in map frame/world frame
            # simply, where is source in target frame
            tf = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
            pose = PoseStamped() # has Pose, which has position: (as Point- x, y, z) and orientation: (as Quaternion- x, y, z, w)
            pose.header.frame_id = "map" # to which frame this pose belongs to (map frame)
            pose.header.stamp = tf.header.stamp
            pose.pose.position.x = tf.transform.translation.x
            pose.pose.position.y = tf.transform.translation.y
            pose.pose.position.z = tf.transform.translation.z
            pose.pose.orientation = tf.transform.rotation
            return pose
        except Exception as e:
            rospy.logwarn("TF lookup failed: %s", str(e))
            return None
        
    def is_door_on_path(self):
        """Run door detection on current frame and check if door is on planned path."""
        if self.current_plan is None:
            return False

        # robot pose in map frame
        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            return False
        
        # get current RGB-D frame
        rgb_image = self.rgbd_receiver.latest_frame_color
        depth_image_rs = self.rgbd_receiver.latest_frame_depth
        
        if rgb_image is None or depth_image_rs is None:
            return False
        
        # YOLO detection at intervals to save computation
        self.frame_count += 1
        if self.frame_count % self.detection_interval == 0:
            rospy.loginfo("Running door detection on current frame")
            # cache detections till next detection run
            self.cached_detections = self.door_detector.run_yolo_model(rgb_image=rgb_image, confidence_threshold=0.5, visualize=False)
            self.last_detection_time = rospy.Time.now()
        
        # use cached detections if available
        if len(self.cached_detections) == 0:
            return False
        
        # get best door detection (highest confidence)
        #best_door = max(self.cached_detections, key=lambda d: d['conf'])
        #rospy.loginfo(f"Best door detection: class={best_door['cls_id']}, conf={best_door['conf']:.2f}")
        
        # there can be multiple detections, we should check each one for whether it intersects path, if any one does, we trigger door logic
        for door_detection in self.cached_detections:

            # get door pose in map
            door_pose_map = self.compute_door_pose_in_map_frame(door_detection, rgb_image, depth_image_rs)
            if door_pose_map is None:
                rospy.logwarn("Failed to compute door pose in map frame")
                continue
            
            # check if door pose intersects path, return on the first door that is in path
            if self.check_door_intersects_path(door_pose_map, robot_pose):
                return True
            
        return False
    
    def compute_door_pose_in_map_frame(self, door, rgb_image, depth_image_rs_mm):

        try:
            rospy.loginfo(f"Door detection for pose estimation: class={door['cls_id']}, conf={door['conf']:.2f}")
            depth_image_rs = depth_image_rs_mm.astype(np.float32) / 1000.0 # convert mm to meters
            # Get door bounding box from YOLO detection dict
            bbox = door['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get door type from detection
            door_cls_id = door.get('cls_id', 1)  # default to single door
            LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}
            door_type = LABEL_MAP.get(door_cls_id, 'door_single')
            rospy.loginfo(f"Door type detected: {door_type}")
                
            # Create door box dict for compute_door_3d_pose_from_detection
            door_box_dict = {"bbox": [x1, y1, x2, y2]}
                
            door_centre_cam, normal_vector_cam, door_width = compute_door_3d_pose_from_detection(
                    rgb_image, 
                    depth_image_rs, 
                    door_box_dict, 
                    self.door_detector,
                    door_type=door_type,
                    visualize_roi=False
                )
                
            if door_centre_cam is None or normal_vector_cam is None:
                rospy.logwarn("Failed to compute door 3D pose from vision")
                return None
                
            rospy.loginfo(f"Door detected in camera frame: center={door_centre_cam}, width={door_width}")
                
            # Transform from camera frame to map frame
            door_pose_map = self.transform_camera_to_map(door_centre_cam, normal_vector_cam)
            # door_pose_map is a dict with "position": [x, y, z], "normal": [nx, ny, nz] in the map frame
            door_pose_map["width"] = door_width  # add width to the map pose dict
            return door_pose_map
            
        except Exception as e:
            rospy.logerr(f"Error computing door pose: {e}")
            traceback.print_exc()
            return None
    
    def transform_camera_to_map(self, door_centre_cam, door_normal_cam):
        """Transform door pose from camera frame to map frame using TF."""
        try:
            # Lookup composed transform directly: camera_link -> map
            tf_cam_to_map = self.tf_buffer.lookup_transform("map", "camera_link", rospy.Time(0), rospy.Duration(0.5))

            # Transform dooe centre point from camera to map frame
            door_point_cam = PointStamped()  # for point tranformation only
            door_point_cam.header.frame_id = "camera_link"
            door_point_cam.header.stamp = rospy.Time.now()
            door_point_cam.point.x = door_centre_cam[0]
            door_point_cam.point.y = door_centre_cam[1]
            door_point_cam.point.z = door_centre_cam[2]

            door_point_map = tf2_geometry_msgs.do_transform_point(door_point_cam, tf_cam_to_map)

            # transform normal vector from camera to map frame
            door_normal_cam = Vector3Stamped()
            door_normal_cam.header.frame_id = "camera_link"
            door_normal_cam.header.stamp = rospy.Time.now()
            door_normal_cam.vector.x = door_normal_cam[0]
            door_normal_cam.vector.y = door_normal_cam[1]
            door_normal_cam.vector.z = door_normal_cam[2]

            door_normal_map = tf2_geometry_msgs.do_transform_vector3(door_normal_cam, tf_cam_to_map)
            
            # Normalize and sanity-check normal vector
            norm_door_normal_map = np.array([door_normal_map.vector.x, door_normal_map.vector.y, door_normal_map.vector.z], dtype=np.float64)
            norm = np.linalg.norm(norm_door_normal_map)
            if norm < 1e-6:
                rospy.logwarn("Transformed normal vector has near-zero length")
                return None
            normalized_normal_vec = norm_door_normal_map / norm

            # Sanity: ensure normal points roughly toward robot; flip if it points away
            # only if testing
            if self.test_mode == True:
                try:
                    robot_pose = self.get_robot_pose_in_map()
                    if robot_pose is not None:
                        door_pos = np.array([door_point_map.point.x, door_point_map.point.y, door_point_map.point.z], dtype=np.float64)
                        robot_pos = np.array([robot_pose.pose.position.x, robot_pose.pose.position.y, robot_pose.pose.position.z], dtype=np.float64)
                        to_robot = robot_pos - door_pos
                        if np.linalg.norm(to_robot) > 1e-6:
                            to_robot = to_robot / np.linalg.norm(to_robot)
                            dot = float(np.dot(normalized_normal_vec, to_robot))
                            if dot < 0:
                                normalized_normal_vec = -normalized_normal_vec
                                rospy.loginfo("Flipped normal to face robot after TF transform")
                except Exception:
                    pass

            return {
                "position": [door_point_map.point.x, door_point_map.point.y, door_point_map.point.z],
                "normal": [float(normalized_normal_vec[0]), float(normalized_normal_vec[1]), float(normalized_normal_vec[2])]
            }
            
        except Exception as e:
            rospy.logwarn(f"TF transform failed: {e}")
            return None
    

    def orientation(self, a, b, c):
        """Returns orientation of triplet (a,b,c):
        0 -> colinear
        1 -> clockwise
        2 -> counterclockwise
        """
        val = (b[1] - a[1]) * (c[0] - b[0]) - \
            (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2


    def segments_intersect(self, p1, p2, q1, q2):
        """Check if segment p1-p2 intersects q1-q2"""
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        return False


    def check_door_intersects_path(self, door_pose_map, robot_pose_map):

        if self.current_plan is None or len(self.current_plan.poses) < 2:
            return False

        # DOOR SPAN CALCULATION
        xd, yd = door_pose_map["position"][:2] # door centre in map frame
        door_yaw_map = math.atan2(door_pose_map["normal"][1], door_pose_map["normal"][0])
        door_width = door_pose_map.get("width")  # default width if not available

        # door span direction = perpendicular to door normal
        # door_yaw is door normal direction, span is yaw + 90°
        span_yaw = door_yaw_map + math.pi / 2.0

        half_w = door_width / 2.0

        door_p1 = (xd + half_w * math.cos(span_yaw), yd + half_w * math.sin(span_yaw))
        door_p2 = (xd - half_w * math.cos(span_yaw), yd - half_w * math.sin(span_yaw))

        # ROBOT LOCATION in the PLANNED PATH - find closest point on path to robot
        path = self.current_plan.poses # list of PoseStamped, in map frame
        rx = robot_pose_map.pose.position.x
        ry = robot_pose_map.pose.position.y

        closest_i = min(
            range(len(path)),
            key=lambda i: (path[i].pose.position.x - rx) ** 2 +
                        (path[i].pose.position.y - ry) ** 2
        )

        # FUTURE PATH SEGMENTS only
        end_i = min(len(path) - 1, closest_i + LOOKAHEAD_POINTS)

        for i in range(closest_i, end_i):
            ax = path[i].pose.position.x
            ay = path[i].pose.position.y
            bx = path[i + 1].pose.position.x
            by = path[i + 1].pose.position.y

            if self.segments_intersect((ax, ay), (bx, by), door_p1, door_p2):
                return True

        return False

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
        """Compute post-door goal to continue navigation after traversing door."""
        if self.current_door_pose_map is None:
            rospy.logwarn("No door pose available for post-door goal")
            # Fallback: use end of current plan
            if self.current_plan and len(self.current_plan.poses) > 0:
                return self.current_plan.poses[-1]
            return None
        
        # Use vision-based door pose to compute post-door position
        door_pos = self.current_door_pose_map["position"]
        door_normal = self.current_door_pose_map["normal"]
        
        # Compute post-door pose (1.5m past door along normal, opposite direction)
        door_centre = np.array(door_pos)
        normal_vector = np.array(door_normal)
        
        # Go through the door (negative normal direction)
        post_x = door_centre[0] - normal_vector[0] * self.post_door_distance
        post_y = door_centre[1] - normal_vector[1] * self.post_door_distance
        post_z = door_centre[2] - normal_vector[2] * self.post_door_distance
        
        # Calculate yaw pointing forward through door
        post_yaw = np.arctan2(-normal_vector[1], -normal_vector[0])
        
        # Create PoseStamped goal
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = post_x
        goal.pose.position.y = post_y
        goal.pose.position.z = 0.0  # Keep on ground plane
        
        # Convert yaw to quaternion
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, post_yaw)
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        
        rospy.loginfo(f"Vision-based post-door goal: x={post_x:.2f}, y={post_y:.2f}, yaw={np.degrees(post_yaw):.1f}°")
        
        return goal

    def send_goal(self, pose_stamped):
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        self.move_base_client.send_goal(goal)
        rospy.loginfo("Sent navigation goal")

    def trigger_pre_door(self):
        # Save the original goal before interrupting
        if self.original_goal is None and self.current_plan:
            self.original_goal = self.current_plan.poses[-1]  # end of planned path
            rospy.loginfo("Saved original navigation goal")
        
        rospy.loginfo("Triggering pre-door pose")
        pre_goal = self.compute_pre_door_goal()
        if pre_goal:
            self.send_goal(pre_goal)
            self.state = DoorState.APPROACHING_DOOR
    
    def estimate_door_state_at_pre_door(self):
        """Run door state estimation at pre-door position using current RGB-D images."""
        try:
            # Get current RGB-D frame
            rgb_image = self.rgbd_receiver.latest_frame_color
            depth_image_mm = self.rgbd_receiver.latest_frame_depth
            
            if rgb_image is None or depth_image_mm is None:
                rospy.logwarn("No camera images available for door state estimation")
                return None
            
            # Convert depth from mm to meters for door_state_estimator
            depth_image = depth_image_mm.astype(np.float32) / 1000.0
            
            # Get door detection from cached detections
            if len(self.cached_detections) == 0:
                rospy.logwarn("No cached door detection for state estimation")
                return None
            
            # Get best door detection
            best_door = max(self.cached_detections, key=lambda d: d['conf'])
            door_cls_id = best_door['cls_id']
            door_bbox = best_door['bbox']
            
            # Determine door type: 0=door_double, 1=door_single
            LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}
            door_type = LABEL_MAP.get(door_cls_id, 'unknown')
            
            rospy.loginfo(f"Estimating state for {door_type} at pre-door position...")
            
            # Crop ROI depth for door
            from utils.utils import crop_to_bbox_depth
            door_box_dict = {"bbox": door_bbox}
            roi_depth = crop_to_bbox_depth(depth_image, door_box_dict)
            full_depth = depth_image
            
            # Call appropriate estimator based on door type
            door_state = None
            if door_type == 'door_single':
                door_state = estimate_single_door_state(
                    door_bbox, rgb_image, roi_depth, full_depth, 
                    visualize=False, use_vlm=False
                )
            elif door_type == 'door_double':
                door_state = estimate_double_door_state(
                    door_bbox, rgb_image, roi_depth, full_depth,
                    visualize=False, use_vlm=False
                )
            else:
                rospy.logwarn(f"Unknown door type: {door_type}")
                return None
            
            if door_state:
                rospy.loginfo(f"Door state estimation result: {door_state}")
            else:
                rospy.logwarn("Door state estimation failed")
            
            return door_state
            
        except Exception as e:
            rospy.logerr(f"Error in door state estimation: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def check_pre_door_reached(self):
        """Check if robot reached pre-door position."""
        if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached pre-door pose")
            self.state = DoorState.AT_PRE_DOOR
            
            # Run door initial state estimation
            rospy.loginfo("Running door initial state estimation...")
            door_state = self.estimate_door_state_at_pre_door()
            
            # In check_pre_door_reached(), you can add logic like:
            if door_state and door_state.get('is_passable') and door_state.get('door_state') == 'open':
                rospy.loginfo("Door is open and passable, proceeding automatically")
                self.send_post_door_goal()  # Skip human confirmation!
            else:
                rospy.loginfo("Door requires attention, waiting for human")
                self.state = DoorState.WAIT_HUMAN
                        
            # Transition to waiting for human confirmation
            self.state = DoorState.WAIT_HUMAN
            rospy.loginfo("Waiting for human confirmation on /door/human_confirm...")

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
            
            elif self.state == DoorState.WAIT_HUMAN:
                # Waiting for human confirmation via /door/human_confirm topic
                # When confirmed, human_confirm_callback sets door_handled=True
                # TODO: need to implement HITL
                if self.door_handled:
                    rospy.loginfo("Human confirmed door is safe to traverse")
                    self.send_post_door_goal()
                    self.door_handled = False  # Reset for next door

            elif self.state == DoorState.TRAVERSING:
                if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Door traversal complete, resuming original goal")
                    if self.original_goal:
                        self.send_goal(self.original_goal)  # Resume original navigation
                        self.original_goal = None  # Clear saved goal
                    self.state = DoorState.NAVIGATING
            rate.sleep()

if __name__ == "__main__":
    coordinator = DoorCoordinator()
    coordinator.spin()









