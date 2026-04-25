#!/home/ias/satya/venv38/bin/python3
"""
Door Coordinator (Refactored)
Subscribes to door poses from door_pose_estimator_node and coordinates door traversal logic.
Runs at 10 Hz without blocking on heavy vision computation.
"""

import os
import sys
import rospy
import tf2_ros
import math
import time
import numpy as np
from enum import Enum

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import custom messages
from door_navigation.msg import DoorPoseArray
from door_navigation.srv import EstimateDoorState
from utils.config import *
from door_pose_estimator_utils import get_post_door_pose, get_pre_door_pose
from voice_assistant import get_voice_assistant


class DoorState(Enum):
    NAVIGATING = 0
    APPROACHING_DOOR = 1
    AT_PRE_DOOR = 2
    TRAVERSING = 3


class DoorCoordinator:
    def __init__(self):
        rospy.init_node("door_coordinator")
        
        self.pre_door_distance = PRE_DOOR_DISTANCE
        self.post_door_distance = POST_DOOR_DISTANCE
        
        self.state = DoorState.NAVIGATING # default state
        self.current_plan = None
        self.latest_door_poses = []  # latest door poses from perception for one detection, poses are calculated as and when the are detected
        self.current_door_pose_map = None  # currently handled door
        self.original_goal = None
        self.use_voice_confirmation = USE_VOICE_CONFIRMATION
        self.voice_confirmation_timeout_sec = VOICE_CONFIRMATION_TIMEOUT_SEC
        self.voice_confirmation_max_tries = VOICE_CONFIRMATION_MAX_TRIES
        self.human_confirmation_cooldown_sec = HUMAN_CONFIRMATION_COOLDOWN_SEC
        self.last_human_confirmation_prompt_ts = 0.0
        
        # TF listerner setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # subscribe to door poses (from door_pose_estimator_node)
        rospy.Subscriber(DOOR_POSE_TOPIC, DoorPoseArray, self.door_pose_callback, queue_size=10)
        
        # subscribe to global plan
        rospy.Subscriber(TEB_GLOBAL_PLAN_TOPIC, Path, self.plan_callback, queue_size=1)
        
        # subscribe to human confirmation
        # rospy.Subscriber("/door/human_confirm", Bool, self.human_confirm_callback, queue_size=1)
        
        # move_base client
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("Connected to move_base")
        
        # Door state estimator service client
        rospy.loginfo("Waiting for door state estimator service...")
        try:
            rospy.wait_for_service("/door/estimate_state", timeout=30)
            self.door_state_service = rospy.ServiceProxy("/door/estimate_state", EstimateDoorState)
            rospy.loginfo("Connected to door state estimator service")
        except rospy.ROSException:
            rospy.logwarn("Door state estimator service not available, will skip state checks")
            self.door_state_service = None

        # Voice assistant is optional. If unavailable, coordinator still runs.
        self.voice_assistant = None
        try:
            self.voice_assistant = get_voice_assistant(enable_listening=True)
            rospy.loginfo("Voice assistant ready for door coordinator announcements")
        except Exception as e:
            rospy.logwarn(f"Voice assistant unavailable, continuing without speech: {e}")
            self.use_voice_confirmation = False
        
        rospy.loginfo("DoorCoordinator initialized")

    def _speak(self, text):
        """Best-effort speech helper that never blocks coordinator on failures."""
        if not text or self.voice_assistant is None:
            return
        try:
            self.voice_assistant.speak(text)
        except Exception as e:
            rospy.logwarn(f"Speech output failed: {e}")
    
    def door_pose_callback(self, msg):
        """Cache latest door poses snapshot"""
        if msg is None or len(msg.doors) == 0:
            return

        snapshot = []
        for door in msg.doors:
            snapshot.append({
                "position": [door.position.x, door.position.y, door.position.z],
                "normal": [door.normal.x, door.normal.y, door.normal.z],
                "width": door.width,
                "door_type": door.door_type,
                "confidence": door.confidence,
                "timestamp": msg.header.stamp
            })

        # Keep only the latest snapshot to represent the world state for this frame
        self.latest_door_poses = snapshot
    
    def plan_callback(self, msg):
        self.current_plan = msg
    
    def interact_with_human(self, conversation):
        # operates in blocking mode
        try:
            # SPEAK
            rospy.loginfo(f"Interacting with human: {conversation}")

            if self.use_voice_confirmation and self.voice_assistant is not None:
                prompt = "Is the door safe to traverse? Please say yes or no."
                self._speak(prompt)
                for attempt in range(self.voice_confirmation_max_tries):
                    feedback = self.voice_assistant.get_voice_input(timeout_sec=self.voice_confirmation_timeout_sec)
                    if not feedback:
                        rospy.loginfo(f"No voice confirmation captured (attempt {attempt + 1}/{self.voice_confirmation_max_tries})")
                        continue

                    fb = feedback.lower()
                    rospy.loginfo(f"Human voice confirmation: {fb}")
                    if any(word in fb for word in ["yes", "sure", "go ahead", "okay", "ok"]):
                        rospy.loginfo(f"Human confirmation received: {conversation}")
                        return True
                    if any(word in fb for word in ["no", "wait", "stop", "not safe"]):
                        return False
            
            else:
                rospy.loginfo("Voice confirmation unavailable, falling back to keyboard input")
                # ASK FOR FEEDBACK
                feedback = input("Is the door safe to traverse? (yes/no): ")
                
                # Check FEEDBACK
                if "yes" in feedback.lower() or "sure" in feedback.lower() or "go ahead" in feedback.lower():
                    rospy.loginfo(f"Human confirmation received: {conversation}")
                    return True
                return False
        except Exception as e:
            rospy.logwarn(f"Invalid human confirm message: {e}")
            return False
    
    def get_robot_pose_in_map(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
            pose = PoseStamped()
            pose.header.frame_id = "map"
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
        """Check if any detected door intersects the planned path (non-blocking)"""
        if self.current_plan is None or len(self.latest_door_poses) == 0:
            return False
        
        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            return False
        
        for door_pose in self.latest_door_poses:
            if self.check_door_intersects_path(door_pose, robot_pose):
                self.current_door_pose_map = door_pose  # can handle only 1 door for traversal at a time
                return True
        
        return False
    
    def check_door_intersects_path(self, door_pose_map, robot_pose_map):
        """Check if door intersects future path"""
        if self.current_plan is None or len(self.current_plan.poses) < 2:
            return False
        
        # Door span calculation
        xd, yd = door_pose_map["position"][:2]
        door_yaw_map = math.atan2(door_pose_map["normal"][1], door_pose_map["normal"][0])
        door_width = door_pose_map.get("width", 0.9)
        
        # Span direction perpendicular to normal
        span_yaw = door_yaw_map + math.pi / 2.0
        half_w = door_width / 2.0
        
        door_p1 = (xd + half_w * math.cos(span_yaw), yd + half_w * math.sin(span_yaw))
        door_p2 = (xd - half_w * math.cos(span_yaw), yd - half_w * math.sin(span_yaw))
        
        # Find closest point on path to robot
        path = self.current_plan.poses
        rx = robot_pose_map.pose.position.x
        ry = robot_pose_map.pose.position.y
        
        closest_i = min(
            range(len(path)),
            key=lambda i: (path[i].pose.position.x - rx)**2 + (path[i].pose.position.y - ry)**2
        )
        
        # future path segments only
        end_i = min(len(path) - 1, closest_i + LOOKAHEAD_POINTS)
        
        for i in range(closest_i, end_i):
            ax = path[i].pose.position.x
            ay = path[i].pose.position.y
            bx = path[i + 1].pose.position.x
            by = path[i + 1].pose.position.y
            
            if self.segments_intersect((ax, ay), (bx, by), door_p1, door_p2):
                rospy.loginfo(f"Door intersects path at segment {i}")
                return True
        
        return False
    
    def orientation(self, a, b, c):
        """Returns orientation of triplet (a,b,c)"""
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2
    
    def segments_intersect(self, p1, p2, q1, q2):
        """Check if segment p1-p2 intersects q1-q2"""
        o1 = self.orientation(p1, p2, q1)
        o2 = self.orientation(p1, p2, q2)
        o3 = self.orientation(q1, q2, p1)
        o4 = self.orientation(q1, q2, p2)
        return o1 != o2 and o3 != o4
    
    def compute_pre_door_goal(self):
        """Compute pre-door goal"""
        if self.current_door_pose_map is None:
            rospy.logwarn("No door pose available")
            return None
        
        door_centre_pose = self.current_door_pose_map["position"]
        door_normal = self.current_door_pose_map["normal"]
        
        pre_x, pre_y, pre_yaw = get_pre_door_pose(np.array(door_centre_pose), 
                                                     np.array(door_normal), 
                                                     offset_distance=self.pre_door_distance)
        
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = pre_x
        goal.pose.position.y = pre_y
        goal.pose.position.z = 0.0 # only 2D is considered for navigation
        
        quat = quaternion_from_euler(0, 0, pre_yaw) # need to face the door, so convert yaw to quaternion
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        
        rospy.loginfo(f"Pre-door goal: x={pre_x:.2f}, y={pre_y:.2f}, yaw={np.degrees(pre_yaw):.1f}°")
        return goal
    
    def compute_post_door_goal(self):
        """Compute post-door goal"""
        if self.current_door_pose_map is None:
            rospy.logwarn("No door pose available")
            return None
        
        door_centre_pose = self.current_door_pose_map["position"]
        door_normal = self.current_door_pose_map["normal"]
        # get post-door pose
        post_x, post_y, post_yaw = get_post_door_pose(np.array(door_centre_pose), 
                                                      np.array(door_normal), 
                                                      offset_distance=self.post_door_distance)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = post_x
        goal.pose.position.y = post_y
        goal.pose.position.z = 0.0
        
        from tf.transformations import quaternion_from_euler
        quat = quaternion_from_euler(0, 0, post_yaw)
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        
        rospy.loginfo(f"Post-door goal: x={post_x:.2f}, y={post_y:.2f}, yaw={np.degrees(post_yaw):.1f}°")
        return goal
    
    def send_goal(self, pose_stamped):
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped
        self.move_base_client.send_goal(goal)
        rospy.loginfo("Sent navigation goal")
    
    def trigger_pre_door(self):
        if self.original_goal is None and self.current_plan: # save original goal to return to after door traversal
            self.original_goal = self.current_plan.poses[-1]
            rospy.loginfo("Saved original navigation goal")
        
        rospy.loginfo("Triggering pre-door pose")
        self._speak("Door detected on path. Moving to pre-door position.")
        pre_goal = self.compute_pre_door_goal()
        if pre_goal:
            self.send_goal(pre_goal)
            self.state = DoorState.APPROACHING_DOOR
    
    def perfrom_door_state_check(self):
        # door state via service is called (at pre-door pose)
        if self.door_state_service is not None:
            try:
                rospy.loginfo("Calling door state estimator service...")
                self._speak("Checking the door state.")
                response = self.door_state_service()
                rospy.loginfo(f"Door state: {response.door_state}, passable: {response.is_passable}, conversation: {response.conversation}")
                self._speak(response.conversation)
                    
                if response.is_passable and response.door_state == "open":
                    rospy.loginfo("Door is passable, proceeding through")
                    self._speak("Door is open and safe. Proceeding through the door.")
                    self.last_human_confirmation_prompt_ts = 0.0
                    self.send_post_door_goal()
                    return
                else:
                    # Human feedback, ask to open the door if not open # TODO
                    now_ts = time.monotonic()
                    if now_ts - self.last_human_confirmation_prompt_ts < self.human_confirmation_cooldown_sec:
                        return
                    self.last_human_confirmation_prompt_ts = now_ts

                    self._speak("Door is not ready to pass. Waiting for human confirmation.")
                    approved = self.interact_with_human(response.conversation) # TODO: can pass conversation snippet from state estimator to use in human interaction
                        
                    # if YES, perform state eastimation again then proceed
                    if approved:
                        # response = self.door_state_service() # Dont scan 2nd time just traverse
                        self.door_handled = True # passable and the door is handled
                        # SPEAK that robot is proceeding through the door
                        rospy.loginfo("Human confirmed door is safe to traverse")
                        self._speak("Human confirmed. Proceeding through the door.")
                        self.last_human_confirmation_prompt_ts = 0.0
                        self.send_post_door_goal()
                        return
                            
            except rospy.ServiceException as e:
                rospy.logwarn(f"Door state service call failed: {e}")
                self._speak("Door state service failed. Please check the system.")
    
    def check_pre_door_reached(self):
        if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached pre-door pose")
            self._speak("Reached pre-door position.")
            
            self.state = DoorState.AT_PRE_DOOR

        # if not succeeded till now, that means the robot has not yet reached the Pre-door pose
        # state == APPROACHING_DOOR is still valid
        
    def send_post_door_goal(self):
        rospy.loginfo("Sending post-door goal")
        self._speak("Navigating through the doorway.")
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
                
            elif self.state == DoorState.AT_PRE_DOOR:
                self.perfrom_door_state_check()
            
            elif self.state == DoorState.TRAVERSING:
                if self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Door traversal complete, resuming original goal")
                    self._speak("Door traversal complete. Resuming original goal.")
                    if self.original_goal:
                        self.send_goal(self.original_goal)
                        self.original_goal = None
                    self.state = DoorState.NAVIGATING
            rate.sleep()

        if self.voice_assistant is not None:
            self.voice_assistant.close()


if __name__ == "__main__":
    try:
        coordinator = DoorCoordinator()
        coordinator.spin()
    except rospy.ROSInterruptException:
        pass
