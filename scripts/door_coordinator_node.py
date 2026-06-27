#!/home/ias/satya/venv38/bin/python3
"""
Door Coordinator (Refactored)
Subscribes to door poses from door_pose_estimator_node and coordinates door traversal logic.
Runs at 10 Hz without blocking on heavy vision computation.
"""

import os
import sys
import logging
import rospy
import tf2_ros
import math
import time
import numpy as np
from enum import Enum

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from tf.transformations import quaternion_from_euler

import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

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
from utils.utils import segments_intersect
from door_pose_estimator_utils import get_post_door_pose, get_pre_door_pose
from voice_assistant import get_voice_assistant


class DoorState(Enum):
    NAVIGATING = 0
    APPROACHING_DOOR = 1
    AT_PRE_DOOR = 2
    TRAVERSING = 3
    FAILED = 4  # terminal: robot stopped, manual recovery required


# minimum seconds between two consecutive door-state service calls.
# prevents calling VLM at 10 Hz while waiting in AT_PRE_DOOR.
STATE_SERVICE_COOLDOWN_SEC = 10.0 # seconds

# Drop perception snapshots older than this when checking door intersection.
# Detection runs at ~5 Hz, so 2 s gives ample margin while still expiring ghosts.
DOOR_POSE_MAX_AGE_SEC = 2.0  # seconds


class DoorCoordinator:
    def __init__(self):
        rospy.init_node("door_coordinator")

        # Set up dedicated debug log file BEFORE anything else so every
        # subsequent rospy.loginfo/logwarn/logerr from this node lands in it.
        self._debug_log_path = None
        self._setup_debug_log_file()

        self.pre_door_distance = PRE_DOOR_DISTANCE
        self.post_door_distance = POST_DOOR_DISTANCE
        
        self.state = DoorState.NAVIGATING # default 0: NAVIGATING
        self.current_plan = None
        self.latest_door_poses = []  # latest door poses from perception for one detection
        self.current_door_pose_map = None  # currently door pose
        self.original_goal = None
        self.use_voice_confirmation = USE_VOICE_CONFIRMATION
        self.voice_confirmation_timeout_sec = VOICE_CONFIRMATION_TIMEOUT_SEC
        self.voice_confirmation_max_tries = VOICE_CONFIRMATION_MAX_TRIES
        self.human_confirmation_cooldown_sec = HUMAN_CONFIRMATION_COOLDOWN_SEC
        self.last_human_confirmation_prompt_ts = 0.0
        self.last_state_service_call_ts = 0.0  # cooldown for VLM service in AT_PRE_DOOR

        # visualization of door poses
        self.marker_topic = "/door_pose_markers"
        
        # TF listerner setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1)

        # for the goal manager, latch is used to store the value in publisher's memory so that any new node gives the same value on subscription
        self.handling_door_pub = rospy.Publisher("/door_coordinator/handling_door", Bool, queue_size=1, latch=True)
        try:
            self.handling_door_pub.publish(Bool(data=False)) # initial its false
        except Exception:
            rospy.logwarn("Failed to publish handling_door: %s", e)

        self._last_handling_published = False # initial its false, as no door yet handled

        # Failure channel: latched String. Empty when healthy; populated with a
        # reason once the coordinator transitions to DoorState.FAILED. External
        # monitors can `rostopic echo` this to know why the robot stopped.
        self.failure_pub = rospy.Publisher("/door_coordinator/failure_reason", String, queue_size=1, latch=True)
        try:
            self.failure_pub.publish(String(data=""))
        except Exception as e:
            rospy.logwarn("Failed to publish initial failure_reason: %s", e)

        # GOAL MANAGER
        # latched current target from the goal manager
        # Used as the "real" goal to resume after door traversal, instead of inferring from the local plan's tail plan[-1]
        self.external_target_goal = None
        rospy.Subscriber("/goal_manager/current_target", PoseStamped, self._external_target_callback, queue_size=1)

        # subscribe to door poses (from door_pose_estimator_node)
        rospy.Subscriber(DOOR_POSE_TOPIC, DoorPoseArray, self.door_pose_callback, queue_size=10)
        
        # subscribe to global plan
        rospy.Subscriber(TEB_GLOBAL_PLAN_TOPIC, Path, self.plan_callback, queue_size=1)
                
        # MOVE BASE client
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base (up to 15s)...")
        if self.move_base_client.wait_for_server(rospy.Duration(15.0)):
            rospy.loginfo("Connected to move_base")
        else:
            rospy.logwarn("move_base action server not available after 15s; coordinator will keep running, goals may be rejected until it comes up")
        
        # TERMINAL FAILURE STATES FOR THE MOVE BASE ACTION
        _MB_FAILURE_STATES = (
            actionlib.GoalStatus.PREEMPTED,
            actionlib.GoalStatus.ABORTED,
            actionlib.GoalStatus.REJECTED,
            actionlib.GoalStatus.LOST,
        )

        # DOOR STATE ESTIMATOR SERVICE client
        rospy.loginfo("Waiting for door state estimator service...")
        try:
            rospy.wait_for_service("/door/estimate_state", timeout=30)
            self.door_state_service = rospy.ServiceProxy("/door/estimate_state", EstimateDoorState)
            rospy.loginfo("Connected to door state estimator service")
        except rospy.ROSException:
            rospy.logwarn("Door state estimator service not available, will skip state checks")
            self.door_state_service = None

        # VOICE ASSISTANT (optional)
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

    def _setup_debug_log_file(self):
        """Mirror this node's rospy log calls to a dedicated debug file.

        File path: from ROS param ``~debug_log_file`` if set, else
        ``~/.ros/log/door_coordinator_debug.log``.

        Captures everything emitted by ``rospy.loginfo/logwarn/logerr`` from
        the coordinator (state transitions, TF failures, voice fallbacks,
        FAILED reasons, etc.). Append mode keeps history across runs; each
        session is preceded by a banner so they're easy to tell apart.
        """
        default_path = os.path.expanduser('~/.ros/log/door_coordinator_debug.log')
        log_path = rospy.get_param('~debug_log_file', default_path)

        try:
            parent = os.path.dirname(log_path) or '.'
            os.makedirs(parent, exist_ok=True)

            fh = logging.FileHandler(log_path, mode='a')
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(
                fmt='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            ))

            # rospy.loginfo/etc. route through the 'rosout' logger.
            rospy_logger = logging.getLogger('rosout')
            rospy_logger.addHandler(fh)
            if rospy_logger.level == logging.NOTSET or rospy_logger.level > logging.INFO:
                rospy_logger.setLevel(logging.INFO)

            self._debug_log_path = log_path

            sep = '=' * 70
            rospy.loginfo(sep)
            rospy.loginfo(f"Door coordinator session start | debug log: {log_path}")
            rospy.loginfo(sep)
        except Exception as e:
            rospy.logwarn(f"Failed to set up debug log file at {log_path}: {e}")
            self._debug_log_path = None
    
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
        # self.publish_door_pose_markers(snapshot, msg.header.stamp)

    def _make_point(self, x, y, z):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)
        return point

    def publish_door_pose_markers(self, door_poses, stamp):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for index, door_pose in enumerate(door_poses):
            position = door_pose["position"]
            normal = door_pose["normal"]
            door_width = float(door_pose.get("width", 0.9) or 0.9)

            center_marker = Marker()
            center_marker.header.frame_id = "map"
            center_marker.header.stamp = stamp
            center_marker.ns = "door_center"
            center_marker.id = index * 3
            center_marker.type = Marker.SPHERE
            center_marker.action = Marker.ADD
            center_marker.pose.position.x = float(position[0])
            center_marker.pose.position.y = float(position[1])
            center_marker.pose.position.z = float(position[2])
            center_marker.pose.orientation.w = 1.0
            center_marker.scale.x = 0.18
            center_marker.scale.x = 0.40
            center_marker.scale.y = 0.40
            center_marker.scale.z = 0.40
            center_marker.scale.z = 0.18
            center_marker.color.r = 0.1
            center_marker.color.g = 0.9
            center_marker.color.b = 0.1
            center_marker.color.a = 1.0
            center_marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(center_marker)

            normal_marker = Marker()
            normal_marker.header.frame_id = "map"
            normal_marker.header.stamp = stamp
            normal_marker.ns = "door_normal"
            normal_marker.id = index * 3 + 1
            normal_marker.type = Marker.LINE_STRIP
            normal_marker.action = Marker.ADD
            normal_marker.scale.x = 0.08
            normal_marker.color.r = 1.0
            normal_marker.color.g = 0.2
            normal_marker.color.b = 0.2
            normal_marker.color.a = 1.0
            normal_marker.lifetime = rospy.Duration(0)
            normal_marker.points = [
                self._make_point(position[0], position[1], position[2]),
                self._make_point(
                    position[0] + normal[0] * max(door_width, 1.0),
                    position[1] + normal[1] * max(door_width, 0.5),
                    position[2] + normal[2] * max(door_width, 0.5),
                ),
            ]
            marker_array.markers.append(normal_marker)

            pre_goal_x = position[0] + normal[0] * self.pre_door_distance
            pre_goal_y = position[1] + normal[1] * self.pre_door_distance
            pre_goal_z = position[2] + normal[2] * self.pre_door_distance

            pre_marker = Marker()
            pre_marker.header.frame_id = "map"
            pre_marker.header.stamp = stamp
            pre_marker.ns = "pre_door_goal"
            pre_marker.id = index * 3 + 2
            pre_marker.type = Marker.SPHERE
            pre_marker.action = Marker.ADD
            pre_marker.pose.position.x = float(pre_goal_x)
            pre_marker.pose.position.y = float(pre_goal_y)
            pre_marker.pose.position.z = float(pre_goal_z)
            pre_marker.pose.orientation.w = 1.0
            pre_marker.scale.x = 0.30
            pre_marker.scale.y = 0.30
            pre_marker.scale.z = 0.30
            pre_marker.color.r = 0.1
            pre_marker.color.g = 0.3
            pre_marker.color.b = 1.0
            pre_marker.color.a = 1.0
            pre_marker.lifetime = rospy.Duration(0)
            marker_array.markers.append(pre_marker)

        self.marker_pub.publish(marker_array)
    
    def plan_callback(self, msg):
        self.current_plan = msg

    def _external_target_callback(self, msg):
        """Cache the latest high-level goal published by the goal manager."""
        self.external_target_goal = msg
    
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

        # Drop stale snapshots (perception may have stopped publishing) and
        # rank remaining doors by distance to the robot so we engage the
        # closest one first instead of "first detected" wins.
        now = rospy.Time.now()
        rx = robot_pose.pose.position.x
        ry = robot_pose.pose.position.y

        # calculates the distance to the robot for each door pose
        def _dist_to_robot(dp):
            return math.hypot(dp["position"][0] - rx, dp["position"][1] - ry)

        fresh_doors = []
        for dp in self.latest_door_poses:
            ts = dp.get("timestamp")
            if ts is not None and (now - ts).to_sec() > DOOR_POSE_MAX_AGE_SEC:
                continue # drop stale door poses
            fresh_doors.append(dp)

        fresh_doors.sort(key=_dist_to_robot)

        for door_pose in fresh_doors:
            if self.check_door_intersects_path(door_pose, robot_pose):
                self.current_door_pose_map = door_pose  # can handle only 1 door for traversal at a time
                return True
        
        return False
    
    def check_door_intersects_path(self, door_pose_map, robot_pose_map):
        """Check if door intersects future path"""
        if self.current_plan is None or len(self.current_plan.poses) < 2:
            return False

        # Plan must be in the same frame as the door pose for direct comparison.
        # If TEB ever publishes the plan in odom (or anything other than map), then i is be skipped.
        plan_frame = (self.current_plan.header.frame_id or "").strip()
        if plan_frame and plan_frame != MAP_FRAME:
            rospy.logwarn_throttle(10.0, f"Skipping door intersect: plan frame '{plan_frame}' != '{MAP_FRAME}'")
            return False

        xd, yd = door_pose_map["position"][:2]
        rx = robot_pose_map.pose.position.x
        ry = robot_pose_map.pose.position.y

        # only engage doors within the configured trigger radius.
        if math.hypot(xd - rx, yd - ry) > DOOR_TRIGGER_DISTANCE:
            return False

        # Vertical-normal sanity: a real door has a mostly-horizontal normal.
        # When perception misfires on floor/ceiling/glass the normal points
        # sharply up/down and the 2D projection of the door span becomes garbage.
        normal = door_pose_map.get("normal", [0.0, 0.0, 0.0])
        if abs(float(normal[2])) > 0.5: # if the y-component of the normal is greater than 0.5, then skip
            return False

        # Door span calculation
        door_yaw_map = math.atan2(normal[1], normal[0])
        door_width = float(door_pose_map.get("width", 0.9) or 0.9)
        # clamp to physically plausible door widths to absorb perception noise
        door_width = max(0.5, min(door_width, 1.6)) # max and min width of the door

        # Span direction perpendicular to normal
        span_yaw = door_yaw_map + math.pi / 2.0
        half_w = door_width / 2.0
        
        door_p1 = (xd + half_w * math.cos(span_yaw), yd + half_w * math.sin(span_yaw))
        door_p2 = (xd - half_w * math.cos(span_yaw), yd - half_w * math.sin(span_yaw))
        
        # closest point on path to robot
        path = self.current_plan.poses
        rx = robot_pose_map.pose.position.x
        ry = robot_pose_map.pose.position.y
        
        # index of the closest point on the path to the robot
        closest_i = min(range(len(path)), key=lambda i: (path[i].pose.position.x - rx)**2 + (path[i].pose.position.y - ry)**2)
        
        # future path segments only
        end_i = min(len(path) - 1, closest_i + LOOKAHEAD_POINTS)
        
        for i in range(closest_i, end_i):
            ax = path[i].pose.position.x
            ay = path[i].pose.position.y
            bx = path[i + 1].pose.position.x
            by = path[i + 1].pose.position.y
            
            if segments_intersect((ax, ay), (bx, by), door_p1, door_p2):
                rospy.loginfo(f"Door intersects path at segment {i}")
                return True
        
        return False
    
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
        # Save original goal to return to after door traversal.
        # Prefer the goal manager's latched target (the real high-level goal)
        # over the local plan's tail (which can be truncated by the planner).
        if self.original_goal is None:
            if self.external_target_goal is not None:
                self.original_goal = self.external_target_goal # preferred goal from goal manager
                rospy.loginfo("Saved original navigation goal (from goal manager)")
            elif self.current_plan and len(self.current_plan.poses) > 0: # this is just BACKUP plan
                self.original_goal = self.current_plan.poses[-1]
                rospy.loginfo("Saved original navigation goal (from local plan tail)")
            else:
                rospy.logwarn("No original goal source available; will not resume after door")

        rospy.loginfo("Triggering pre-door pose")
        self._speak("Door detected on path. Moving to pre-door position.")
        pre_goal = self.compute_pre_door_goal()
        if pre_goal:
            self.send_goal(pre_goal)
            self.state = DoorState.APPROACHING_DOOR
    
    def perfrom_door_state_check(self):
        # If the VLM service is unavailable, don't deadlock at AT_PRE_DOOR.
        # Best-effort: try to traverse; if the door is closed, move_base will
        # fail to reach the post-door pose and the TRAVERSING failure handler
        # will gracefully resume the original goal.
        if self.door_state_service is None:
            rospy.logwarn_throttle(10.0, "Door state service unavailable; skipping check and attempting traversal")
            self._speak("Door state check skipped. Attempting to traverse.")
            self.send_post_door_goal()
            return

        # Cooldown: don't hit the VLM at 10 Hz while waiting in AT_PRE_DOOR.
        now_ts = time.monotonic()
        if now_ts - self.last_state_service_call_ts < STATE_SERVICE_COOLDOWN_SEC:
            return
        self.last_state_service_call_ts = now_ts

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

    def _resume_original_goal(self, reason=""):
        """Resume original navigation after a *successful* door traversal.

        Only called on the SUCCESS path: the robot has actually crossed the
        door and should continue toward the goal. Failure paths must call
        ``_fail(reason)`` instead.
        """
        if reason:
            rospy.loginfo("Resuming original goal: %s", reason)
        if self.original_goal is not None:
            self.send_goal(self.original_goal)
            self.original_goal = None
        else:
            rospy.logwarn("No original goal saved; just dropping back to NAVIGATING")
        self.current_door_pose_map = None
        self.state = DoorState.NAVIGATING

    def _fail(self, reason):
        """Terminal failure: stop the robot, publish reason, sit in FAILED.

        Used when move_base aborts/rejects a pre-door or post-door goal — at
        that point retrying would just loop into the same obstacle, so we
        cancel everything and require manual recovery (operator inspects
        ``/door_coordinator/failure_reason`` and decides what to do).
        """
        rospy.logerr("Door coordination FAILED: %s", reason)
        self._speak(f"Navigation failed. {reason}. Stopping.")

        # Stop chasing the doomed goal so the robot actually halts.
        try:
            self.move_base_client.cancel_all_goals()
        except Exception as e:
            rospy.logwarn("cancel_all_goals failed: %s", e)

        self.current_door_pose_map = None
        self.original_goal = None

        try:
            self.failure_pub.publish(String(data=reason))
        except Exception as e:
            rospy.logwarn("Failed to publish failure_reason: %s", e)

        self.state = DoorState.FAILED

    def check_pre_door_reached(self):
        mb_state = self.move_base_client.get_state()
        if mb_state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Reached pre-door pose")
            self._speak("Reached pre-door position.")
            self.state = DoorState.AT_PRE_DOOR
            return

        if mb_state in self._MB_FAILURE_STATES:
            # Don't try to resume the original goal — if move_base can't reach
            # the pre-door pose, it almost certainly can't reach anything past
            # the door either. Halt and let an operator decide.
            self._fail(reason=f"pre-door goal failed (move_base state={mb_state})")
            return

        # otherwise still APPROACHING_DOOR (PENDING/ACTIVE) — keep waiting
        
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
            # "handling" only while the coordinator is *actively* engaged with
            # a door. NAVIGATING and FAILED both mean idle (the robot is not
            # being commanded by door logic), so both publish False.
            is_handling = self.state in (
                DoorState.APPROACHING_DOOR,
                DoorState.AT_PRE_DOOR,
                DoorState.TRAVERSING,
            )
            if is_handling != self._last_handling_published: # publish only when last handling not yet published, this for 10Hz rate
                try:
                    self.handling_door_pub.publish(Bool(data=is_handling)) # publish only 1st time when handling door
                except Exception as e:
                    rospy.logwarn_throttle(10.0, f"handling_door publish failed: {e}")
                self._last_handling_published = is_handling # becomes true as same as is_handling, so next time will not publish again

            if self.state == DoorState.NAVIGATING: # 0: NAVIGATING
                if self.is_door_on_path():
                    rospy.loginfo("Door detected ahead on path")
                    self.trigger_pre_door()
            
            elif self.state == DoorState.APPROACHING_DOOR: # 1: APPROACHING_DOOR
                self.check_pre_door_reached()
                
            elif self.state == DoorState.AT_PRE_DOOR: # 2: AT_PRE_DOOR
                self.perfrom_door_state_check()
            
            elif self.state == DoorState.TRAVERSING: # 3: TRAVERSING
                mb_state = self.move_base_client.get_state()
                if mb_state == actionlib.GoalStatus.SUCCEEDED:
                    # Success: robot actually crossed the door, continue navigation.
                    rospy.loginfo("Door traversal complete, resuming original goal")
                    self._speak("Door traversal complete. Resuming original goal.")
                    self._resume_original_goal()
                elif mb_state in self._MB_FAILURE_STATES:
                    # Failure during traversal: the door is the blocker. Don't
                    # try to plan around it — halt and surface the failure.
                    self._fail(reason=f"post-door goal failed (move_base state={mb_state})")
            rate.sleep()

        if self.voice_assistant is not None:
            self.voice_assistant.close()

if __name__ == "__main__":
    try:
        coordinator = DoorCoordinator()
        coordinator.spin()
    except rospy.ROSInterruptException:
        pass
