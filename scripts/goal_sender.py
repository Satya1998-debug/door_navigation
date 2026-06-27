#!/usr/bin/env python3

import argparse
import rospy
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
import yaml
import threading
import os
import glob
import math
import tf2_ros

try:
    import rospkg
except Exception:
    rospkg = None
    
GOAL_TOPIC = '/goal'
GOAL_STATUS_TOPIC = '/move_base/status'
BASE_FRAME = 'base_link'
POSITION_TOLERANCE = 0.35  # meters

class GoalManager:
    def __init__(self, init_node=True, enable_inactivity_thread=True, locations_yaml_path=None):
        if init_node:
            rospy.init_node('goal_sender', anonymous=True)

        self.goal_topic = GOAL_TOPIC
        self.status_topic = GOAL_STATUS_TOPIC
        self.base_frame = BASE_FRAME

        self.pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=10) # publishes goal
        # latched current target from the goal manager
        self.current_target_pub = rospy.Publisher('/goal_manager/current_target', PoseStamped, queue_size=1, latch=True)
        rospy.Subscriber(self.status_topic, GoalStatusArray, self.status_callback) # check goal status

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Optional explicit YAML path override for standalone usage.
        self.locations_yaml_path = locations_yaml_path

        # Load locations from YAML file
        self.locations = self.load_locations(self._resolve_locations_yaml())

        # Define home location
        self.home_location_name = rospy.get_param('~home_location', 'Home')
        self.home_location = self.locations.get(self.home_location_name)
        if not self.home_location:
            rospy.logwarn("Home location is not defined in the locations file.")

        # Set initial variables
        self.last_command_time = rospy.get_time()
        self.command_received = False
        self.goal_reached = False
        self.goal_reached_logged = False
        self.latest_status = None
        self.latest_status_text = ""
        self.current_target_goal = None

        self.timer_thread = None
        if enable_inactivity_thread:
            self.timer_thread = threading.Thread(target=self.check_inactivity)
            self.timer_thread.daemon = True
            self.timer_thread.start()

    def _resolve_locations_yaml(self):
        """Pick the locations YAML to load.

        Priority:
          1. Explicit constructor override (standalone CLI).
          2. ROS param ``~locations_yaml`` (explicit override).
          3. ``saved_locations.yaml`` inside the door_navigation package.
          4. First matching ``saved_locations*.yaml`` in the package (sorted).
          5. Legacy hardcoded Unitree path (will fail loudly if missing).
        """
        if self.locations_yaml_path:
            rospy.loginfo("locations_yaml (from argument): %s", self.locations_yaml_path)
            return self.locations_yaml_path

        param_path = rospy.get_param('~locations_yaml', '')
        if param_path:
            rospy.loginfo("locations_yaml (from ROS param): %s", param_path)
            return param_path

        if rospkg is not None:
            try:
                pkg_path = rospkg.RosPack().get_path('door_navigation')
                preferred = os.path.join(pkg_path, 'saved_locations.yaml')
                if os.path.exists(preferred):
                    rospy.loginfo("locations_yaml (package default): %s", preferred)
                    return preferred

                matches = sorted(glob.glob(os.path.join(pkg_path, 'saved_locations*.yaml')))
                if matches:
                    rospy.loginfo("locations_yaml (package fallback, first match): %s",matches[0])
                    return matches[0]
            except Exception as e:
                rospy.logwarn("Failed to resolve locations YAML from package: %s", e)

        default_path = '/home/unitree/UnitreeSLAM/catkin_lidar_slam_3d/src/lidar_slam_3d/a2_ros2udp/params/locations.yaml'
        rospy.logwarn("locations_yaml falling back to legacy path: %s", default_path)
        return default_path

    def load_locations(self, yaml_file):
        try:
            with open(yaml_file, 'r') as file:
                data = yaml.safe_load(file) or {}

            if isinstance(data, dict) and 'locations' in data and isinstance(data['locations'], dict):
                return data['locations']
            if isinstance(data, dict):
                return data

            rospy.logerr("Invalid locations YAML format in %s", yaml_file)
            return {}
        except Exception as e:
            rospy.logerr("Failed to load locations YAML %s: %s", yaml_file, e)
            return {}

    def send_goal(self, location_name: str):
        location_name = (location_name or '').strip()

        if location_name not in self.locations:
            msg = "Location '{}' not found in the locations file.".format(location_name) # message becomes the reason
            rospy.logerr(msg)
            return False, msg

        location = self.locations[location_name] # get pose from locations yaml
        # formulate PoseStamped message from the location data
        try:
            goal = PoseStamped()
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = location['header']['frame_id']
            goal.pose.position.x = location['pose']['position']['x']
            goal.pose.position.y = location['pose']['position']['y']
            goal.pose.position.z = location['pose']['position']['z']
            goal.pose.orientation.x = location['pose']['orientation']['x']
            goal.pose.orientation.y = location['pose']['orientation']['y']
            goal.pose.orientation.z = location['pose']['orientation']['z']
            goal.pose.orientation.w = location['pose']['orientation']['w']
        except Exception as e:
            msg = "Invalid location pose for '{}': {}".format(location_name, e)
            rospy.logerr(msg)
            return False, msg
        
        # Publish the goal to the goal topic for the robot navigation
        self.pub.publish(goal)
        # publish on the latched current-target topic for door coordinator node to use
        try:
            self.current_target_pub.publish(goal)
        except Exception as e:
            rospy.logwarn("Failed to publish current_target: %s", e)
        rospy.loginfo("Goal sent to {}.".format(location_name))
        self.command_received = True
        self.goal_reached = False  # Reset goal reached status
        self.goal_reached_logged = False
        self.latest_status = None
        self.latest_status_text = ""
        self.current_target_goal = goal
        self.last_command_time = rospy.get_time()
        # this returns True when the send is success, not that it reached goal
        return True, "goal_sent:{}".format(location_name)

    def _distance_to_current_target(self):
        if self.current_target_goal is None:
            return None

        target_frame = self.current_target_goal.header.frame_id or 'map'
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(0.5)
            )
            dx = tf.transform.translation.x - self.current_target_goal.pose.position.x
            dy = tf.transform.translation.y - self.current_target_goal.pose.position.y
            return math.hypot(dx, dy)
        except Exception:
            return None

    def wait_for_target_reached(self, timeout_sec=300.0, position_tolerance=POSITION_TOLERANCE, use_status_failures=False,
                                enable_status_check=False, enable_timeout=True):
        """Block until robot reaches target by distance, or timeout. This for the FINAL GOAL of the navigation.

        If use_status_failures is True, terminal move_base failures are also treated as failure.
        If require_status_success is True, arrival requires BOTH distance and status=SUCCEEDED.
        """
        if enable_timeout: # this will check if the navigation takes longer than the timeout and return failure if it does
            deadline = rospy.Time.now() + rospy.Duration(float(timeout_sec))
        
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            dist = self._distance_to_current_target()
            dist_reached = dist is not None and dist <= float(position_tolerance)
            status_succeeded = self.latest_status == 3

            if enable_status_check: # when enabled, we require both distance reached and status succeeded to return success
                if dist_reached and status_succeeded:
                    return True, "arrived"
            else: # else only distance is checked for arrival
                if dist_reached:
                    return True, "arrived"

            # Optional hard failures from move_base status stream.
            if use_status_failures and self.latest_status in [4, 5, 8, 9]:
                text = self.latest_status_text or "navigation failed"
                return False, "move_base_failed:{}".format(text)

            if enable_timeout and rospy.Time.now() >= deadline:
                return False, "navigation_timeout_after_{}s".format(int(timeout_sec))

            rate.sleep()

        return False, "ros_shutdown"

    def status_callback(self, msg):
        # check if the goal is reached
        # uint8 PENDING         = 0   # The goal has yet to be processed by the action server
        # uint8 ACTIVE          = 1   # The goal is currently being processed by the action server
        # uint8 PREEMPTED       = 2   # The goal received a cancel request after it started executing
        #                             #   and has since completed its execution (Terminal State)
        # uint8 SUCCEEDED       = 3   # The goal was achieved successfully by the action server (Terminal State)
        # uint8 ABORTED         = 4   # The goal was aborted during execution by the action server due
        #                             #    to some failure (Terminal State)
        # uint8 REJECTED        = 5   # The goal was rejected by the action server without being processed,
        #                             #    because the goal was unattainable or invalid (Terminal State)
        # uint8 PREEMPTING      = 6   # The goal received a cancel request after it started executing
        #                             #    and has not yet completed execution
        # uint8 RECALLING       = 7   # The goal received a cancel request before it started executing,
        #                             #    but the action server has not yet confirmed that the goal is canceled
        # uint8 RECALLED        = 8   # The goal received a cancel request before it started executing
        #                             #    and was successfully cancelled (Terminal State)
        # uint8 LOST            = 9   # An action client can determine that a goal is LOST. This should not be
        #                             #    sent over the wire by an action server
        
        # status list contains the status of all current goals, we check the last one which is the most recent goal
        if len(msg.status_list) > 0:
            # status item: goal_id, status, text, 
            latest_goal = msg.status_list[-1]
            status = latest_goal.status
            self.latest_status = status
            self.latest_status_text = latest_goal.text # this text often contains error messages when status is a failure
            # status == 3 >>> "Goal Reached"
            if status == 3 and not self.goal_reached_logged:
                rospy.loginfo("Goal reached.")
                self.goal_reached = True
                self.goal_reached_logged = True
                self.last_command_time = rospy.get_time()  # Reset timer after reaching the goal
        else:
            self.goal_reached = False
            self.goal_reached_logged = False

    def check_inactivity(self):
        # Monitor inactivity and send the robot back to Home if needed
        rate = rospy.Rate(1)  # Check every second
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if self.goal_reached and (current_time - self.last_command_time > 10):  # 10 seconds of inactivity
                rospy.loginfo("No command received for 10 seconds after reaching a goal, returning to Home.")
                if self.home_location_name in self.locations:
                    self.send_goal(self.home_location_name)
                    self.command_received = False  # Reset command received flag
                    self.goal_reached = False  # Reset goal reached flag
            rate.sleep()


if __name__ == "__main__":
    # USAGE:
    # send location
    # rosrun door_navigation goal_sender.py --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml home
    # list locations
    # rosrun door_navigation goal_sender.py --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml --list
    try:
        parser = argparse.ArgumentParser(
            description="Send saved navigation goals by location name."
        )
        parser.add_argument(
            "target",
            nargs="?",
            help="Location name from the YAML file (optional; prompts if omitted).",
        )
        parser.add_argument(
            "-l", "--list",
            action="store_true",
            help="List available location names and exit.",
        )
        parser.add_argument(
            "-y", "--yaml",
            dest="yaml_path",
            default=None,
            help="Path to locations YAML file to use.",
        )

        cli_args = parser.parse_args(rospy.myargv()[1:])
        yaml_override = os.path.expanduser(cli_args.yaml_path) if cli_args.yaml_path else None

        gm = GoalManager(
            init_node=True,
            enable_inactivity_thread=False,
            locations_yaml_path=yaml_override,
        )

        if not gm.locations:
            rospy.logerr("No locations available. Check ~locations_yaml.")
            raise SystemExit(1)

        if cli_args.list:
            rospy.loginfo("Available locations:")
            for name in sorted(gm.locations.keys()):
                rospy.loginfo("  - %s", name)
            raise SystemExit(0)

        if cli_args.target:
            target = cli_args.target
        else:
            print("\nAvailable locations:")
            for name in sorted(gm.locations.keys()):
                print(f"  - {name}")
            target = input("\nEnter location name: ").strip()

        ok, reason = gm.send_goal(target)
        if not ok:
            rospy.logerr("Failed to send goal: %s", reason)
            raise SystemExit(1)

        rospy.loginfo("Goal accepted: %s", reason)
        rospy.loginfo("Use `rostopic echo /move_base/status` to monitor progress.")
    except rospy.ROSInterruptException:
        pass
