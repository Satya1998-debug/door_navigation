#!/home/ias/satya/venv38/bin/python3

"""Agent-facing bridge exposing a single navigation service.

Preconditions (started manually by the operator):
- roscore
- navigation stack (move_base + localization)
- rosbridge_server (for the LLM agent side via websocket)
- roslaunch door_navigation door_navigation.launch (camera + pose estimator +
  state estimator + coordinator)

This node only wraps :class:`GoalManager` behind a ROS service so an external
agent (e.g. robotdog_guide over rosbridge) can request navigation by
(person, room) without knowing the robot side details.
"""

import os
import signal
import sys

import rospkg
import rospy
from door_navigation.srv import StartNavigation, StartNavigationResponse

try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"[CONFIG] rospkg not available, using relative path: {PACKAGE_PATH}")

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from goal_sender import GoalManager


class RobotCommandBridge:
    def __init__(self):
        rospy.init_node("robot_command_bridge")

        self.nav_wait_timeout_sec = float(rospy.get_param("~nav_wait_timeout_sec", 500.0))
        self.nav_position_tolerance = float(rospy.get_param("~nav_position_tolerance", 0.15))

        self.goal_manager = GoalManager(init_node=False, enable_inactivity_thread=False)

        rospy.Service("/agent/start_navigation", StartNavigation, self.start_navigation)

        # log which signal (if any) tore us down so ros_shutdown responses
        # in the guide log can be traced back to their real cause
        # (SIGINT=operator Ctrl-C, SIGTERM=roslaunch/kill, SIGHUP=terminal closed).
        for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(_sig, lambda s, _f: rospy.logwarn(
                "[Bridge] received signal %s (%s); shutting down.", s, signal.Signals(s).name
            ))

        rospy.loginfo("[Bridge] Robot Command Bridge ready (service: /agent/start_navigation)")

    def start_navigation(self, req):
        rospy.loginfo(
            "[NAV] Navigation requested -> person: %s, room: %s", req.person, req.room)

        try:
            location_key = (req.room or "").strip()
            if not location_key:
                location_key = (req.person or "").strip()
            if not location_key:
                return StartNavigationResponse(False, "empty_target")

            ok, reason = self.goal_manager.send_goal(location_key)
            if not ok:
                return StartNavigationResponse(False, reason)

            arrived, wait_reason = self.goal_manager.wait_for_target_reached(
                timeout_sec=self.nav_wait_timeout_sec,
                position_tolerance=self.nav_position_tolerance,
                use_status_failures=True,
                enable_status_check=True,
                enable_timeout=True,
            )
            return StartNavigationResponse(arrived, wait_reason)

        except Exception as e:
            rospy.logerr(f"[NAV] Navigation error: {e}")
            return StartNavigationResponse(False, str(e))


if __name__ == "__main__":
    RobotCommandBridge()
    rospy.spin()
