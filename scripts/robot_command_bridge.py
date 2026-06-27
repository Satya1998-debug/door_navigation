#!/home/ias/satya/venv38/bin/python3

import sys
import time
import os
import subprocess
import rospkg
import rospy
import subprocess
from std_srvs.srv import Trigger, TriggerResponse
from door_navigation.srv import StartNavigation, StartNavigationResponse

# --- path setup ---
try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    # Fallback: utils/config.py -> scripts/utils -> scripts -> door_navigation
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"[CONFIG] rospkg not available, using relative path: {PACKAGE_PATH}")
    
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from goal_sender import GoalManager

class RobotCommandBridge:
    def __init__(self):
        rospy.init_node("robot_command_bridge")

        self.door_launch_process = None
        self.door_launch_pkg = "door_navigation"
        self.door_launch_file = "door_navigation.launch"

        # Simple navigation backend: publish goal and wait by distance-to-target.
        self.nav_wait_timeout_sec = 300
        self.nav_position_tolerance = 0.35
        
        # starts the goal manager node in the background, 
        # which will listen for goal commands
        self.goal_manager = GoalManager(init_node=False, 
                                        enable_inactivity_thread=False)

        rospy.on_shutdown(self._shutdown_cleanup)
        # Door coordinator: fast, one-shot command
        rospy.Service("/agent/start_door_coordinator", Trigger, self.start_door_coordinator)
        # Navigation: long-running task
        rospy.Service("/agent/start_navigation", StartNavigation, self.start_navigation)
        rospy.loginfo("[Bridge] Robot Command Bridge ready")
        
        self.testing = True

    def _shutdown_cleanup(self):
        if self.door_launch_process and self.door_launch_process.poll() is None:
            rospy.loginfo("[DOOR] Stopping door coordinator launch process")
            self.door_launch_process.terminate()
            try:
                self.door_launch_process.wait(timeout=5)
            except Exception:
                self.door_launch_process.kill()

    # -------------------------------------------------
    # Navigation service
    # -------------------------------------------------

    def start_navigation(self, req): # req has person and room
        rospy.loginfo(f"[NAV] Navigation requested -> person: {req.person}, room: {req.room}")

        try:
            if self.testing:
                rospy.loginfo("[NAV] Testing mode: simulating navigation")
                time.sleep(5) # simulate some delay
                
                # call door state estimator test module
                from testing.test_interfaces import RosTestInterface
                test_interface = RosTestInterface(testing=True)
                if test_interface.call_door_state_estimator():
                    rospy.loginfo("[NAV] Simulated navigation success after door state estimation")
                    return StartNavigationResponse(True, "simulated_success")
                else:
                    rospy.loginfo("[NAV] Simulated navigation failure after door state estimation")
                    return StartNavigationResponse(False, "simulated_failure")
            
            location_key = (req.room or "").strip()
            if not location_key:
                location_key = (req.person or "").strip()
            if not location_key:
                return StartNavigationResponse(False, "empty_target") # success=False, reason="empty_target"

            ok, reason = self.goal_manager.send_goal(location_key) # returns (True, "") if goal sent successfully, otherwise (False, "reason")
            if not ok: # if sending goal failed
                return StartNavigationResponse(False, reason)

            # block and wait until we reach the target or timeout or failure status
            arrived, wait_reason = self.goal_manager.wait_for_target_reached(timeout_sec=self.nav_wait_timeout_sec,
                                                                             position_tolerance=self.nav_position_tolerance,
                                                                             use_status_failures=True,
                                                                             enable_status_check=True,
                                                                             enable_timeout=True)
            return StartNavigationResponse(arrived, wait_reason)

        except Exception as e:
            rospy.logerr(f"[NAV] Navigation error: {e}")
            return StartNavigationResponse(
                success=False,
                reason=str(e)
            )

    # -------------------------------------------------
    # Door coordinator service
    # -------------------------------------------------

    def start_door_coordinator(self, req):
        """
        This roslaunch the launch file, that starts:
            - door_detect_pose_estimate_node
            - door_state_estimator_node (a service)
            - door_coordinator_node
            - camera_node
        """
        rospy.loginfo("[DOOR] Starting door coordinator...")

        try:
            if self.door_launch_process and self.door_launch_process.poll() is None:
                rospy.loginfo("[DOOR] Door coordinator already running")
                return TriggerResponse(True, "Door coordinator already running")

            cmd = ["roslaunch", self.door_launch_pkg, self.door_launch_file] # >> roslaunch door_navigation door_navigation.launch
            
            rospy.loginfo(f"[DOOR] Executing: {' '.join(cmd)}")
            self.door_launch_process = subprocess.Popen(cmd)

            rospy.sleep(1.0) # give it a moment to start and check if it exited immediately
            
            if self.door_launch_process.poll() is not None:
                raise RuntimeError("roslaunch process exited immediately")
            
            rospy.loginfo("[DOOR] Door coordinator started (launch file)")
            return TriggerResponse(True, "Door coordinator started")

        except Exception as e:
            rospy.logerr(f"[DOOR] Failed to start door coordinator: {e}")
            return TriggerResponse(False, str(e))


if __name__ == "__main__":
    RobotCommandBridge()
    rospy.spin()
