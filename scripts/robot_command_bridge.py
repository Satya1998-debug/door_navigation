#!/home/ias/satya/venv38/bin/python3

import rospy
import time
import subprocess
from std_srvs.srv import Trigger, TriggerResponse
from door_navigation.srv import StartNavigation, StartNavigationResponse

class RobotCommandBridge:
    def __init__(self):
        rospy.init_node("robot_command_bridge")

        self.door_launch_process = None
        self.door_launch_pkg = "door_navigation"
        self.door_launch_file = "door_navigation.launch"
        rospy.on_shutdown(self._shutdown_cleanup)

        # Door coordinator: fast, one-shot command
        rospy.Service("/agent/start_door_coordinator", Trigger, self.start_door_coordinator)

        # Navigation: long-running task (UDP-backed)
        rospy.Service("/agent/start_navigation", StartNavigation, self.start_navigation)

        rospy.loginfo("[Bridge] Robot Command Bridge ready")

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

    def start_navigation(self, req):
        rospy.loginfo(f"[NAV] Navigation requested → {req}")

        try:
            # req = {"person": req.person, "room": req.room}
            # TODO: send navigation goal via goal sender

            # rospy.loginfo("[NAV] Navigation started (UDP)")
            # time.sleep(5)  # simulate navigation

            # TODO: wait for UDP response instead of sleep
            success_res_udp = True  # set based on UDP response

            if success_res_udp:
                rospy.loginfo("[NAV] Navigation succeeded")
                return StartNavigationResponse(
                    success=True,
                    reason="arrived"
                )
            else:
                rospy.logerr("[NAV] Navigation failed")
                return StartNavigationResponse(
                    success=False,
                    reason="blocked"
                )

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
