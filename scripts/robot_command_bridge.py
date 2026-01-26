#!/usr/bin/env python3

import rospy
import time
from std_srvs.srv import Trigger, TriggerResponse
from door_navigation.srv import StartNavigation, StartNavigationResponse

class RobotCommandBridge:
    def __init__(self):
        rospy.init_node("robot_command_bridge")

        # Door coordinator: fast, one-shot command
        rospy.Service(
            "/agent/start_door_coordinator",
            Trigger,
            self.start_door_coordinator
        )

        # Navigation: long-running task (UDP-backed)
        rospy.Service(
            "/agent/start_navigation",
            StartNavigation,
            self.start_navigation
        )

        rospy.loginfo("[Bridge] Robot Command Bridge ready")

    # -------------------------------------------------
    # Navigation service
    # -------------------------------------------------

    def start_navigation(self, req):
        rospy.loginfo(f"[NAV] Navigation requested → target: {req.target}")

        try:
            # TODO: send UDP command to robot
            # udp_send("NAVIGATE", req.target)

            rospy.loginfo("[NAV] Navigation started (UDP)")
            time.sleep(5)  # simulate navigation

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
        rospy.loginfo("[DOOR] Starting door coordinator...")

        try:
            # TODO: actual door coordinator startup
            time.sleep(1)
            rospy.loginfo("[DOOR] Door coordinator started")
            return TriggerResponse(True, "Door coordinator started")

        except Exception as e:
            rospy.logerr(f"[DOOR] Failed to start door coordinator: {e}")
            return TriggerResponse(False, str(e))


if __name__ == "__main__":
    RobotCommandBridge()
    rospy.spin()
