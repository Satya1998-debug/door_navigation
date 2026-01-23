#!/home/satya/MT/uv_ros_py38/bin python3

from time import time
import ros
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String
import actionlib
from door_navigation.msg import NavigateTaskAction, NavigateTaskResult, NavigateTaskFeedback

class RobotCommandBridge:
    def __init__(self):
        rospy.init_node("robot_command_bridge")

        # service to start door coordinator (services are used for fast one-time blocking commands)
        rospy.Service("/agent/start_door_coordinator", Trigger, self.start_door_coordinator)

        # navigation action server (actions are used for long-running tasks with feedback)
        # TODO: Replace MoveBaseAction with the actual navigation action used
        self.nav_action = actionlib.SimpleActionServer(
            "/agent/start_navigation",
            NavigateTaskAction,
            execute_cb=self.execute_navigation,
            auto_start=False # disable auto start to control when to start i.e race conditions
        )
        self.nav_action.start()

        rospy.loginfo("[Bridge] Robot Command Bridge ready")

    def execute_navigation(self, goal):
        rospy.loginfo("[NAV] Navigation started")
        
        # send feedback
        feedback = NavigateTaskFeedback()
        feedback.status = "Navigating"
        self.nav_action.publish_feedback(feedback)  # send feedback to client explicitly

        # TODO: navigation happening here... 
        # NOTE: (navigation bridge using UDP)
        time.sleep(10)
        
        # TODO: wait for UDP response instead of sleep

        rospy.loginfo("[NAV] Navigation completed")
        # send result (when UDP response received and is successful)
        success_res_udp = True  # TODO: set based on UDP response
        result = NavigateTaskResult()
        result.success = success_res_udp
        if success_res_udp:
            result.reason = "arrived"
            rospy.loginfo("[NAV] Navigation succeeded")
            self.nav_action.set_succeeded(result)  # send success result implicitly
        else:
            result.reason = "aborted"
            rospy.logerr("[NAV] Navigation failed")
            self.nav_action.set_aborted(result) # send failure result implicitly
        
    def execute_navigation_test(self, goal):
        time.sleep(5)
        rospy.loginfo("[NAV] Navigation started")

    def start_door_coordinator(self, req):
        rospy.loginfo("[DOOR] Starting door coordinator...")
        try:
            # Example: launch another ROS node
            # subprocess.Popen(["roslaunch", "door_pkg", "door.launch"])

            # TODO: door coordinator starting logic here...
            # NOTE: launch door coordinator ROS node (directly launch DOOR COORDINATOR node using ROS launch or subprocess)
            time.sleep(2) # to simulate door coordinator startup
            rospy.loginfo("[DOOR] Door coordinator started")
            return TriggerResponse(True, "Door coordinator started")

        except Exception as e:
            rospy.logerr(f"[DOOR] Failed to start door coordinator: {e}")
            return TriggerResponse(False, str(e))


if __name__ == "__main__":
    RobotCommandBridge()
    rospy.spin()
