#!/home/satya/MT/uv_ros_py38/bin python3
import rospy
import actionlib
import socket
import threading
import time
from move_base_msgs.msg import MoveBaseAction, MoveBaseFeedback, MoveBaseResult

class UdpNavigationActionServer:
    def __init__(self):
        rospy.init_node("udp_navigation_action_server")

        # UDP setup
        self.robot_addr = ("192.168.1.50", 9000)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1.0)

        self._nav_done = False
        self._nav_success = False

        self.server = actionlib.SimpleActionServer(
            "/agent/start_navigation",
            MoveBaseAction,
            execute_cb=self.execute,
            auto_start=False
        )
        self.server.start()

        rospy.loginfo("[NAV] UDP-backed navigation action server ready")

    def execute(self, goal):
        rospy.loginfo("[NAV] Action goal received → sending UDP")

        self._nav_done = False
        self._nav_success = False

        # 1️⃣ Send UDP command
        self.sock.sendto(b"NAVIGATE_TO_DOOR", self.robot_addr)

        # 2️⃣ Start UDP listener
        listener = threading.Thread(target=self._listen_udp)
        listener.daemon = True
        listener.start()

        rate = rospy.Rate(10)
        feedback = MoveBaseFeedback()

        while not self._nav_done:
            if self.server.is_preempt_requested():
                rospy.logwarn("[NAV] Preempt requested")
                self.sock.sendto(b"CANCEL_NAV", self.robot_addr)
                self.server.set_preempted()
                return

            # Optional feedback
            self.server.publish_feedback(feedback)
            rate.sleep()

        # 3️⃣ Finish action
        result = MoveBaseResult()
        if self._nav_success:
            rospy.loginfo("[NAV] Navigation succeeded")
            self.server.set_succeeded(result)
        else:
            rospy.logerr("[NAV] Navigation failed")
            self.server.set_aborted(result)

    def _listen_udp(self):
        try:
            data, _ = self.sock.recvfrom(1024)
            if data == b"NAV_SUCCESS":
                self._nav_success = True
            else:
                self._nav_success = False
        except socket.timeout:
            self._nav_success = False
        finally:
            self._nav_done = True


if __name__ == "__main__":
    UdpNavigationActionServer()
    rospy.spin()
