#!/usr/bin/env python3
"""
Fake move_base action server for offline bag/coordinator testing.

Behavior:
- Exposes action server at: /move_base
- Accepts every goal
- After a configurable delay, returns SUCCEEDED (default)
- Can be configured to ABORT for failure-path testing
"""

import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseFeedback, MoveBaseResult


class FakeMoveBaseServer:
    def __init__(self):
        rospy.init_node("fake_move_base_server")

        self.result_mode = rospy.get_param("~result_mode", "succeeded").strip().lower()
        self.goal_delay_sec = float(rospy.get_param("~goal_delay_sec", 2.0))
        self.use_wall_time = bool(rospy.get_param("~use_wall_time", True))
        self.publish_feedback_hz = float(rospy.get_param("~publish_feedback_hz", 2.0))

        self.server = actionlib.SimpleActionServer(
            "move_base", MoveBaseAction, execute_cb=self._execute_cb, auto_start=False
        )
        self.server.start()
        rospy.loginfo(
            "fake_move_base_server ready | result_mode=%s delay=%.2fs use_wall_time=%s",
            self.result_mode,
            self.goal_delay_sec,
            self.use_wall_time,
        )

    def _sleep_step(self, dt_sec):
        if self.use_wall_time:
            time.sleep(max(0.0, dt_sec))
        else:
            rospy.sleep(max(0.0, dt_sec))

    def _execute_cb(self, goal):
        target = goal.target_pose
        rospy.loginfo(
            "fake_move_base: goal received | frame=%s x=%.2f y=%.2f",
            target.header.frame_id,
            target.pose.position.x,
            target.pose.position.y,
        )

        start = time.monotonic()
        feedback_period = 1.0 / max(self.publish_feedback_hz, 0.1)
        next_feedback_ts = start
        feedback = MoveBaseFeedback()
        feedback.base_position = target

        while not rospy.is_shutdown():
            if self.server.is_preempt_requested():
                self.server.set_preempted()
                rospy.loginfo("fake_move_base: goal preempted")
                return

            now = time.monotonic()
            elapsed = now - start
            if elapsed >= self.goal_delay_sec:
                break

            if now >= next_feedback_ts:
                feedback.base_position.header.stamp = rospy.Time.now()
                self.server.publish_feedback(feedback)
                next_feedback_ts += feedback_period

            self._sleep_step(0.02)

        result = MoveBaseResult()
        mode = self.result_mode
        if mode in ("abort", "aborted", "fail", "failed"):
            self.server.set_aborted(result, text="fake_move_base configured ABORTED")
            rospy.logwarn("fake_move_base: returning ABORTED")
        elif mode in ("reject", "rejected"):
            # set_rejected is available in actionlib SimpleActionServer.
            self.server.set_rejected(result, text="fake_move_base configured REJECTED")
            rospy.logwarn("fake_move_base: returning REJECTED")
        else:
            self.server.set_succeeded(result, text="fake_move_base configured SUCCEEDED")
            rospy.loginfo("fake_move_base: returning SUCCEEDED")


if __name__ == "__main__":
    try:
        FakeMoveBaseServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
