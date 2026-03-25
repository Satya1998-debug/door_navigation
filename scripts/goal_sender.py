#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatusArray
import yaml
import threading

class GoalManager:
    def __init__(self):
        rospy.init_node('goal_sender', anonymous=True)
        self.pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.status_callback)

        # Load locations from YAML file
        self.locations = self.load_locations('/home/unitree/UnitreeSLAM/catkin_lidar_slam_3d/src/lidar_slam_3d/a2_ros2udp/params/locations.yaml')

        # Define home location
        self.home_location = self.locations.get("Home")
        if not self.home_location:
            rospy.logerr("Home location is not defined in the YAML file.")
            return

        # Set initial variables
        self.last_command_time = rospy.get_time()
        self.command_received = False
        self.goal_reached = False
        self.goal_reached_logged = False

        # Start the timer thread
        self.timer_thread = threading.Thread(target=self.check_inactivity)
        self.timer_thread.start()

    def load_locations(self, yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def send_goal(self, location_name):
        if location_name not in self.locations:
            rospy.logerr("Location '{}' not found in the locations file.".format(location_name))
            return

        location = self.locations[location_name]
        goal = PoseStamped()
        goal.header.frame_id = location['header']['frame_id']
        goal.pose.position.x = location['pose']['position']['x']
        goal.pose.position.y = location['pose']['position']['y']
        goal.pose.position.z = location['pose']['position']['z']
        goal.pose.orientation.x = location['pose']['orientation']['x']
        goal.pose.orientation.y = location['pose']['orientation']['y']
        goal.pose.orientation.z = location['pose']['orientation']['z']
        goal.pose.orientation.w = location['pose']['orientation']['w']
        
        # Publish the goal
        self.pub.publish(goal)
        rospy.loginfo("Goal sent to {}.".format(location_name))
        self.command_received = True
        self.goal_reached = False  # Reset goal reached status
        self.goal_reached_logged = False
        self.last_command_time = rospy.get_time()

    def status_callback(self, msg):
        # Check if the goal is reached
        if len(msg.status_list) > 0:
            status = msg.status_list[-1].status
            # Status 3 means "Goal Reached"
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
                self.send_goal("Home")
                self.command_received = False  # Reset command received flag
                self.goal_reached = False  # Reset goal reached flag
            rate.sleep()

    def run(self):
        # Main loop to take user input
        while not rospy.is_shutdown():
            try:
                user_input = raw_input("Enter location name (e.g., 'Office_A'): ")
                self.send_goal(user_input.strip())
            except rospy.ROSInterruptException:
                break

if __name__ == '__main__':
    manager = GoalManager()
    manager.run()

