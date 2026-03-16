#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_synced_frames():
    rospy.init_node('sync_image_publisher')
    
    # Publishers
    rgb_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    depth_pub = rospy.Publisher('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    
    bridge = CvBridge()
    rate = rospy.Rate(2) # 10 Hz

    # Load images once to save CPU
    rgb_path = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_color_lab_35.jpg"
    depth_path = "/home/satya/MT/catkin_ws/src/door_navigation/scripts/data_new/latest_image_depth_lab_35.png"

    cv_rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    cv_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # Keep 16-bit

    if cv_rgb is None or cv_depth is None:
        rospy.logerr("Could not load images! Check your paths.")
        return

    rospy.loginfo("Publishing synced RGB and Depth frames...")

    while not rospy.is_shutdown():
        # 1. Capture ONE timestamp for both
        now = rospy.Time.now()

        # 2. Create RGB message
        rgb_msg = bridge.cv2_to_imgmsg(cv_rgb, encoding="bgr8")
        rgb_msg.header.stamp = now
        rgb_msg.header.frame_id = "camera_link"

        # 3. Create Depth message
        depth_msg = bridge.cv2_to_imgmsg(cv_depth, encoding="16UC1")
        depth_msg.header.stamp = now
        depth_msg.header.frame_id = "camera_link"

        # 4. Publish
        rgb_pub.publish(rgb_msg)
        depth_pub.publish(depth_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_synced_frames()
    except rospy.ROSInterruptException:
        pass