#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

bridge = CvBridge()

COLOR_WINDOW = "Color"
DEPTH_WINDOW = "Depth"

last_color = None
last_depth = None

def color_callback(msg):
    try:
        # RealSense color is often RGB8; convert to BGR for OpenCV display.
        cv_color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        global last_color
        last_color = cv2.cvtColor(cv_color_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        rospy.logerr("Error converting color image: %s", e)

def depth_callback(msg):
    try:
        cv_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_depth_np = np.array(cv_depth_image, dtype=np.float32)

        # Normalize for display and apply a colormap for readability.
        depth_norm = cv2.normalize(cv_depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_u8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

        global last_depth
        last_depth = depth_color
    except Exception as e:
        rospy.logerr("Error converting depth image: %s", e)

def main():
    rospy.init_node("camera_tester", anonymous=True)  # many instances with same node name can run
    rospy.loginfo("Camera tester node started.")

    cv2.namedWindow(COLOR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEPTH_WINDOW, cv2.WINDOW_NORMAL)

    
    color_image_topic = "/camera/color/image_raw"
    rospy.Subscriber(color_image_topic, Image, color_callback)

    depth_image_topic = "/camera/depth/image_rect_raw"
    rospy.Subscriber(depth_image_topic, Image, depth_callback)

    rospy.on_shutdown(cv2.destroyAllWindows)

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        if last_color is not None:
            cv2.imshow(COLOR_WINDOW, last_color)
        if last_depth is not None:
            cv2.imshow(DEPTH_WINDOW, last_depth)
        cv2.waitKey(1)
        rate.sleep()

if __name__ == "__main__":
    main()