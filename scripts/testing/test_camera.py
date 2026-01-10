#!/home/satya/MT/uv_ros_py38/bin python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

bridge = CvBridge()

def color_callback(msg):
    try:
        cv_color_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Received color image of size: %s", cv_color_image.shape)
    except Exception as e:
        print("Error converting color image: %s", e)

def depth_callback(msg):
    try:
        cv_depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_depth_np = np.array(cv_depth_image, dtype=np.float32)  # ensure depth is in float32, because depth images are usually in float format
        print("Received depth image of size: %s", cv_depth_image.shape)

        # max min depth values for visualization
        min_depth = np.nanmin(cv_depth_np)
        max_depth = np.nanmax(cv_depth_np)
        print("Depth image min: %f, max: %f", min_depth, max_depth)
    except Exception as e:
        print("Error converting depth image: %s", e)

def main():
    rospy.init_node('camera_tester', anonymous=True) # many instances with same node name can run
    print("Camera tester node started.")

    
    color_image_topic = "/camera/color/image_raw"
    rospy.Subscriber(color_image_topic, Image, color_callback)

    depth_image_topic = "/camera/depth/image_rect_raw"
    rospy.Subscriber(depth_image_topic, Image, depth_callback)

    rospy.spin()  # will keep the node running

if __name__ == "__main__":
    main()