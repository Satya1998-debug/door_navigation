#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from door_navigation.msg import DoorDetection
from cv_bridge import CvBridge
from utils.config import DOOR_DETECTION_TOPIC, LABEL_MAP, MODEL_PATH, CONFIDENCE_THRESHOLD, DETECTION_RATE, RGB_TOPIC
from door_ros_interfaces import DoorDetector

class DoorDetectorNode:
    def __init__(self):
        rospy.init_node("door_detector_node")
        
        self.door_detector = DoorDetector()
        self.bridge = CvBridge()
        
        self.latest_rgb_frame = None
        
        # subscribe to RGB images
        self.rgb_sub = rospy.Subscriber(RGB_TOPIC, Image, self.rgb_callback, queue_size=1, buff_size=2**24) # A standard 640×480 RGB image is about 0.9 MB., so buff_size is set to 16 MB to avoid dropping frames
        # publish door detections
        self.detection_pub = rospy.Publisher(DOOR_DETECTION_TOPIC, DoorDetection, queue_size=10)
        rospy.loginfo("DoorDetectorNode initialized")
    
    def rgb_callback(self, msg):
        """Cache latest RGB image"""
        try:
            self.latest_rgb_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # convert from ROS msg to OpenCV format
        except Exception as e:
            rospy.logwarn(f"RGB conversion failed: {e}")
    
    def detect_and_publish(self):
        """Run YOLO detection and publish results"""
        
        if self.latest_rgb_frame is None:
            return
        
        try:
            # Run YOLO detection
            detections = self.door_detector.run_yolo_model(rgb_image=self.latest_rgb_frame,)
            
            # publishes each detection
            stamp = rospy.Time.now()
            for det in detections:
                msg = DoorDetection()
                msg.header.stamp = stamp
                msg.header.frame_id = "camera_color_optical_frame"
                msg.bbox = det['bbox']  # [x1, y1, x2, y2]
                msg.class_id = int(det['cls_id'])
                msg.confidence = float(det['conf'])
                self.detection_pub.publish(msg)
            
            if len(detections) > 0:
                rospy.logdebug(f"Published {len(detections)} door detections")
                
        except Exception as e:
            rospy.logerr(f"Detection failed: {e}")
    
    def spin(self):
        """Main loop"""
        rate = rospy.Rate(DETECTION_RATE)
        while not rospy.is_shutdown():
            self.detect_and_publish()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorDetectorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
