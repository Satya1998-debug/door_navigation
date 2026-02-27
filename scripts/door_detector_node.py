#!/usr/bin/env python3
"""
Door Detector Node
Continuously runs YOLO detection on RGB images and publishes door detections.
Runs at ~5-10 Hz to avoid blocking navigation control loop.
"""

import rospy
from sensor_msgs.msg import Image
from door_navigation.msg import DoorDetection
from cv_bridge import CvBridge
import numpy as np

# Import detector
from door_ros_interfaces import DoorDetector


class DoorDetectorNode:
    def __init__(self):
        rospy.init_node("door_detector_node")
        
        # Parameters
        self.detection_rate = rospy.get_param("~detection_rate", 5.0)  # Hz
        self.confidence_threshold = rospy.get_param("~confidence_threshold", 0.5)
        
        # Initialize detector
        self.door_detector = DoorDetector()
        self.bridge = CvBridge()
        
        # Latest RGB image
        self.latest_rgb = None
        self.rgb_lock = rospy.Lock()
        
        # Subscribers
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", 
            Image, 
            self.rgb_callback, 
            queue_size=1, 
            buff_size=2**24
        )
        
        # Publishers
        self.detection_pub = rospy.Publisher(
            "/door/detections", 
            DoorDetection, 
            queue_size=10
        )
        
        rospy.loginfo("DoorDetectorNode initialized")
        rospy.loginfo(f"Detection rate: {self.detection_rate} Hz")
        rospy.loginfo(f"Confidence threshold: {self.confidence_threshold}")
    
    def rgb_callback(self, msg):
        """Cache latest RGB image"""
        try:
            with self.rgb_lock:
                self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logwarn(f"RGB conversion failed: {e}")
    
    def detect_and_publish(self):
        """Run YOLO detection and publish results"""
        with self.rgb_lock:
            rgb_image = self.latest_rgb
        
        if rgb_image is None:
            return
        
        try:
            # Run YOLO detection
            detections = self.door_detector.run_yolo_model(
                rgb_image=rgb_image,
                confidence_threshold=self.confidence_threshold,
                visualize=False
            )
            
            # Publish each detection
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
        rate = rospy.Rate(self.detection_rate)
        while not rospy.is_shutdown():
            self.detect_and_publish()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = DoorDetectorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
