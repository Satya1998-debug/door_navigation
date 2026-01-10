#!/home/satya/MT/uv_ros_py38/bin python3

import sys
import os
import cv2
import numpy as np
import rospkg
import rospy

# ------ path setup -----
# Get package path using rospkg (works with rosrun)
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_ros_interfaces import RGBDImageReciever, DoorDetector, DoorDetectionPublisher
from utils.config import *


def main(visualize_detection=True):
    rospy.init_node('door_detector_node')
    rgbd_reciever = RGBDImageReciever() # subscribe to RGBD images
    door_detector = DoorDetector() # door detector
    detection_publisher = DoorDetectionPublisher()  # door detection publisher

    frame_count = 0
    detection_interval = 5  # run detection every 5 frames to save computation
    
    # display loop in main thread 
    rate = rospy.Rate(30)  # 30hz display rate
    while not rospy.is_shutdown(): # while the node is running
        if rgbd_reciever.latest_frame_color is not None and rgbd_reciever.latest_frame_depth is not None:
            # get frames
            color_image = rgbd_reciever.latest_frame_color
            depth_image_rs = rgbd_reciever.latest_frame_depth

            detections = []
            # run detection at intervals
            if frame_count % detection_interval == 0:
                rospy.loginfo("Running door detection on frame %d", frame_count)
                # ------ get  detection results -----
                # run inference on choosen model, get results
                detections = door_detector.run_yolo_model(model_path=MODEL_PATH, 
                                                          rgb_image=color_image, 
                                                          img_size=IMG_SIZE, 
                                                          confidence_threshold=CONFIDENCE_THRESHOLD,
                                                          visualize=False)
                
                # depth correction using DepthAnything V2 (not needed in this node)
                # depth_image_corrected = door_detector.run_depth_anything_v2_on_image(color_image, depth_image_rs)

                
                # ------ ros detections publish -------
                if len(detections) > 0:
                    rospy.loginfo("Detected %d doors/handles", len(detections))
                else:
                    rospy.loginfo("No doors/handles detected")
                # detection_msg = detection_publisher.create_detections_msg(detections)
                # detection_publisher.door_detection_pub.publish(detection_msg)
                # rospy.loginfo("Published %d door detections", len(detection_msg.detections))

                # visualization
                if visualize_detection:
                    color_image = color_image.copy()
                    for vb in detections:
                        x1, y1, x2, y2 = map(int, vb['bbox'])
                        conf = vb['conf']
                        cls_id = vb['cls_id']
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{LABEL_MAP.get(cls_id, 'Unknown')} {conf:.2f}"
                        cv2.putText(color_image, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow("Door Detections", color_image)
                    cv2.waitKey(1)  # non-blocking update for window

                # TODO: door pose estimation (distance from robot)

                # TODO: check global path plan and check if door is on path

                # TODO: set /pre_door_pose_enabled to True if door is on path (pre-door pose estimation node will run then)

            frame_count += 1
        else:
            rospy.loginfo("Waiting for RGBD images...")
        rate.sleep() # maintain loop rate
    
    # cleanup
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Node shutdown") # when loop ends, shutdown node

def recieve_rgbd_images():
    rospy.init_node('rgbd_image_reciever_node')
    rgbd_reciever = RGBDImageReciever() # subscribe to RGBD images

    frames = 0
    img_count = 1
    save_interval = 5 # save every 5 seconds

    rate = rospy.Rate(30)  # 30hz display rate
    while not rospy.is_shutdown(): # while the node is running
        frames += 1
        if rgbd_reciever.latest_frame_color is not None and rgbd_reciever.latest_frame_depth is not None:
            # display the color image
            cv2.imshow("Color Image", rgbd_reciever.latest_frame_color)

            # save rgb and depth images
            if frames % (save_interval * 30) == 0:  # assuming 30 fps, save after every save_interval seconds
                print("Saving latest RGBD images...")
                cv2.imwrite(f"src/door_navigation/latest_image_color_lab_{img_count}.jpg", rgbd_reciever.latest_frame_color)
                assert rgbd_reciever.latest_frame_depth.dtype == np.uint16, "Depth image is not in uint16 format"
                cv2.imwrite(f"src/door_navigation/latest_image_depth_lab_{img_count}.png", rgbd_reciever.latest_frame_depth)
                img_count += 1
            
            # create depth visualization
            depth_display = rgbd_reciever.latest_frame_depth.astype(np.float32) / 1000.0  # convert mm to meters for display
            max_depth = 5.0  # max depth for visualization
            min_depth = np.nanmin(depth_display[depth_display > 0])  # ignore
            norm_depth = depth_display/max_depth if max_depth > min_depth else depth_display
            depth_vis = (norm_depth * 255).astype(np.uint8)  # scale
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", depth_vis)
        
        # handle cv2 events and check for ESC key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            rospy.loginfo("ESC pressed, shutting down...")
            break
        
        rate.sleep()
    
    # cleanup
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Node shutdown")
    

if __name__ == '__main__':
    main()