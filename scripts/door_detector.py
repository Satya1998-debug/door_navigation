#!/home/satya/MT/uv_ros_py38/bin python3

import sys
import os
import cv2
import numpy as np
import torch
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

# Add script directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_navigation.scripts.utils.config import *
from models import load_yolo, depth_anything_v2_metric, visualize_depth


class RGBDImageReciever:
    def __init__(self):
        rospy.loginfo("Initializing RGBD Image Reciever...")
        self.bridge = CvBridge()
        self.latest_frame_color = None
        self.latest_frame_depth = None

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.K = None

        self.rgb_subscriber = message_filters.Subscriber(RGB_TOPIC, Image, queue_size=10) # color image
        self.depth_subscriber = message_filters.Subscriber(DEPTH_TOPIC, Image, queue_size=10) # aligned depth to color
        self.cam_info_subscriber = rospy.Subscriber(CAMERA_INFO_TOPIC, 
                                                    CameraInfo, 
                                                    self.camera_info_callback, 
                                                    queue_size=1) # for intrinsics

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_subscriber, self.depth_subscriber], # synchronize rgb and depth
                                                              queue_size=10, slop=0.1)
        self.ts.registerCallback(self.rgbd_callback) # register the callback

        rospy.loginfo("RGBD Image Reciever initialized, subscribed to topics: %s , %s", RGB_TOPIC, DEPTH_TOPIC)
        rospy.loginfo("Camera Info topic: %s", CAMERA_INFO_TOPIC)

    def camera_info_callback(self, msg):
        # camera intrinsics (static parameters)
        # [fx  0 cx]
        # [0  fy cy]
        # [0  0  1]
        self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3) # covert the msg list to 3x3 numpy array

        self.fx = self.K[0, 0]  # effective focal length in x
        self.fy = self.K[1, 1]  # effective focal length in y
        self.cx = self.K[0, 2]  # principal point x
        self.cy = self.K[1, 2]  # principal point y

        rospy.loginfo_once("Camera intrinsics received: fx=%f, fy=%f, cx=%f, cy=%f", self.fx, self.fy, self.cx, self.cy)

    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            # convert ROS Image messages to OpenCV images
            cv_color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # if cv_depth_image.dtype == np.uint16: # depth in 'mm'
            #     cv_depth_image = cv_depth_image.astype(np.float32) / 1000.0
            # else: # depth in 'm'
            #     cv_depth_image = cv_depth_image.astype(np.float32)

            # always save depth in 'mm' as uint16, as it is standard for depth images
            if cv_depth_image.dtype == np.float32 or cv_depth_image.dtype == np.float64:
                cv_depth_image = (cv_depth_image * 1000).astype(np.uint16) # convert to mm and uint16

            rospy.loginfo_once("Received RGBD images of size: %s , %s", cv_color_image.shape, cv_depth_image.shape)

            # store the latest frames (will be displayed in main thread)
            self.latest_frame_color = cv_color_image
            self.latest_frame_depth = cv_depth_image

        except Exception as e:
            rospy.logerr("Error converting RGBD images: %s", e)

class DoorDetector:
    def __init__(self):
        rospy.loginfo("Initializing Door Detector Node...")

        # model params
        self.model_path = MODEL_PATH
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.img_size = IMG_SIZE  # input image size for the model

        # loding the model
        self.model_bbox = load_yolo(self.model_path)  # bbox detection model
        self.model_depth = depth_anything_v2_metric()  # depth estimation model
        rospy.loginfo("Loaded YOLO model from %s", self.model_path)

        self.door_detection_pub = rospy.Publisher(DOOR_DETECTION_TOPIC, Detection2DArray, queue_size=10) # publish detections
        rospy.loginfo("Door Detector Node initialized, publishing to %s", DOOR_DETECTION_TOPIC)

    def detect(self, model_name, color_image, confidence_threshold, img_size, verbose=False, stream=True):
        
        try:
            results = None
            # run inference
            if model_name in YOLO_DETECTION_MODELS: # only for yolo models for bbox 
                # all yolo models have similar interface
                results = self.model.predict(source=color_image, 
                                             conf=confidence_threshold,
                                             size=img_size, 
                                             verbose=verbose, 
                                             stream=stream)
            elif model_name == "grounding_dino_base": # grounding dino base for box detection (not using huggingface directly), but uses hf internally
                # TODO: need to implement grounding dino base model loading and inference here
                # from groundingdino.util.inference import load_image, predict, annotate
                # TEXT_PROMPT = "door ."
                # BOX_TRESHOLD = 0.80
                # TEXT_TRESHOLD = 0.80

                # image_source, image = load_image(color_image)

                # boxes, logits, phrases = predict(
                #     model=self.model,
                #     image=image,
                #     caption=TEXT_PROMPT,
                #     box_threshold=BOX_TRESHOLD,
                #     text_threshold=TEXT_TRESHOLD,
                #     device="cuda" if torch.cuda.is_available() else "cpu"
                # )

                # # reprocess to make it similar to yolo results
                # results = []
                # class_ids = [0] * len(boxes)  # assuming all are doors with class id 0
                # confidences = logits.tolist()
                # bboxes = boxes.tolist()  # list of [x1, y1, x2, y2]
                # result_dict = {
                #     'boxes': []

                # }
                # results.append(result_dict)
                # return results
                pass
         
            elif model_name == "grounding_dino_hf": # grounding dino base from direct huggingface for bbox
                # TODO: need to implement grounding dino huggingface model inference here
                # text = "a door."

                # inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                # with torch.no_grad():
                #     outputs = model(**inputs)

                # results = processor.post_process_grounded_object_detection(
                #     outputs,
                #     inputs.input_ids,
                #     box_threshold=0.4,
                #     text_threshold=0.3,
                #     target_sizes=[image.size[::-1]]
                # )
                # return results
                pass
            
            else:
                return results
        
        except Exception as e:
            rospy.logerr("Error in detect(): %s", e)
            return None

        
        return results
    
    def estimate_depth(self, rgb_image):
        try:
            
            # depth estimation
            if DEPTH_MODEL in DEPTH_ESTIMATION_MODELS: # depth anything v2
                # run depth estimation
                depth = self.model_depth.infer_image(rgb_image) # HxW raw depth map

                # depth values from DepthAnything
                print(f"\nDepth Anything V2-metric statistics:")
                print(f"  Shape: {depth.shape}")
                print(f"  Dtype: {depth.dtype}")
                print(f"  Min: {np.min(depth):.6f}m")
                print(f"  Max: {np.max(depth):.6f}m")
                print(f"  Mean: {np.mean(depth):.6f}m")
                print(f"  Median: {np.median(depth):.6f}m")

                visualize_depth(depth) # visualize the depth map

        except Exception as e:
            rospy.logerr("Error in estimate_depth(): %s", e)
            return None
        
    def compare_depths(self, depth_anything_image, realsense_depth):
        try:

            # Both inputs are HxW numpy arrays in meters
            if depth_anything_image.shape != realsense_depth.shape:
                print("Depth map shapes do not match for comparison.")
                return

            # Create mask for valid RealSense depths (ignore zeros)
            valid_mask = realsense_depth > 0.1  # Ignore very small/zero depths
            
            if np.sum(valid_mask) == 0:
                print("No valid RealSense depth values to compare.")
                return
            
            # Compute difference only on valid pixels
            difference = depth_anything_image[valid_mask] - realsense_depth[valid_mask]
            abs_difference = np.abs(difference)

            # Compute statistics
            mean_diff = np.mean(difference)
            mean_abs_diff = np.mean(abs_difference)
            std_diff = np.std(difference)
            median_diff = np.median(difference)
            max_diff = np.max(abs_difference)

            print(f"\n{'='*60}")
            print(f"Depth Comparison: Depth Anything V2 vs RealSense")
            print(f"{'='*60}")
            print(f"Valid pixels compared: {np.sum(valid_mask):,} / {valid_mask.size:,}")
            print(f"Mean Difference: {mean_diff:.3f}m")
            print(f"Mean Absolute Error: {mean_abs_diff:.3f}m")
            print(f"Std Deviation: {std_diff:.3f}m")
            print(f"Median Difference: {median_diff:.3f}m")
            print(f"Max Absolute Error: {max_diff:.3f}m")
            print(f"{'='*60}\n")

            # Create full-size difference map
            diff_map = np.zeros_like(depth_anything_image)
            diff_map[valid_mask] = np.abs(depth_anything_image[valid_mask] - realsense_depth[valid_mask])
            
            # Visualize differences
            diff_norm = (diff_map - np.min(diff_map)) / (np.max(diff_map) - np.min(diff_map) + 1e-6)
            diff_uint8 = (diff_norm * 255).astype(np.uint8)
            diff_colormap = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_HOT)
            
            # Add text overlay
            cv2.putText(diff_colormap, f"Max Error: {max_diff:.2f}m, MAE: {mean_abs_diff:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Visualize RealSense depth
            rs_display = realsense_depth.copy()
            rs_min = np.min(rs_display[rs_display > 0])
            rs_max = np.max(rs_display)
            rs_normalized = (rs_display - rs_min) / (rs_max - rs_min) if rs_max > rs_min else np.zeros_like(rs_display)
            rs_uint8 = (rs_normalized * 255).astype(np.uint8)
            rs_colormap = cv2.applyColorMap(rs_uint8, cv2.COLORMAP_JET)
            
            cv2.putText(rs_colormap, f"RealSense Depth: {rs_min:.2f}m - {rs_max:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RealSense Depth", rs_colormap)
            cv2.imshow("Absolute Difference (Hot=High Error)", diff_colormap)

        except Exception as e:
            rospy.logerr("Error in compare_depths(): %s", e)
            return None

def detect_door_in_image():

    rospy.init_node('door_detector_node')
    rgbd_reciever = RGBDImageReciever() # subscribe to RGBD images
    door_detector = DoorDetector() # door detector

    frame_count = 0
    detection_interval = 5  # run detection every 5 frames to save computation
    
    # display loop in main thread 
    rate = rospy.Rate(30)  # 30hz display rate
    while not rospy.is_shutdown(): # while the node is running
        if rgbd_reciever.latest_frame_color is not None and rgbd_reciever.latest_frame_depth is not None:
            # get frames
            color_image = rgbd_reciever.latest_frame_color
            depth_image = rgbd_reciever.latest_frame_depth

            latest_detections = []  # to store latest detections for visualization

            # run detection at intervals
            if frame_count % detection_interval == 0:
                # ------ get  detection results -----
                # run inference on choosen model, get results
                results = door_detector.detect(
                    model_name=MODEL_BBOX, 
                    color_image=color_image, 
                    confidence_threshold=door_detector.confidence_threshold, 
                    img_size=door_detector.img_size,
                    verbose=False,
                    stream=True
                )

                detections_msg = Detection2DArray() # array of Detection2D with fields header and detections
                # detection header (has timestamp and frame_id)
                detections_msg.header.stamp = rospy.Time.now()
                detections_msg.header.frame_id = "camera_color_optical_frame"  # assuming this frame

                # ------ process the results -------
                # detection body has list of Detection2D (each with bbox and objposehypothesis, each bbox has center x,y and size x,y)
                # get detections from each image after detection
                for result in results:
                    # result structure (from ultralytics)
                    """
                    results is a generator that yields results for each image
                        results = [result1, result2, ...]

                        result = {  'boxes': [box1, box2, ...], # for bbox outputs
                                    'masks': ..., # for segmentation outputs
                                    'keypoints': ..., # for pose outputs
                                    'probs': ... # for classification outputs
                                    } 

                        box = { 'cls': class_id, # property for class id >> tensor 
                                'conf': confidence, # property for confidence score >> tensor
                                'xyxy': [x1, y1, x2, y2] # property for bbox coordinates >> tensor
                                'xywh': [x, y, w, h] # property for bbox center x,y and width,height >> tensor
                                'id': index  # property for index of the box (if tracking is used and available) >> tensor
                                }

                    """
                    for box in result.boxes:
                        cls_id = int(box.cls[0]) # tensor to int
                        conf = float(box.conf[0])


                        # only keep door detections (assuming door class id is 0) and if higher than threshold
                        if cls_id == 0 and conf >= door_detector.confidence_threshold:
                            # extract bbox coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            bbox_center_x = x1 + bbox_width / 2.0
                            bbox_center_y = y1 + bbox_height / 2.0

                            # create Detection2D message for ROS
                            detection = Detection2D()
                            detection.bbox.center.x = bbox_center_x
                            detection.bbox.center.y = bbox_center_y
                            detection.bbox.size_x = bbox_width
                            detection.bbox.size_y = bbox_height
                            # detection.source_img # no need to fill this image again, because we are publishing all detections in one image

                            hypothesis = ObjectHypothesisWithPose() # it has id and score, and 6D pose (we won't use pose here)
                            hypothesis.id = cls_id
                            hypothesis.score = conf
                            detection.results.append(hypothesis)

                            detections_msg.detections.append(detection)

                            latest_detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'conf': conf, 'cls_id': cls_id
                            })

                # ------ publish detections -------
                door_detector.door_detection_pub.publish(detections_msg)
                rospy.loginfo("Published %d door detections", len(detections_msg.detections))
            
            # draw bboxes on the frame (using latest detections)
            for det in latest_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Door {conf:.2f}"
                cv2.putText(color_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # display the color image with detections
            cv2.imshow("Door Detection", color_image)
            
            # create depth visualization
            depth_vis = (depth_image * 255).astype(np.uint8)  # scale to 0-255
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Image", depth_vis)
            
            frame_count += 1
            
            # display the color image
            # cv2.imshow("Color Image", rgbd_reciever.latest_frame_color)
            
            # create depth visualization
            # depth_vis = np.clip(rgbd_reciever.latest_frame_depth, 0, 2) / 2.0  # clip depth for visualization
            # depth_vis = (rgbd_reciever.latest_frame_depth * 255).astype(np.uint8)  # scale to 0-255
            # depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # cv2.imshow("Depth Image", depth_vis)
        
        # handle cv2 events and check for ESC key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            rospy.loginfo("ESC pressed, shutting down...")
            break
        
        rate.sleep()
    
    # cleanup
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Node shutdown")

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

def main():
    # detect_door_in_image()
    recieve_rgbd_images()
    

if __name__ == '__main__':
    main()