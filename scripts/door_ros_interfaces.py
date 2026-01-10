# this contains all ROS subscribers and publishers for door navigation

import sys
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import os
import glob
import torch
import cv2
import message_filters
import numpy as np
import rospkg
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


# ------ path setup -----
# Get package path using rospkg (works with rosrun)
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Add py_packages to path so we can import depth_anything_v2, etc.
py_packages_path = os.path.join(PACKAGE_PATH, 'src/door_navigation/py_packages')
if py_packages_path not in sys.path:
    sys.path.insert(0, py_packages_path)

depth_anything_v2_path = os.path.join(py_packages_path, 'depth_anything_v2')
if depth_anything_v2_path not in sys.path:
    sys.path.insert(0, depth_anything_v2_path)

from utils.config import *
from utils.depth_calibration import COEF_QUAD, apply_inverse_depth_correction


class RGBDImageReciever:
    # Subscriber class for synchronized RGB and Depth images
    # latest RGB & Depth images are stored in self.latest_frame_color and self.latest_frame_depth
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

        rospy.loginfo_throttle(5.0, "Camera intrinsics received: fx=%f, fy=%f, cx=%f, cy=%f", self.fx, self.fy, self.cx, self.cy)

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

            rospy.loginfo_throttle(5.0, "Received RGBD images of size: %s , %s", cv_color_image.shape, cv_depth_image.shape) # log every 5 seconds

            # store the latest frames (will be displayed in main thread)
            self.latest_frame_color = cv_color_image
            self.latest_frame_depth = cv_depth_image

        except Exception as e:
            rospy.logerr("Error converting RGBD images: %s", e)


class DoorDetector:
    # detects Bounding boxes of doors in RGB images
    # estimates the Depth map from RGB images
    # returns door bounding boxes and corrected depth map (from calibrated model)
    def __init__(self):
        rospy.loginfo("Initializing Door Detector Node...")
        # model params
        self.model_path = MODEL_PATH
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.img_size = IMG_SIZE  # input image size for the model

    def run_yolo_model(self, model_path=MODEL_PATH, 
                       rgb_image=None, 
                       img_size=IMG_SIZE, 
                       confidence_threshold=CONFIDENCE_THRESHOLD,
                       visualize=True):
        try:
            from ultralytics import YOLO

            if rgb_image is None:
                print("No RGB image provided for YOLO model inference.")
                return

            # load the model
            model = YOLO(model_path)

            valid_boxes = []
            jsonable_valid_boxes = []

            detections = dict()

            # run inference
            results = model(source=rgb_image, imgsz=img_size, conf=confidence_threshold)

            # print results
            for result in results:
                print(f"got {len(result.boxes)} boxes in test image")
                for i, box in enumerate(result.boxes):
                    print("for box-{}:".format(i))
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # get bbox coordinates
                    print(f"Class ID: {cls_id}, Confidence: {conf:.2f}, BBox: {bbox}")

                    # filter boxes above confidence threshold
                    if conf >= confidence_threshold:
                        print(f"Detected door with confidence {conf:.2f} at bbox {bbox}")
                        valid_boxes.append({
                            'cls_id': cls_id,  # 0 for door, 1 for handle
                            'conf': conf,
                            'bbox': bbox
                        })
                        jsonable_valid_boxes.append({
                            'cls_id': cls_id,  # 0 for door, 1 for handle
                            'conf': conf,
                            'bbox': bbox.tolist()  # convert numpy array to list for json serialization
                        })
                detections.update({
                    'valid_detections': jsonable_valid_boxes
                })

            # write detections to json file
            with open(os.path.join(DETECTION_JSON_PATH), 'w') as f:
                import json
                json.dump(detections, f, indent=4)

            if visualize:
                color_image = rgb_image.copy()
                for vb in valid_boxes:
                    x1, y1, x2, y2 = map(int, vb['bbox'])
                    conf = vb['conf']
                    cls_id = vb['cls_id']
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{LABEL_MAP.get(cls_id, 'Unknown')} {conf:.2f}"
                    cv2.putText(color_image, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                try:
                    cv2.imshow("Test Image Detections", color_image)
                    # handle cv2 events and check for ESC key to exit
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except Exception as viz_error:
                    print(f"Visualization skipped (no GUI support): {viz_error}")

            return valid_boxes

        except Exception as e:
            print(f"Error in run_yolo_model: {e}")
            return np.array([])
   
    def run_depth_anything_v2_on_image(self, img_dir=None, rgb_image=None):
        # img_dir = caliberation_dataset
        try:
            from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
            
            # Disable xformers for CPU inference (xformers requires CUDA)
            import depth_anything_v2.metric_depth.depth_anything_v2.dinov2_layers.attention as attn_module
            if not torch.cuda.is_available():
                attn_module.XFORMERS_AVAILABLE = False

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
            }

            encoder = 'vits' # or 'vits', 'vitb'
            dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
            max_depth = 20 # 20 for indoor model, 80 for outdoor model

            model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
            checkpoint_path = os.path.join(PACKAGE_PATH, f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth')
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
            model.to(device)
            model.eval()
            print(f"DepthAnythingV2 model loaded with checkpoint: {checkpoint_path} on {device}")

            if rgb_image is not None: # RGB image CV2 BGR format
                print(f"Processing the provided RGB image")
                depth = model.infer_image(rgb_image) # HxW depth map in meters in numpy
                # Save depth map as .npy file
                depth = depth.astype(np.float32) # depth in meters
                print(f"Depth map processing completed.")
                return depth

            if img_dir is not None:  # only used during testing/calibration
                for image_path in sorted(glob.glob(os.path.join(img_dir, "color", '*.jpg'))):
                    print(f"Processing image: {image_path}")

                    depth_img_save_path = image_path.replace("color", "depth_da").replace(".jpg", ".npy").replace("/color/", "/depth_da/")

                    raw_img = cv2.imread(image_path)

                    depth = model.infer_image(raw_img) # HxW depth map in meters in numpy
                    # Save depth map as .npy file
                    depth = depth.astype(np.float32) # depth in meters
                    np.save(depth_img_save_path, depth)
                    print(f"Depth map saved to: {depth_img_save_path}")
                    return None
        except Exception as e:
            print(f"Error in run_depth_anything_v2_on_image: {e}")
            return None
        
    def get_corrected_depth_image(self, coef=COEF_QUAD, cal_dir=None, model="quad", is_test=False, depth_da=None):

        if is_test: # only if testing
            actual_depth_da_dir = os.path.join(cal_dir, "depth_da")  # example path, adjust as needed

            os.makedirs(os.path.join(cal_dir, f"depth_da_correct_{model}"), exist_ok=True)

            for depth_img_path in sorted(glob.glob(os.path.join(actual_depth_da_dir, '*.npy'))):
                print(f"Applying correction to DepthAnything depth image: {depth_img_path}")
                depth_da = np.load(depth_img_path).astype(np.float32)  # depth in meters
                depth_corrected = apply_inverse_depth_correction(depth_da, coef, model=model)
                depth_corrected_save_path = depth_img_path.replace("/depth_da", f"/depth_da_correct_{model}")
                np.save(depth_corrected_save_path, depth_corrected)
                print(f"Corrected depth map saved to: {depth_corrected_save_path}")
                return
        else: # in production
            depth_corrected = apply_inverse_depth_correction(depth_da, coef, model=model)
            return depth_corrected


class DoorDetectionPublisher:
    # publisher class for door detection results (when required)
    def __init__(self):
        self.publisher = rospy.Publisher(DOOR_DETECTION_TOPIC, Detection2DArray, queue_size=10)
        rospy.loginfo("Door Detection Publisher initialized, publishing to topic: %s", DOOR_DETECTION_TOPIC)

    def publish_detections(self, detections_msg):
        self.publisher.publish(detections_msg)
        rospy.loginfo("Published %d door detections", len(detections_msg.detections))

    def create_detections_msg(self, valid_boxes):
        detections_msg = Detection2DArray() # array of Detection2D with fields header and detections
        # detection header (has timestamp and frame_id)
        detections_msg.header.stamp = rospy.Time.now()
        detections_msg.header.frame_id = "camera_color_optical_frame"  # assuming this frame

        for vb in valid_boxes:
            detection = Detection2D()
            detection.bbox.center.x = (vb['bbox'][0] + vb['bbox'][2]) / 2.0
            detection.bbox.center.y = (vb['bbox'][1] + vb['bbox'][3]) / 2.0
            detection.bbox.size_x = vb['bbox'][2] - vb['bbox'][0]
            detection.bbox.size_y = vb['bbox'][3] - vb['bbox'][1]

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = vb['cls_id']
            hypothesis.score = vb['conf']

            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)

        return detections_msg
