RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
RGB_ROS_MSG_TYPE = 'sensor_msgs/Image'
DEPTH_ROS_MSG_TYPE = 'sensor_msgs/Image'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'

# detector parameters
LABEL_MAP = {0: 'door', 1: 'handle'}
MODEL_PATH = '/home/satya/MT/catkin_ws/src/door_navigation/weights/last_yolo11m_ias12.pt'  # path to door detection model # last_yolo8m.pt
DETECTION_JSON_PATH = '/home/satya/MT/catkin_ws/src/door_navigation/scripts/door_detections.json'  # path to save detection results
CONFIDENCE_THRESHOLD = 0.5
IMG_SIZE = 640  # input image size for the model
DOOR_DETECTION_TOPIC = "/door_detections"  # assuming door class id is 0 in the model
YOLO_DETECTION_MODELS = ["yolo_11m", "yolo_v8l", "yolo_v8m", "yolo_v5l"]  # model name for detection
DEPTH_ESTIMATION_MODELS = ["depth_anything_v2"]  # model name for depth estimation
MODEL_BBOX = "yolo_v5l"  # choose model for detection
DEPTH_MODEL = "depth_anything_v2"  # choose model for depth estimation

# temporary
IMAGE_PATH = ""

# CAMERA INTRINSICS (aligned depth to color), units in pixels
FX = 385.88861083984375
FY = 385.3906555175781
CX = 317.80999755859375
CY = 243.65032958984375

# camera parameters
# rostopic echo -n 1 /camera/aligned_depth_to_color/camera_info
# header: 
#   seq: 3
#   stamp: 
#     secs: 1766760902
#     nsecs: 532018423
#   frame_id: "camera_color_optical_frame"
# height: 480
# width: 640
# distortion_model: "plumb_bob"
# D: [-0.054627396166324615, 0.06297337263822556, -0.0005539487465284765, 0.00043509574607014656, -0.019929416477680206]
# K: [385.88861083984375, 0.0, 317.80999755859375, 0.0, 385.3906555175781, 243.65032958984375, 0.0, 0.0, 1.0]
# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P: [385.88861083984375, 0.0, 317.80999755859375, 0.0, 0.0, 385.3906555175781, 243.65032958984375, 0.0, 0.0, 0.0, 1.0, 0.0]
# binning_x: 0
# binning_y: 0
# roi: 
#   x_offset: 0
#   y_offset: 0
#   height: 0
#   width: 0
#   do_rectify: False
# ---

