import os
import rospkg

# path setup
try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    # Fallback: utils/config.py -> scripts/utils -> scripts -> door_navigation
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    print(f"[CONFIG] rospkg not available, using relative path: {PACKAGE_PATH}")

# ROS topics (image)
RGB_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw' # depth not used so far
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
CAM_OPTICAL_FRAME = "camera_color_optical_frame"
MAP_FRAME = "map"

# ROS topics (detection)
DOOR_DETECTION_TOPIC = "/door_detections"
DOOR_POSE_TOPIC = "/door_poses"

USE_DEPTH_ANYTHING = True  # set to False to disable depth anything and use raw depth instead (for testing)
USE_VLM = True  # set to False to disable VLM and use only geometric reasoning for state estimation (for testing)

# DOOR navigation parameters
TEB_GLOBAL_PLAN_TOPIC = "/move_base/TebLocalPlannerROS/global_plan"
POST_DOOR_DISTANCE = 2.0  # meters after door (only used when USE_POST_DOOR_POSE=True)
PRE_DOOR_DISTANCE = 2.5   # meters before door
DOOR_TRIGGER_DISTANCE = 6.0  # start door logic when closer than this
# when True, coordinator commands a short "just past the door" checkpoint
# before resuming the original goal. Gives cleaner failure isolation and a
# straighter TEB path through tight door frames, at the cost of a brief stop.
# when False, coordinator sends the saved original goal directly at the "door is passable" branch and treats reaching it as full door traversal.
USE_POST_DOOR_POSE = True
# arc-length safety margin added on top of DOOR_TRIGGER_DISTANCE when scanning the global plan for door intersections. Lookahead = trigger + this margin.
LOOKAHEAD_SAFETY_MARGIN_M = 1.0  # meters
# hard upper bound on number of forward path segments to scan. Prevents
# pathological scans on very long plans (acts as a safety cap, not the limit).
LOOKAHEAD_POINTS = 500

# detector parameters
LABEL_MAP = {0: 'door_double', 1: 'door_single', 2: 'handle'}  # class id to name mapping
LABEL_DOORS = [0, 1]  # class ids for doors
MODEL_PATH = os.path.join(PACKAGE_PATH, 'weights', 'last_yolo11m_ias_door_type1.engine')  # path to door detection model # last_yolo8m.pt
DETECTION_JSON_PATH = os.path.join(PACKAGE_PATH, 'scripts', 'door_detections.json')  # path to save detection results
CONFIDENCE_THRESHOLD = 0.8
DETECTION_RATE = 5  # Hz (should be more than image publish rate to avoid missing frames)
IMG_SIZE = 640  # input image size for the model
IMG_DIM = (640, 480)  # original image dimensions (width, height)
DEPTH_ANYTHING_V2_PATH = os.path.join(PACKAGE_PATH, 'checkpoints', 'depth_anything_v2_metric_hypersim_vits.pth')  # path to depth anything v2 model
DEPTH_ANYTHING_V2_PATH_TRT = os.path.join(PACKAGE_PATH, 'checkpoints', 'depth_anything_v2_vits.engine')  # path to depth anything v2 model in onnx format for trt inference

# CAMERA INTRINSICS (aligned depth to color), units in pixels (hardcoded but not used, because intrinsics are fetched live from the camera info topic)
FX = 385.88861083984375
FY = 385.3906555175781
CX = 317.80999755859375
CY = 243.65032958984375

# speech recognition model
SPEECH_RECOGNITION_MODEL = "vosk-model-en-us-0.22" # "vosk-model-small-en-us-0.15" # "vosk-model-en-us-0.22"
SPEECH_RECOGNITION_MODEL_PATH = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/utils/speech_model/"
SPEECH_OUTPUT_DIR = "/home/ias/satya/catkin_ws/src/door_navigation/scripts/output/"
VOSK_ENABLE_LOGS = False
QUIET_ALSA_WARNINGS = True

# SPEAKER and MIC device indices (NOTE: needs to be verified every time the system is booted, and then modified here if required, use get_audio_devices.py to get the indices)
SPEAKER_DEVICE_INDEX = 25 # default output device (after pavucontrol volume setup)
MIC_DEVICE_INDEX = 0

# human confirmation behavior in door coordinator
USE_VOICE_ASSISTANT = True  # when False, _speak() will log instead of speaking, and voice assistant is not initialized
USE_VOICE_CONFIRMATION = True
VOICE_CONFIRMATION_TIMEOUT_SEC = 7.0 # seconds
VOICE_CONFIRMATION_MAX_TRIES = 2 # max tries for voice confirmation
HUMAN_CONFIRMATION_COOLDOWN_SEC = 8.0 # seconds
# give up on the door after this many unsuccessful confirmation cycles so the
# coordinator can never get stuck indefinitely at a closed door.
MAX_HUMAN_CONFIRMATION_CYCLES = 6
