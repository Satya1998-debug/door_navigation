### uv virtual env setup instructions

- create uv venv outside the catkin_ws directory
- source uv venv in bashrc
- install uv (assumed that ROS1 noetic is alreday installed)
- create venv 
```bash
uv venv --python 3.10 ~/MT/uv_ros_py10
```
- install uv dependencies (uv pip install -r requirements.txt)
- check which python is being used (which python should point to ~/MT/uv_ros_py10/bin/python)


### build cv bridge for python 3.8 (must be done)
- clone only noetic branch of cv_bridge repo
```bash
git clone git@github.com:ros-perception/vision_opencv.git -b noetic 
```
- keep only cv_bridge folder and delete others from vision_opencv
- build cv_bridge for python 3.8 explicitly [IMPORTANT]
- clone the cv_bridge repo into src folder of catkin_ws (but only noetic branch)
- build cv_bridge with the following command from catkin_ws directory
```bash
catkin_make   -DPYTHON_EXECUTABLE=$(which python)   -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")   -DPYTHON_LIBRARY=$(python - <<EOF
import sysconfig; import pathlib;
print(next(pathlib.Path(sysconfig.get_config_var("LIBDIR")).glob("libpython*.so")))
EOF
)
```
- verify if cv_bridge is built for python 3.10
```bash
python - <<EOF
from cv_bridge import CvBridge
import cv2
print("cv_bridge OK")
EOF

```

### install deps for ROS OpenCV
```bash
sudo apt update
sudo apt install -y \
libopencv-dev \
ros-noetic-cv-bridge \
ros-noetic-image-transport \
ros-noetic-image-proc
```

### installation steps for realsense ros wrapper

- follow instructions from https://github.com/realsenseai/realsense-ros/tree/ros1-legacy?tab=readme-ov-file
- connect camera and ready to go
```bash
roscore
roslaunch realsense2_camera rs_camera.launch
roslaunch realsense2_camera rs_camera.launch initial_reset:=true align_depth:=true enable_sync:=true
```
- we need faster processing for depth images, so we will use the pointcloud to depth image conversion node from realsense ros wrapper
- we set the depth and rgb image resolution to 640x480 with 30 fps for robotics, for faster processing

```bash
roslaunch realsense2_camera rs_camera.launch initial_reset:=true depth_width:=640 depth_height:=480 depth_fps:=30 color_width:=640 color_height:=480 color_fps:=30 align_depth:=true
```

- after this librealsense sdk needs to be install to check the realsense camera status
- follow this link: https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages
- to launch the realsense viewer
```bash
realsense-viewer
```

### LangGraph and Agentic 
- all these high-level things will be in a separate venv with python 3.10
- need to use "roslibpy" package to interface with ROS from langchain agents
- install roslibpy in the langchain venv
```bash
uv pip install roslibpy
```

### Interface connect with 3.10 and 3.8 venvs (ros_bridge + roslibpy)
- need to use ros_bridge on ROS side to connect with langchain agentic system running on python 3.8 venv
- install rosbridge server package
```bash
sudo apt-get install ros-noetic-rosbridge-suite
```
- launch rosbridge server (always try to use global system-wide ROS python), rosbridge automatically uses python from venv if activated
```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```
- follow this for tutorial:https://wiki.ros.org/rosbridge_suite/Tutorials/RunningRosbridge

- need to install missing packages for ros noetic python 3.8 venv:
```bash
sudo apt install python3-tornado python3-twisted
```


### Build workspace
- after all installations are done, build the catkin workspace (catkin_make is mostly used here and is straightforward)
```bash
cd ~/MT/catkin_ws
catkin_make
```
- install catkin tools for python package management
```bash
sudo apt install python3-catkin-tools
```
- for selective build use: (this performs better than catkin_make for selective builds)
```bash
catkin build door_navigation
```

- configure the workspace but compiles(source code to machine code) only the specified package
```bash
catkin_make --pkg door_navigation
```

NOTE: Donot mix catkin build with catkin_make in the same workspace, it may lead to build errors. Use only one of them. They have different build systems.


### some packages to be install for vision related tasks
- install this for 2D bbox msgs
```bash
sudo apt-get install ros-noetic-vision-msgs
```


### Installation of other useful packages for Go1 Navigation
- When taking Ahmed's Go1 Navigation repo, some dependencies may be missing. Install them using the following commands:
```bash
sudo apt update
sudo apt-get install -y ros-noetic-move-base-msgs # for start pkg
sudo apt-get install -y ros-noetic-openslam-gmapping # for gmapping pkg
sudo add-apt-repository ppa:borglab/gtsam-release-4.0 # for lio_slam packages
sudo apt update  
sudo apt install libgtsam-dev libgtsam-unstable-dev
sudo apt-get install -y liblcm-dev # for a2_ros2udp pkg (Unitree Go1 SDK)
```

**Note:** Some packages are excluded from the x86_64 build and should be built on the Jetson ARM64 platform:
- `lio_sam` - has PCL 1.10 compatibility issues with newer C++ standards (CATKIN_IGNORE added)
- `a2_ros2udp` - requires ARM64 Unitree SDK libraries for Jetson (CATKIN_IGNORE added)


### Transformer models installation
- create a directory to store all transformer models
```bash
mkdir ~/door_navigation/src/door_navigation/py_packages/
cd ~/door_navigation/src/door_navigation/py_packages/
``` 
- clone all respositories in py_packages directory and install requirements in uv venv
- Download the models as per instructions in respective repos (usually checkpoints or weights folders)


#### Ollama
- install via terminal (this will be installed system-wide)
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
#### DepthAnythingV2
- clone repo and install requirements inside the uv venv
```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
uv pip install -r Depth-Anything-V2/requirements.txt
```
- rename the parent dir ater cloning to "depth_anything_v2" for easy imports
- download the model weights as per instructions in the repo (store in door_navigation/checkpoints/)

#### Yolo via Ultralytics
- install ultralytics package inside uv venv
```bash
uv pip install ultralytics
```
- keep the weights in door_navigation/weights/

#### VLMs
- pull the models via ollama commands inside uv venv
- install ollama python client inside uv venv
```bash
uv pip install ollama
```
