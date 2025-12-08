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

### build cv bridge for python 3.10
- build cv_bridge for python 3.10 explicitly [IMPORTANT]
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

### inatllation steps for realsense ros wrapper
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
roslaunch realsense2_camera rs_camera.launch initial_reset:=true depth_width:=848 depth_height:=480 depth_fps:=30 color_width:=848 color_height:=480 color_fps:=30 align_depth:=true enable_sync:=true
```