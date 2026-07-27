### 0. Make workspace

- make directory

```bash
mkdir -p ~/MT/catkin_ws/src
cd ~/MT/catkin_ws/src
```

- git clone repo (inside src)

```bash
git clone git@github.com:Satya1998-debug/door_navigation.git
```



### 1. Python venv (Jetson)

> On the Jetson we use JetPack's system Python 3.8 + system CUDA. See [`setup_issues_jetson_env.md`](setup_issues_jetson_env.md) for the underlying reasoning.
>
> **Do NOT use `uv`, `conda`, or `pipx`.** They isolate the venv away from the system-CUDA / JetPack PyTorch stack, so `torch.cuda.is_available()` will silently return `False`. The venv here is an **overlay**, not a sandbox.

- install the `venv` module for Python 3.8

```bash
sudo apt install python3.8-venv
```

- create the venv **outside** the catkin workspace, with system site-packages inherited (so ROS's `python3-*` packages and the JetPack PyTorch wheel remain visible)

```bash
python3 -m venv ~/MT/venv38 --system-site-packages
```

- source it in `~/.bashrc` so every terminal picks it up

```bash
echo 'source ~/MT/venv38/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

- verify the venv is active and points at Python 3.8

```bash
which python        # -> ~/MT/venv38/bin/python
python --version    # -> Python 3.8.x
```

- upgrade `pip` inside the venv (warnings about system packages are expected and safe)

```bash
python -m pip install --upgrade pip
```

- PyTorch is a **platform dependency**, not a Python package. Never run `pip install torch`. Install the NVIDIA wheel matching your JetPack:

```bash
cat /etc/nv_tegra_release   # note the R3x.y.z / JetPack version
# then follow https://forums.developer.nvidia.com/t/pytorch-for-jetson/ for the matching wheel
```

- verify CUDA is reachable from the venv

```bash
python - <<EOF
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
```



### 2. CV Bridge for python 3.8 (must be done)

- clone only noetic branch of cv_bridge repo

```bash
git clone git@github.com:ros-perception/vision_opencv.git -b noetic 
```

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

- verify if cv_bridge is built for python 3.8

```bash
python - <<EOF
from cv_bridge import CvBridge
import cv2
print("cv_bridge OK")
EOF

```



### 3. Install dependencies for ROS OpenCV

```bash
sudo apt update
sudo apt install -y \
libopencv-dev \
ros-noetic-cv-bridge \
ros-noetic-image-transport \
ros-noetic-image-proc
```



### 4. Installation steps for required Dependencies (VLM, camera, etc.)

### realsense ros

- follow instructions from [https://github.com/realsenseai/realsense-ros/tree/ros1-legacy?tab=readme-ov-file](https://github.com/realsenseai/realsense-ros/tree/ros1-legacy?tab=readme-ov-file)
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
- follow this link: [https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages)
- to launch the realsense viewer

```bash
realsense-viewer # verify
```


### some packages to be install for vision related tasks

- install this for 2D bbox msgs

```bash
sudo apt-get install ros-noetic-vision-msgs
```



### Installation of other useful packages for Go1 Navigation (not required on Jetson. skip)

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

> All `pip install` commands below run **inside the Py 3.8 venv from Section 1** (`~/MT/venv38`). Do **not** substitute `uv pip install` — see [`setup_issues_jetson_env.md`](setup_issues_jetson_env.md).

- create a directory to store all transformer models

```bash
mkdir ~/door_navigation/src/door_navigation/py_packages/
cd ~/door_navigation/src/door_navigation/py_packages/
```

- clone all repositories in `py_packages/` and install their requirements inside the venv
- Download the models as per instructions in respective repos (usually `checkpoints/` or `weights/` folders)



#### DepthAnythingV2

- clone repo and install requirements inside the venv

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
pip install -r Depth-Anything-V2/requirements.txt
```

- rename the parent dir after cloning to `depth_anything_v2` for easy imports
- download the model weights as per instructions in the repo (store in `door_navigation/checkpoints/`)

> Before installing, strip any `nvidia-*` packages from `requirements.txt` and pin `tokenizers==0.15.2`, `huggingface-hub==0.24.7`. Also `export HF_HUB_DISABLE_XET=1`. See [`setup_issues_jetson_env.md`](setup_issues_jetson_env.md) for why.



#### Yolo via Ultralytics

- install `ultralytics` inside the venv

```bash
pip install ultralytics
```

- keep the weights in `door_navigation/weights/`



#### VLMs

- pull the models via ollama commands (system-wide)
- install ollama python client inside the venv

```bash
pip install ollama
```



### Latest CATKIN MAKE

- after all installations and code changes, build the catkin workspace again to reflect the changes

```bash
export SETUPTOOLS_USE_DISTUTILS=stdlib
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```


### 5. Build workspace

- after all installations are done, build the catkin workspace (catkin_make is mostly used here and is straightforward)

```bash
cd ~/MT/catkin_ws
catkin_make
```

- install catkin tools for python package management

```bash
sudo apt install python3-catkin-tools
```

- configure the workspace but compiles(source code to machine code) only the specified package

```bash
catkin_make --pkg door_navigation
```

NOTE: Donot mix catkin build with catkin_make in the same workspace, it may lead to build errors. Use only one of them. They have different build systems.


### Run ROS Bridge server

- install rosbridge (system-wide, **not** inside the venv)

```bash
sudo apt-get install ros-noetic-rosbridge-suite
sudo apt install python3-tornado python3-twisted
```

- to enable communication between ROS and langchain agents, run the rosbridge server

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

> If `roslaunch` throws an `attr.s` / `AttributeError` from `twisted`, deactivate the venv and reinstall the system `python3-attr`/`python3-twisted`. Full fix in [`setup_issues_jetson_env.md`](setup_issues_jetson_env.md#ros-bridge-issue).



# Deployment of Checkpoint Models on Jetson

- run export script to convert the PyTorch model to ONNX format (run this on x86_64 machine with GPU for faster export)

```bash
python3 scripts/tensorrt_export.py
```

- use tensorrt trtexec tool to convert the ONNX model to TensorRT engine format (run this on Jetson ARM64 platform)

```bash
/usr/src/tensorrt/bin/trtexec   --onnx=/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_vits.onnx   --saveEngine=/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_vits.engine   --fp16   --workspace=4096   --timingCacheFile=trt_cache.bin   --verbose
```

