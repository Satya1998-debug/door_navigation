# Door Navigation — Useful Commands Cheatsheet

A single reference for the most frequently used commands while developing,
testing, and recording bags for the `door_navigation` pipeline.

Companion docs:

- [`README.md`](README.md) — setup & install instructions
- [`time_sync_setup.md`](time_sync_setup.md) — chrony/PTP setup
- [`internet_setup.md`](internet_setup.md) — networking notes
- [`unitree_go1_steps.md`](unitree_go1_steps.md) — Go1 robot startup notes

---

## 1. ROS environment

```bash
# Sourcing this workspace
source /home/ias/satya/catkin_ws/devel/setup.bash

# When working over the Go1 wireless network (Jetson on robot)
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_HOSTNAME=192.168.123.15

# Start the master
roscore
```

## 2. Build

```bash
cd ~/satya/catkin_ws

# Full build
export SETUPTOOLS_USE_DISTUTILS=stdlib
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Only the door_navigation package
catkin_make --pkg door_navigation
```

## 3. Camera (RealSense)

```bash
# Full launch with depth alignment (used by the door pipeline)
roslaunch realsense2_camera rs_camera.launch \
    initial_reset:=true \
    depth_width:=640 depth_height:=480 depth_fps:=30 \
    color_width:=640 color_height:=480 color_fps:=30 \
    align_depth:=true

# Quick sanity check
realsense-viewer
rostopic hz /camera/color/image_raw
rostopic hz /camera/aligned_depth_to_color/image_raw
rostopic echo -n1 /camera/color/camera_info
```

## 4. Localization, map, and navigation stack

```bash
# Move base + AMCL (TEB local planner, /goal remapped from move_base_simple/goal)
roslaunch navigation go1_move_base.launch

# Full nav (map_server + move_base)
roslaunch navigation go1_navigation.launch map_file:=/home/maps/gmapping/map_test.yaml

# Useful navigation topics
rostopic echo /move_base/status
rostopic echo /move_base/result
rostopic echo /move_base/feedback
rostopic echo /move_base/TebLocalPlannerROS/global_plan
rostopic echo /tf_static
```

### Set initial pose (without RViz)

```bash
# By raw coordinates
rosrun door_navigation set_initial_pose.py --x 7.77 --y 2.44 --yaw -1.08

# By a saved location
rosrun door_navigation set_initial_pose.py \
    --location home \
    --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml
```

### Save current pose to YAML

```bash
rosrun door_navigation save_location.py \
    --name kitchen_door \
    --output /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml
```

## 5. Goal sender (`goal_sender.py`)

Publishes a saved pose to `/goal` and waits-for-arrival logic for the
robot_command_bridge service.

```bash
# Send a single goal by name
rosrun door_navigation goal_sender.py \
    --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml \
    home

# List available locations in a YAML
rosrun door_navigation goal_sender.py \
    --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml \
    --list

# Interactive prompt (no positional target)
rosrun door_navigation goal_sender.py \
    --yaml /home/ias/satya/catkin_ws/src/door_navigation/saved_locations_iaslab1.yaml
```

Relevant topics:

```bash
rostopic echo /goal                         # PoseStamped goals to move_base
rostopic echo /goal_manager/current_target  # latched current target for coordinator
```

## 6. Door pipeline nodes

### Launch everything together

```bash
# Camera + detector/pose + state estimator service
roslaunch door_navigation door_navigation.launch

# Offline variant (no camera, expects images on the configured topics)
roslaunch door_navigation door_navigation_offline.launch
```

### Door detector + pose estimator

Node: `door_detect_pose_estimate_node.py`
Inputs: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/color/camera_info`
Outputs: `/door_poses` (`door_navigation/DoorPoseArray`), `/door_pose_markers` (`MarkerArray`)

```bash
rosrun door_navigation door_detect_pose_estimate_node.py

rostopic hz   /door_poses
rostopic echo /door_poses
rostopic echo /door_pose_markers
```

### Door state estimator (service)

Node: `door_state_estimator_node.py`
Service: `/door/estimate_state` (`door_navigation/EstimateDoorState`)

```bash
rosrun door_navigation door_state_estimator_node.py

# Call the service (request is empty, uses the latest camera images)
rosservice call /door/estimate_state "{}"

# Inspect
rosservice info /door/estimate_state
rossrv  show door_navigation/EstimateDoorState
```

### Door coordinator

Node: `door_coordinator_node.py`
Subscribes: `/door_poses`, `/move_base/TebLocalPlannerROS/global_plan`,
`/goal_manager/current_target`
Publishes: `/door_pose_markers`, `/door_coordinator/handling_door` (latched),
`/door_coordinator/failure_reason` (latched)
Uses move_base action server + `/door/estimate_state` service.

```bash
rosrun door_navigation door_coordinator_node.py

# for goal manager
rostopic echo /door_coordinator/handling_door
rostopic echo /door_coordinator/failure_reason

# debug topics for door intersection
rostopic echo /door_coordinator/door_on_path_reason
rostopic echo /door_coordinator/door_on_path

```

### Voice assistant (optional)

```bash
rosrun door_navigation voice_assistant.py
```

## 7. Robot command bridge (agent-facing services)

Node: `robot_command_bridge.py`
Services:
- `/agent/start_door_coordinator` (`std_srvs/Trigger`) — roslaunches the door
  pipeline
- `/agent/start_navigation` (`door_navigation/StartNavigation`) — sends a goal
  to `goal_sender` and blocks until arrival / failure / timeout

```bash
rosrun door_navigation robot_command_bridge.py _locations_yaml:=/home/ias/satya/catkin_ws/src/door_navigation/saved_locations_map_area_04.yaml

# Trigger door coordinator launch
rosservice call /agent/start_door_coordinator

# Drive to a saved location (room name takes priority over person)
rosservice call /agent/start_navigation "person: ''
room: 'home'"
```

Tip: in `robot_command_bridge.py` set `self.testing = False` to actually call
the goal manager instead of the simulated 5-second sleep.

## 8. ROS Bridge (LangChain / Python 3.10 agents ↔ ROS)

```bash
roslaunch rosbridge_server rosbridge_websocket.launch
```

## 9. Inspecting the graph

```bash
rosnode list
rosnode info /door_coordinator
rostopic list
rostopic info /door_poses
rqt_graph
rqt_tf_tree
rosrun tf2_tools view_frames.py     # produces frames.pdf in cwd
rosrun tf tf_echo map base_link
```

## 10. Recording bags

### Full coordinator test recording (RGB + depth + nav + door pipeline)

```bash
rosbag record --lz4 --split --size=2048 \
  -O door_coord_test_$(date +%Y%m%d_%H%M%S).bag \
  /door_poses \
  /move_base/TebLocalPlannerROS/global_plan \
  /tf /tf_static \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info \
  /move_base/status /move_base/goal /move_base/result /move_base/feedback \
  /move_base_simple/goal \
  /goal_manager/current_target \
  /rosout
```

Flags:
- `--lz4` — fast streaming compression
- `--split --size=2048` — roll over every 2 GB to keep files manageable
- `-O <name>` — output bag name (timestamped above)

### Lighter recordings

```bash
# Camera only (for offline detector replay)
rosbag record --lz4 -O camera_only_$(date +%Y%m%d_%H%M%S).bag \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info \
  /tf /tf_static

# Navigation only
rosbag record --lz4 -O nav_only_$(date +%Y%m%d_%H%M%S).bag \
  /tf /tf_static \
  /move_base/status /move_base/goal /move_base/result /move_base/feedback \
  /move_base/TebLocalPlannerROS/global_plan \
  /goal /goal_manager/current_target /rosout
```

### Playback

```bash
# Inspect
rosbag info my_recording.bag

# Play with clock so node timestamps line up
rosparam set /use_sim_time true
rosbag play --clock my_recording.bag

# Filter topics on playback
rosbag play --clock --topics /camera/color/image_raw /camera/aligned_depth_to_color/image_raw my_recording.bag

rosbag play --clock --pause   door_coord_test_20260628_162442_0.bag   door_coord_test_20260628_162442_1.bag   door_coord_test_20260628_162442_2.bag

rosbag play --clock --pause --rate 0.5 --start=80 --duration=20 door_coord_test_20260628_162442_0.bag door_coord_test_20260628_162442_1.bag door_coord_test_20260628_162442_2.bag


```

## 11. TensorRT / model commands

```bash
# Export PyTorch → ONNX (run on x86_64 with GPU)
python3 /home/ias/satya/catkin_ws/src/door_navigation/scripts/tensorrt_export.py

# ONNX → TensorRT engine (run on Jetson)
/usr/src/tensorrt/bin/trtexec \
  --onnx=/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_vits.onnx \
  --saveEngine=/home/ias/satya/catkin_ws/src/door_navigation/checkpoints/depth_anything_v2_vits.engine \
  --fp16 --workspace=4096 --timingCacheFile=trt_cache.bin --verbose
```

## 12. Common one-liners

```bash
# Kill every ROS process on this machine
rosnode kill -a ; killall -9 rosmaster roscore

# Check ROS connectivity to Go1
ping 192.168.123.161   # RPi
ping 192.168.123.15    # Jetson

# Time sync sanity check (see time_sync_setup.md for details)
chronyc tracking
chronyc sources -v
```
