# Recording Rosbag for door_navigation — Commands Only

Quick terminal commands to record a rosbag locally, transfer it to the Jetson, and play it back with simulated time.

Prerequisites: source your ROS workspace before running the commands, e.g.:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

1) Record until you stop (Ctrl-C)

```bash
rosbag record -O door_run.bag \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info \
  /tf \
  /tf_static \
  /move_base/TebLocalPlannerROS/global_plan \
  /move_base/status \
  /goal \
  /initialpose \
  /odom \
  /scan \
  --lz4
```

2) Record for a fixed duration (example: 2 minutes)

```bash
timeout 120s rosbag record -O door_run_120s.bag \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info \
  /tf \
  /tf_static \
  /move_base/TebLocalPlannerROS/global_plan \
  /move_base/status \
  /goal \
  /initialpose \
  /odom \
  /scan \
  --lz4
```

3) If your camera topics differ, replace them. Example override:

```bash
rosbag record -O mybag.bag /my/color/topic /my/depth/topic /my/camera_info /tf /tf_static --lz4
```

4) Publish static transforms if TF frames are missing (examples)

```bash
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map camera_link
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link camera_link
```

5) Transfer the bag to Jetson (example)

```bash
scp door_run.bag <jetson_user>@<jetson_ip>:/home/<jetson_user>/
```

6) On the Jetson: enable simulated time and play the bag

```bash
rosparam set /use_sim_time true
rosbag play --clock /home/<jetson_user>/door_run.bag
```

7) Launch the offline test (run while `rosbag play --clock` is active)

```bash
roslaunch door_navigation door_navigation_offline.launch
```

8) Quick checks

```bash
# List camera topics
rostopic list | grep camera

# Verify TF frames
rosrun tf tf_echo map camera_link
rosrun tf tf_monitor
```

That's it — run the `rosbag record` command that matches your topics, copy the `.bag` to the Jetson, set `/use_sim_time` and play with `--clock`, then launch the offline stack.
