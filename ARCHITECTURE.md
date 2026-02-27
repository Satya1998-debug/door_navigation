# Door Navigation - Distributed Architecture

## Overview
The door navigation system has been refactored into a distributed architecture to avoid blocking the 10Hz control loop:

**Architecture:**
```
┌─────────────────────┐      /door/detections     ┌──────────────────────┐
│ door_detector_node  │─────────────────────────>│ door_pose_estimator  │
│  (YOLO @ 5Hz)       │                           │  (DepthAnything +TF) │
└─────────────────────┘                           └──────────────────────┘
                                                             │
                                                             │ /door/poses
                                                             v
                                                   ┌────────────────────────┐
                                                   │ door_coordinator       │
                                                   │ (State machine @ 10Hz) │
                                                   └────────────────────────┘
                                                             │
                                                             │ service call (at pre-door)
                                                             v
                                                   ┌────────────────────────┐
                                                   │ door_state_estimator   │
                                                   │ (Service: open/closed) │
                                                   └────────────────────────┘
```

## Nodes

### 1. door_detector_node.py
- **Purpose:** Continuous YOLO detection at configurable rate (default: 5Hz)
- **Subscribes:** `/camera/color/image_raw`
- **Publishes:** `/door/detections` (DoorDetection.msg)
- **Computation:** YOLO11m inference (200-500ms)

### 2. door_pose_estimator_node.py
- **Purpose:** Computes 3D door poses from detections using depth and publishes in map frame
- **Subscribes:** `/door/detections`, `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`
- **Publishes:** `/door/poses` (DoorPose.msg)
- **Computation:** DepthAnything v2 + RANSAC plane fitting + TF transforms (100-300ms)

### 3. door_coordinator_refactored.py
- **Purpose:** Door-aware navigation state machine (non-blocking @ 10Hz)
- **Subscribes:** `/door/poses`, `/move_base/TebLocalPlannerROS/global_plan`, `/door/human_confirm`  
- **Publishes:** Sends goals to move_base via actionlib
- **Service Calls:** `/door/estimate_state` when at pre-door position
- **States:** NAVIGATING → APPROACHING_DOOR → WAIT_HUMAN → TRAVERSING → NAVIGATING

### 4. door_state_estimator_node.py
- **Purpose:** On-demand door state estimation service (open/closed/passable)
- **Subscribes:** `/door/poses`, `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`
- **Service:** `/door/estimate_state` (EstimateDoorState.srv)
- **Computation:** Plane fitting (door + wall), opening angle calculation, passability check, optional VLM verification

## Message Definitions

### Complete Workflow

**1. Continuous Detection & Pose Estimation:**
- `door_detector_node` runs YOLO at 5Hz → publishes `/door/detections`
- `door_pose_estimator_node` processes each detection → publishes `/door/poses` (in map frame)

**2. Navigation & Door Detection:**
- `door_coordinator` subscribes to `/door/poses` and checks if any door intersects planned path
- If door detected ahead → transitions to `APPROACHING_DOOR` state
- Sends pre-door goal (stops before door)

**3. Door State Check:**
- When robot reaches pre-door position → coordinator calls `/door/estimate_state` service
- `door_state_estimator_service` analyzes current RGB-D images and returns:
  - `door_state`: "open" / "closed" / "semi_open" / "unknown"
  - `is_passable`: True/False
  - `confidence`: estimation confidence

**4. Decision & Traversal:**
- If `is_passable=True` → immediately send post-door goal and traverse
- If `is_passable=False` → wait in `WAIT_HUMAN` state for confirmation on `/door/human_confirm`
- After traversal → resume original navigation goal

### DoorDetection.msg
```
Header header
float32[4] bbox          # [x1, y1, x2, y2] in pixel coordinates
int32 class_id           # 0=regular_door, 1=double_door  
float32 confidence       # Detection confidence [0.0-1.0]
```

### DoorPose.msg
```
Header header            # frame_id = "map"
geometry_msgs/Point position      # Door center in map frame
geometry_msgs/Vector3 normal      # Normal vector in map frame (points toward robot)
float32 width            # Door width in meters
int32 door_type          # 0=regular_door, 1=double_door
float32 confidence       # Pose estimation confidence
```

### EstimateDoorState.srv (optional)
```
# Request (empty)
---
# Response
int32 door_state         # 0=CLOSED, 1=OPEN, 2=PARTIALLY_OPEN
bool is_passable         # Can robot traverse?
float32 confidence
string error_message
```

## Building

1. **Build messages:**
   ```bash
   cd ~/MT/catkin_ws
   catkin_make
   source devel/setup.bash
   ```

2. **Verify message generation:**
   ```bash
   rosmsg show door_navigation/DoorDetection
   rosmsg show door_navigation/DoorPose
   rossrv show door_navigation/EstimateDoorState
   ```

## Running

### Launch all nodes together:
```bash
roslaunch door_navigation door_navigation.launch
```

### Launch parameters:
- `rgb_topic`: RGB camera topic (default: `/camera/color/image_raw`)
- `depth_topic`: Depth camera topic (default: `/camera/aligned_depth_to_color/image_raw`)
- `detection_rate`: YOLO detection rate in Hz (default: 5.0)
- `pre_door_distance`: Stop distance before door in meters (default: 0.8)
- `post_door_distance`: Goal distance after door in meters (default: 1.0)

### Example with custom parameters:
```bash
roslaunch door_navigation door_navigation.launch detection_rate:=3.0 pre_door_distance:=1.0
```

## Testing Individual Nodes

### Test detector node:
```bash
rosrun door_navigation door_detector_node.py
# In another terminal:
rostopic echo /door/detections
```

### Test pose estimator node:
```bash
rosrun door_navigation door_pose_estimator_node.py
# In another terminal:
rostopic echo /door/poses
```

### Test coordinator:
```bash
rosrun door_navigation door_coordinator_refactored.py
```

### Test door state estimator service:
```bash
# Start the service
rosrun door_navigation door_state_estimator_node.py

# In another terminal, call the service manually
rosservice call /door/estimate_state
```

## Key Changes from Original

### What was removed from coordinator:
- ❌ `DoorDetector` class instantiation (moved to detector node)
- ❌ `RGBDImageReceiver` class (moved to pose estimator node)
- ❌ Inline YOLO detection calls
- ❌ Inline DepthAnything inference
- ❌ `compute_door_pose_in_map_frame()` function
- ❌ `transform_camera_to_map()` for door normal

### What was added to coordinator:
- ✅ Subscriber to `/door/poses` topic
- ✅ `door_pose_callback()` to cache latest poses
- ✅ Modified `is_door_on_path()` to use cached poses (non-blocking)
- ✅ Pose caching with 2-second timeout

### Benefits:
1. **Non-blocking control loop:** State machine runs at 10Hz without waiting for vision
2. **Asynchronous perception:** Detection and pose estimation run independently
3. **Message-based architecture:** Clean separation of concerns via ROS topics
4. **Scalability:** Easy to add more perception nodes or change rates
5. **Debugging:** Can inspect intermediate outputs on `/door/detections` and `/door/poses`

## Monitoring

### Check node status:
```bash
rosnode list | grep door
# Should show:
# /door_coordinator
# /door_detector_node  
# /door_pose_estimator_node
# /door_state_estimator_service
```

### Check topics:
```bash
rostopic list | grep door
# Should show:
# /door/detections
# /door/human_confirm
# /door/poses
```

### Check services:
```bash
rosservice list | grep door
# Should show:
# /door/estimate_state
```

### Check topic rates:
```bash
rostopic hz /door/detections  # Should be ~5Hz
rostopic hz /door/poses        # Should be ~5Hz (one per detection)
```

### Visualize detections:
```bash
rostopic echo /door/detections
# Example output:
# header:
#   frame_id: "camera_color_optical_frame"
# bbox: [120.5, 200.3, 450.2, 600.8]
# class_id: 0
# confidence: 0.87
```

## Troubleshooting

### Coordinator not detecting doors:
- Check if `/door/poses` is being published: `rostopic hz /door/poses`
- Verify TF tree: `rosrun tf view_frames`
- Ensure camera frames exist: `rosrun tf tf_echo map camera_link`

### Detection node not publishing:
- Check RGB camera topic: `rostopic echo /camera/color/image_raw`
- Verify YOLO model weights exist in `weights/` directory
- Check confidence threshold (lower if needed)

### Pose estimator not publishing:
- Verify depth topic: `rostopic echo /camera/aligned_depth_to_color/image_raw`
- Check DepthAnything checkpoints in `checkpoints/` directory
- Examine node output for errors: `rosnode info door_pose_estimator_node`

## Human Confirmation

To confirm door is safe after reaching pre-door pose:
```bash
rostopic pub /door/human_confirm std_msgs/Bool "data: true" --once
```

Or integrate with voice assistant (see `voice_assistant.py`).

## Next Steps (Optional Enhancements)

1. **Door state estimation service:** Implement service call in coordinator to check if door is open/closed
2. **Dynamic reconfigure:** Add runtime parameter adjustment for detection rates
3. **RViz visualization:** Create markers to visualize door poses in 3D
4. **Recovery behaviors:** Handle failed door traversals or blocked doors
5. **Multi-door handling:** Queue multiple doors on path
