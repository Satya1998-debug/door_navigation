# door_navigation

ROS Noetic package for **indoor navigation through doors** on a Unitree Go1 with a Jetson Orin and a RealSense RGB-D camera. It plugs a lightweight door-perception + coordinator layer on top of an existing `move_base` / AMCL stack, and exposes an agent-facing ROS service so an external LLM guide (running outside ROS) can request navigation over rosbridge.

## Architecture highlights

### End-to-end workflow

The complete system connects the human-facing agent, rosbridge service layer, and ROS navigation middleware:

![Overall human-agent-rosbridge-ROS workflow](docs/images/01_overall_workflow.png)

### ROS navigation and door traversal

The ROS-side flow is split into goal dispatch and path monitoring (Part 1), followed by door perception, passability decisions, human approval, and traversal (Part 2):

<p align="center">
  <img src="docs/images/04a_ros_flow_after_bridge_part1.png" alt="ROS flow Part 1: navigation trigger, path monitoring, and pre-door approach" width="48%">
  <img src="docs/images/04b_ros_flow_after_bridge_part2.png" alt="ROS flow Part 2: door handling, approval, and traversal" width="48%">
</p>

---

## Short system overview

**Runtime flow of one navigation request**

1. LLM tool calls `/agent/start_navigation` (person, room) via rosbridge.
2. `robot_command_bridge` looks up the pose in `saved_locations_*.yaml`, publishes it on `/goal` (remapped to `move_base_simple/goal`), and blocks in `GoalManager.wait_for_target_reached(...)`.
3. `move_base` drives; `door_detect_pose_estimate_node` publishes `/door_poses` at ~5 Hz.
4. `door_coordinator_node` watches the current global plan; when a door intersects the plan it preempts `move_base`, drives to a **pre-door** pose, calls `/door/estimate_state` (YOLO + DepthAnythingTRT + VLM), optionally asks a human via `/voice/speak` + `/voice/listen`, and on approval drives to a **post-door** pose and resumes the original goal.
5. `GoalManager` returns success (or a failure reason) to the bridge, which returns it to the agent.

**Where things live**

- `scripts/robot_command_bridge.py` — exposes `/agent/start_navigation`.
- `scripts/goal_sender.py` — `GoalManager` (goal publish + arrival wait).
- `scripts/door_detect_pose_estimate_node.py` — perception (YOLO + depth).
- `scripts/door_state_estimator_node.py` — state estimation service.
- `scripts/door_coordinator_node.py` — state machine that supervises door crossings.
- `scripts/voice_assistant_node.py` — audio hardware owner + `/voice/speak`, `/voice/listen`.
- `scripts/voice_assistant.py` — library used by the node (local mode) and by the coordinator (ros-service mode).
- `launch/door_navigation.launch` — camera + perception + coordinator + voice.
- `launch/door_agent_bringup.launch` — the above + rosbridge + `robot_command_bridge`.

---

## Documentation index

| Doc | What it covers |
|---|---|
| [`SETUP.md`](SETUP.md) | Workspace bootstrap, cv_bridge for Python 3.8, RealSense wrapper, catkin build, transformer models (DepthAnythingV2, YOLO, VLM via Ollama), rosbridge, TensorRT conversion. |
| [`internet_setup.md`](internet_setup.md) | Getting the Jetson online via the IAS PC (`192.168.123.148`), then Uni-Stuttgart Wi-Fi, then back onto the Go1 LAN. |
| [`time_sync_setup.md`](time_sync_setup.md) | Chrony topology PC↔Jetson↔Go1, the strict "never install `systemd-timesyncd` on the Jetson" rule, per-boot procedure, verification checklist, troubleshooting. |
| [`commands.md`](commands.md) | Day-to-day cheatsheet: sourcing the workspace, camera launch, `move_base`, goal sender, per-node run commands, rosbridge, rosbag recording/playback, TensorRT commands. |
| [`setup_issues_jetson_env.md`](setup_issues_jetson_env.md) | Jetson-specific gotchas (Python venv, PyTorch wheels, ROS bridge `attrs`/`twisted` fix). |
| [`unitree_go1_steps.md`](unitree_go1_steps.md) | Go1 remote-controller, calibration, power-on/off sequence, high-level vs low-level mode. |
| [`TRT_ENGINE_PERFORMANCE_ACHIEVEMENTS.md`](TRT_ENGINE_PERFORMANCE_ACHIEVEMENTS.md) | DepthAnythingV2 TRT vs PyTorch benchmark notes. |

---

## Quick start (assumes SETUP.md is already done and clocks are in sync)

Terminal 1 — ROS master:
```bash
source ~/satya/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://192.168.123.15:11311
export ROS_HOSTNAME=192.168.123.15
roscore
```

Terminal 2 — base navigation stack (map + move_base + AMCL, from the `navigation` package):
```bash
roslaunch navigation go1_navigation.launch map_file:=/home/maps/gmapping/map_test.yaml
```

Terminal 3 — door pipeline + rosbridge + agent bridge:
```bash
roslaunch door_navigation door_agent_bringup.launch \
    nav_wait_timeout_sec:=1200 \
    nav_position_tolerance:=0.15
```

Terminal 4 (optional, without the LLM) — trigger a goal directly:
```bash
rosservice call /agent/start_navigation "person: ''
room: 'home'"
```

See [`commands.md`](commands.md) for the full command surface.
