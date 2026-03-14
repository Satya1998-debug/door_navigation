# Array of door poses detected in a single frame
std_msgs/Header header          # timestamp + frame_id ("map")
DoorPose[] doors                # list of door poses in this frame
