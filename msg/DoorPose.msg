# Door 3D pose in map frame
std_msgs/Header header          # timestamp + frame_id ("map")
geometry_msgs/Point position    # door center in map frame
geometry_msgs/Vector3 normal    # door normal in map frame (unit vector)
float32 width                   # door width in meters
string door_type                # "door_single" or "door_double"
float32 confidence              # detection confidence
