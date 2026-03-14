#!/usr/bin/env python3

"""
Set initial pose for LIO-SAM localization without RViz.
Usage: 
  rosrun door_navigation set_initial_pose.py --x 7.77 --y 2.44 --yaw -1.08
  OR
  rosrun door_navigation set_initial_pose.py --location "Office_room"
"""

import rospy
import yaml
import argparse
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import math

def set_initial_pose(x, y, yaw):
    """Publish initial pose to /initialpose topic."""
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1, latch=True)
    rospy.init_node('set_initial_pose', anonymous=True)
    rospy.sleep(1.0)  # Wait for publisher to be ready
    
    initial_pose = PoseWithCovarianceStamped()
    initial_pose.header.frame_id = "map"
    initial_pose.header.stamp = rospy.Time.now()
    
    # Position
    initial_pose.pose.pose.position.x = x
    initial_pose.pose.pose.position.y = y
    initial_pose.pose.pose.position.z = 0.0
    
    # Orientation (convert yaw to quaternion)
    quat = quaternion_from_euler(0, 0, yaw)
    initial_pose.pose.pose.orientation.x = quat[0]
    initial_pose.pose.pose.orientation.y = quat[1]
    initial_pose.pose.pose.orientation.z = quat[2]
    initial_pose.pose.pose.orientation.w = quat[3]
    
    # Covariance matrix (x, y, yaw uncertainty)
    # Small values = high confidence in initial pose
    initial_pose.pose.covariance[0] = 0.25   # x variance
    initial_pose.pose.covariance[7] = 0.25   # y variance  
    initial_pose.pose.covariance[35] = 0.068 # yaw variance (about 15 degrees)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("  Setting Initial Pose for LIO-SAM Localization")
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"  Position: x={x:.3f}m, y={y:.3f}m")
    rospy.loginfo(f"  Orientation: yaw={math.degrees(yaw):.1f}° ({yaw:.3f} rad)")
    rospy.loginfo("=" * 60)
    
    # Publish multiple times to ensure it's received
    for _ in range(3):
        pub.publish(initial_pose)
        rospy.sleep(0.5)
    
    rospy.loginfo("✓ Initial pose published!")
    rospy.loginfo("  LIO-SAM will refine this using ICP matching...")

def load_location_from_yaml(location_name, yaml_file):
    """Load location pose from locations.yaml file."""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            
        if 'locations' in data:
            locations = data['locations']
        else:
            locations = data
            
        if location_name not in locations:
            rospy.logerr(f"Location '{location_name}' not found in {yaml_file}")
            return None
            
        loc = locations[location_name]
        x = loc['pose']['position']['x']
        y = loc['pose']['position']['y']
        
        # Convert quaternion to yaw
        quat = [
            loc['pose']['orientation']['x'],
            loc['pose']['orientation']['y'],
            loc['pose']['orientation']['z'],
            loc['pose']['orientation']['w']
        ]
        _, _, yaw = euler_from_quaternion(quat)
        
        return x, y, yaw
        
    except Exception as e:
        rospy.logerr(f"Error loading location: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set initial pose for localization')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--location', '-l', type=str, 
                      help='Location name from locations.yaml (e.g., "Office_room")')
    group.add_argument('--x', type=float, 
                      help='X position in map frame (meters)')
    
    parser.add_argument('--y', type=float, 
                        help='Y position in map frame (meters)')
    parser.add_argument('--yaw', type=float, 
                        help='Yaw orientation in radians (or use --deg for degrees)')
    parser.add_argument('--deg', type=float, 
                        help='Yaw orientation in degrees')
    parser.add_argument('--yaml', type=str, 
                        default='/home/satya/MT/catkin_ws/src/a2_ros2udp/params/locations.yaml',
                        help='Path to locations.yaml file')
    
    args = parser.parse_args()
    
    try:
        if args.location:
            # Load from YAML
            result = load_location_from_yaml(args.location, args.yaml)
            if result is None:
                exit(1)
            x, y, yaw = result
        else:
            # Use provided coordinates
            if args.x is None or args.y is None:
                parser.error("--x and --y are required when not using --location")
            
            x = args.x
            y = args.y
            
            if args.deg is not None:
                yaw = math.radians(args.deg)
            elif args.yaw is not None:
                yaw = args.yaw
            else:
                parser.error("Either --yaw or --deg is required")
        
        set_initial_pose(x, y, yaw)
        
    except rospy.ROSInterruptException:
        pass
