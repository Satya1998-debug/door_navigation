#!/usr/bin/env python3

"""
Simple script to save current robot pose to a YAML file.
Usage: rosrun door_navigation save_location.py --name "Office_door"
"""

import rospy
import tf
import yaml
import argparse
import os
from pathlib import Path

def save_current_pose(location_name, output_file="saved_locations.yaml"):
    """Save current robot pose from TF to YAML file."""
    
    rospy.init_node('save_location', anonymous=True)
    
    # Create TF listener
    tf_listener = tf.TransformListener()
    
    rospy.loginfo(f"Waiting for TF transform from 'map' to 'base_link'...")
    rospy.sleep(1.0)  # Give TF time to populate
    
    try:
        # Get current transform
        tf_listener.waitForTransform('map', 'base_link', rospy.Time(0), rospy.Duration(4.0))
        (trans, rot) = tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
        
        rospy.loginfo(f"Current pose:")
        rospy.loginfo(f"  Position: x={trans[0]:.6f}, y={trans[1]:.6f}, z={trans[2]:.6f}")
        rospy.loginfo(f"  Orientation (quaternion): x={rot[0]:.6f}, y={rot[1]:.6f}, z={rot[2]:.6f}, w={rot[3]:.6f}")
        
        # Create location entry
        location_data = {
            location_name: {
                'header': {
                    'frame_id': 'map'
                },
                'pose': {
                    'position': {
                        'x': float(trans[0]),
                        'y': float(trans[1]),
                        'z': float(trans[2])
                    },
                    'orientation': {
                        'x': float(rot[0]),
                        'y': float(rot[1]),
                        'z': float(rot[2]),
                        'w': float(rot[3])
                    }
                }
            }
        }
        
        # Load existing file or create new dict
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_locations = yaml.safe_load(f)
                if all_locations is None:
                    all_locations = {'locations': {}}
                elif 'locations' not in all_locations:
                    all_locations = {'locations': all_locations}
        else:
            all_locations = {'locations': {}}
        
        # Add/update location
        all_locations['locations'].update(location_data)
        
        # Save to file
        with open(output_file, 'w') as f:
            yaml.dump(all_locations, f, default_flow_style=False, sort_keys=False)
        
        rospy.loginfo(f"✓ Saved location '{location_name}' to {output_file}")
        
        return True
        
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(f"TF Error: {e}")
        rospy.logerr("Make sure the robot is running and TF is being published!")
        return False
    except Exception as e:
        rospy.logerr(f"Error saving location: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save current robot pose to YAML file')
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='Name for this location (e.g., "Office_door", "Lab_entrance")')
    parser.add_argument('--output', '-o', type=str, default='saved_locations.yaml',
                        help='Output YAML file (default: saved_locations.yaml)')
    
    args = parser.parse_args()
    
    # Replace spaces with underscores in location name
    location_name = args.name.replace(' ', '_')
    
    print(f"\n{'='*60}")
    print(f"  Saving Location: {location_name}")
    print(f"{'='*60}\n")
    
    success = save_current_pose(location_name, args.output)
    
    if success:
        print(f"\n{'='*60}")
        print(f"  ✓ SUCCESS - Location saved!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  ✗ FAILED - Could not save location")
        print(f"{'='*60}\n")
