#!/home/ias/satya/venv38/bin/python3

import os
import sys
import time
import rospy

# Path setup
import rospkg
rospack = rospkg.RosPack()
PACKAGE_PATH = rospack.get_path('door_navigation')
script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
from door_navigation.srv import EstimateDoorState
from voice_assistant import get_voice_assistant


class RosTestInterface:
    def __init__(self, testing=True):
        self.testing = testing
        
        # Door state estimator service client
        rospy.loginfo("Waiting for door state estimator service...")
        try:
            rospy.wait_for_service("/door/estimate_state")
            self.door_state_service = rospy.ServiceProxy("/door/estimate_state", EstimateDoorState)
            rospy.loginfo("Connected to door state estimator service")
            
            self.voice_assistant = get_voice_assistant(enable_listening=True)
            rospy.loginfo("Voice assistant ready for coordinator announcements")
        except Exception as e:
            rospy.logwarn(f"Error in RosTestInterface initialization: {e}")
            
    def interact_with_human(self, conversation):
        time.sleep(5)  # wait a bit before asking for confirmation
        prompt = "Is the door safe to traverse? Please say yes or no."
        self.voice_assistant.speak(prompt)
        for attempt in range(3):
            time.sleep(3)  # wait a bit for the human to respond
            self.voice_assistant.speak("Speak now.")
            feedback = self.voice_assistant.get_voice_input()
            if not feedback:
                rospy.loginfo(f"No voice confirmation captured (attempt {attempt + 1}/{self.voice_confirmation_max_tries})")
                continue

            fb = feedback.lower()
            rospy.loginfo(f"Human voice confirmation: {fb}")
            if any(word in fb for word in ["yes", "sure", "go ahead", "okay", "ok"]):
                rospy.loginfo(f"Human confirmation received: {conversation}")
                return True
            if any(word in fb for word in ["no", "wait", "stop", "not safe"]):
                return False
            
    def call_door_state_estimator(self):
        if self.door_state_service is None:
            rospy.logwarn("Door state estimator service not available, returning default state")
            return False
        
        try:
            rospy.loginfo("Calling door state estimator service...")
            response = self.door_state_service()
            rospy.loginfo("Door state estimator response received")
            rospy.loginfo(f"Door state: {response.door_state}, passable: {response.is_passable}, conversation: {response.conversation}")
            self.voice_assistant.speak(response.conversation)
            
            approved = self.interact_with_human(response.conversation)
                        
            # if YES, perform state eastimation again then proceed
            if approved:
                # SPEAK that robot is proceeding through the door
                rospy.loginfo("Human confirmed door is safe to traverse")
                self.voice_assistant.speak("Human confirmed. Proceeding through the door.")
                return True
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False