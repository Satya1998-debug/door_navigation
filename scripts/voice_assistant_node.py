#!/home/ias/satya/venv38/bin/python3

"""Voice assistant ROS node.

The only process in the stack that opens the microphone, espeak/pyttsx3 and
Vosk. Every other component reaches audio through ``/voice/speak`` and
``/voice/listen``, so no two processes ever fight over the ALSA capture
device.

Intentionally thin: this node just wraps a local-mode
:class:`VoiceAssistant` with two ROS services. Requests are handled inline
in the service callbacks -- the same synchronous flow the old (pre-refactor)
coordinator and guide already relied on. The only concurrency guard is a
single :class:`threading.Lock` around ``va.speak(...)``, because rospy may
dispatch overlapping ``/voice/speak`` requests on separate callback threads
and pyttsx3 is not thread-safe.
"""

import os
import sys
import threading

import rospkg
import rospy


try:
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path("door_navigation")
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

script_dir = os.path.join(PACKAGE_PATH, "scripts")
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from door_navigation.srv import (
    Speak,
    SpeakResponse,
    Listen,
    ListenResponse,
)
from voice_assistant import VoiceAssistant


_PLACEHOLDER_TEXTS = ("", "NA", "N/A", "NONE", "NULL")


def _is_placeholder(text):
    if text is None:
        return True
    s = str(text).strip()
    return (not s) or s.upper() in _PLACEHOLDER_TEXTS


class VoiceAssistantNode:
    """Wraps a local-mode VoiceAssistant behind /voice/speak and /voice/listen."""

    def __init__(self):
        rospy.init_node("voice_assistant_node")

        self.enable_listening = bool(rospy.get_param("~enable_listening", True))
        self.default_listen_timeout = float(rospy.get_param("~default_listen_timeout", 10.0))
        self.use_ros_service = bool(rospy.get_param("~use_ros_service", False))

        try:
            self.va = VoiceAssistant(enable_listening=self.enable_listening, use_ros_service=self.use_ros_service)
        except Exception as e:
            rospy.logerr(f"[voice_assistant] Init with listening={self.enable_listening} failed: {e}. Falling back to TTS-only mode.")
            self.enable_listening = False
            self.va = VoiceAssistant(enable_listening=False, use_ros_service=False)

        # pyttsx3 is not thread-safe. In practice the coordinator and the
        # guide never speak at the same instant, but rospy will still dispatch
        # each /voice/speak service call on its own callback thread, so this
        # lock is a near-free defense against accidental overlap.
        self._tts_lock = threading.Lock()

        rospy.Service("/voice/speak", Speak, self._speak_cb)
        rospy.Service("/voice/listen", Listen, self._listen_cb)

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            "[voice_assistant] Ready. listening=%s default_listen_timeout=%.1fs",
            self.enable_listening, self.default_listen_timeout)

    # ------------------------------------------------------------------ services

    def _speak_cb(self, req):
        if _is_placeholder(req.text):
            return SpeakResponse(success=True, message="empty_or_placeholder")

        text = str(req.text).strip()
        rospy.loginfo("[voice_assistant] SPEAK request: %s", text)
        with self._tts_lock:
            try:
                self.va.speak(text)
            except Exception as e:
                rospy.logwarn(f"[voice_assistant] TTS failed for text=%r: {e}", text)
                return SpeakResponse(success=False, message=str(e))
        return SpeakResponse(success=True, message="")

    def _listen_cb(self, req):
        if self.va.recognizer is None:
            rospy.loginfo("[voice_assistant] LISTEN request ignored: recognizer unavailable")
            return ListenResponse(text="", timed_out=True)

        timeout = float(req.timeout_sec) if req.timeout_sec and req.timeout_sec > 0 else self.default_listen_timeout
        rospy.loginfo("[voice_assistant] LISTEN request: timeout=%.2fs", timeout)
        try:
            text = self.va.get_voice_input(timeout_sec=timeout) or ""
        except Exception as e:
            rospy.logwarn(f"[voice_assistant] Listen failed: {e}")
            return ListenResponse(text="", timed_out=True)
        if text:
            rospy.loginfo("[voice_assistant] HEARD: %s", text)
        else:
            rospy.loginfo("[voice_assistant] HEARD: <timeout/no speech>")
        return ListenResponse(text=text, timed_out=(text == ""))

    # ------------------------------------------------------------------ shutdown

    def _shutdown(self):
        try:
            self.va.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        VoiceAssistantNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
