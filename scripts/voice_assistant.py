#!/home/ias/satya/venv38/bin/python3

"""
Voice Assistant class for the door navigation system. Adapted and improved from previous voice assistant code from RAG source code.
"""

import pyttsx3 # used for text-to-speech conversion
from vosk import Model, KaldiRecognizer, SetLogLevel # used for speech-to-text recognition
import pyaudio # used for recording and playing audio (via microphone and speakers)
import wave # used for handling WAV audio files
import audioop
import json
import os, sys
import contextlib
import time
from datetime import datetime

# ------ path setup -----
try:
    import rospkg
    rospack = rospkg.RosPack()
    PACKAGE_PATH = rospack.get_path('door_navigation')
except (rospkg.ResourceNotFound, rospkg.common.ResourceNotFound):
    PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"[door-pose-estimator] rospkg not available, using relative path: {PACKAGE_PATH}")

script_dir = os.path.join(PACKAGE_PATH, 'scripts')
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    
import rospy
from utils.config import (
    SPEECH_RECOGNITION_MODEL_PATH,
    SPEECH_RECOGNITION_MODEL,
    SPEECH_OUTPUT_DIR,
    SPEAKER_DEVICE_INDEX,
    MIC_DEVICE_INDEX,
    VOSK_ENABLE_LOGS,
    QUIET_ALSA_WARNINGS,
)

VOSK_RATE = 16000

# Placeholder / no-op strings we don't send to TTS (avoid mispronouncing
# "NA", "N/A", ... coming from downstream services).
_PLACEHOLDER_TEXTS = ("", "NA", "N/A", "NONE", "NULL")


class VoiceAssistant:
    
    def __init__(self, enable_listening=False, use_ros_service=True, service_wait_timeout=5.0):
        self.use_ros_service = bool(use_ros_service)
        self.enable_listening = bool(enable_listening)

        self.output_dir = SPEECH_OUTPUT_DIR

        # Attributes always defined so close() / probes don't blow up in
        # whichever mode we end up in.
        self.audio_interface = None
        self.stream = None
        self.tts_engine = None
        self.vosk_model = None
        self.recognizer = None
        self._speak_srv = None
        self._listen_srv = None

        if self.use_ros_service: # True (when ros node uses the class, as it has to use audio hardware)
            self._init_ros_service_backend(enable_listening, service_wait_timeout)
        else: # False (when voice_assistant_node.py uses the class, it just does the service calls)
            self._init_local_audio_backend(enable_listening)

    # ------------------------------------------------------------------ init

    def _init_ros_service_backend(self, enable_listening, service_wait_timeout):
        """Wire up rospy.ServiceProxy handles; do NOT touch audio hardware."""
        from door_navigation.srv import Speak, Listen  # deferred: only pulled in client mode

        # remembered so speak() can rebuild the proxy after a transient master hiccup
        self._Speak = Speak

        try:
            rospy.wait_for_service("/voice/speak", timeout=float(service_wait_timeout))
            self._speak_srv = rospy.ServiceProxy("/voice/speak", Speak, persistent=False)
        except (rospy.ROSException, rospy.ROSInterruptException) as e:
            rospy.logwarn(
                f"[VoiceAssistant] /voice/speak not reachable: {e}. "
                "speak() will fall back to logging."
            )

        if enable_listening:
            try:
                rospy.wait_for_service("/voice/listen", timeout=float(service_wait_timeout))
                self._listen_srv = rospy.ServiceProxy("/voice/listen", Listen, persistent=False)
                # Keep the same "truthy => listening available" contract as the local
                # backend: downstream code can still do `if va.recognizer is None`.
                self.recognizer = self._listen_srv
                self.stream = self._listen_srv
            except (rospy.ROSException, rospy.ROSInterruptException) as e:
                rospy.logwarn(
                    f"[VoiceAssistant] /voice/listen not reachable: {e}. "
                    "get_voice_input() will return empty strings."
                )

        print(f"Voice Assistant initialized in ROS-service mode (listening={enable_listening}).")

    def _init_local_audio_backend(self, enable_listening):
        """Original local hardware init: pyttsx3 + Vosk + PyAudio."""
        os.makedirs(self.output_dir, exist_ok=True)
        self.tts_engine = pyttsx3.init('espeak')  # Use 'espeak' for better Linux compatibility

        # get rate and volume for espeak
        rate = self.tts_engine.getProperty('rate')
        volume = self.tts_engine.getProperty('volume')
        print(f"Initial TTS rate: {rate}, volume: {volume}")

        # set espeak properties (tune as needed)
        self.tts_engine.setProperty('rate', 170)  # slower rate for clarity
        self.tts_engine.setProperty('volume', 1.0)

        # Keep Vosk logs optional to reduce console noise on embedded targets.
        SetLogLevel(0 if VOSK_ENABLE_LOGS else -1)

        if enable_listening:
            speech_to_text_model_path = SPEECH_RECOGNITION_MODEL_PATH + SPEECH_RECOGNITION_MODEL
            print(f"Loading Vosk model from: {speech_to_text_model_path}")
            if not os.path.exists(speech_to_text_model_path):
                raise FileNotFoundError("Please download the Vosk model and place it in the working directory.")

            self.vosk_model = Model(speech_to_text_model_path)
            self.recognizer = KaldiRecognizer(self.vosk_model, VOSK_RATE)

            with self._maybe_quiet_alsa():
                self.audio_interface = pyaudio.PyAudio()
                self.stream = self.audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=VOSK_RATE,
                    input=True,
                    input_device_index=MIC_DEVICE_INDEX,
                    frames_per_buffer=8192,
                )
            self.stream.start_stream()
            print("Voice Assistant initialized successfully.")
        else:
            print("Voice Assistant initialized in TTS-only mode.")

    # ------------------------------------------------------------------ shared helpers

    @staticmethod
    def _is_placeholder(text):
        if text is None:
            return True
        s = str(text).strip()
        return (not s) or s.upper() in _PLACEHOLDER_TEXTS

    @contextlib.contextmanager
    def _maybe_quiet_alsa(self):
        """Optionally silence ALSA backend stderr noise during device probing/open."""
        if not QUIET_ALSA_WARNINGS:
            yield
            return

        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 2)
            yield
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)

    # ------------------------------------------------------------------ TTS

    def speak(self, text="", blocking=True):
        """Speak ``text``.

        In ROS-service mode this becomes a ``/voice/speak`` call. In local
        mode it runs pyttsx3/espeak and plays the resulting WAV. Empty or
        placeholder strings ("NA", "N/A", ...) are silently dropped in both
        modes.

        ``blocking`` is only meaningful in ROS-service mode: when ``False``,
        the utterance is enqueued on the node and the call returns
        immediately (useful for fire-and-forget narration). Local mode is
        always synchronous, so the flag is ignored there.
        """
        if self._is_placeholder(text):  # if text is like "NA", "N/A", ..., return immediately
            return

        # --- ROS-service backend -------------------------------------------------
        # will be used by door_coordinator_node.py
        if self.use_ros_service:
            stripped = str(text).strip()
            if self._speak_srv is None:
                rospy.loginfo(f"[SPEAK-fallback] {stripped}")
                return
            try:
                self._speak_srv(text=stripped, blocking=bool(blocking)) # Service call happens here
            except Exception as e:
                # transient "unable to contact master" / stale proxy: rebuild once and retry.
                try:
                    self._speak_srv = rospy.ServiceProxy("/voice/speak", self._Speak, persistent=False)
                    self._speak_srv(text=stripped, blocking=bool(blocking))
                except Exception as e2:
                    rospy.logwarn(f"[VoiceAssistant] speak service failed: {e2}; fallback: [SPEAK] {stripped}")
            return

        # --- Local hardware backend ---------------------------------------------
        # will be used by voice_assistant_node.py
        print("Using Speak ...")
        if text.endswith(".wav"):
            self.play_wav(os.path.join(self.output_dir, text))
            print(f"Played WAV file: {text}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)

        self.tts_engine.save_to_file(text, filepath)
        self.tts_engine.runAndWait()
        # wait briefly for the WAV file to appear and grow past the header.
        for _ in range(10):
            if os.path.exists(filepath) and os.path.getsize(filepath) > 44:
                break
            time.sleep(0.1)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 44:
            try:
                self.play_wav(filepath)
            except Exception as e:
                print(f"[WARN] Could not play generated WAV: {e}")
        else:
            print(f"[ERROR] TTS did not generate a valid WAV file: {filepath}")

    def play_wav(self, filepath):
        """Plays a WAV audio file using PyAudio (local mode only)."""
        chunk = 1024
        with self._maybe_quiet_alsa():
            pa = pyaudio.PyAudio()
        stream = None
        output_device_index = self._resolve_output_device(pa) # None if the ID does not exist or is not set, which will use the system default output device

        try:
            with wave.open(filepath, 'rb') as wf:
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                src_rate = wf.getframerate()
                target_rate = self._select_playback_rate(
                    pa,
                    sample_width,
                    channels,
                    src_rate,
                    output_device_index=output_device_index,
                )
                needs_resample = target_rate != src_rate
                ratecv_state = None

                with self._maybe_quiet_alsa():
                    stream = pa.open(
                        format=pa.get_format_from_width(sample_width),
                        channels=channels,
                        rate=target_rate,
                        output=True,
                        output_device_index=output_device_index,
                    )

                data = wf.readframes(chunk)
                while data:
                    if needs_resample:
                        data, ratecv_state = audioop.ratecv(
                            data,
                            sample_width,
                            channels,
                            src_rate,
                            target_rate,
                            ratecv_state,
                        )
                    stream.write(data)
                    data = wf.readframes(chunk)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()

    def _select_playback_rate(self, pa, sample_width, channels, src_rate, output_device_index=None):
        """Return a speaker-supported playback rate, preferring the source WAV rate."""
        output_format = pa.get_format_from_width(sample_width)
        try:
            with self._maybe_quiet_alsa():
                kwargs = {
                    "output_channels": channels,
                    "output_format": output_format,
                }
                if output_device_index is not None:
                    kwargs["output_device"] = output_device_index
                pa.is_format_supported(src_rate, **kwargs)
            return src_rate # 44100 Hz
        except ValueError:
            fallback_rate = 48000
            if output_device_index is not None:
                dev_info = pa.get_device_info_by_index(output_device_index)
                fallback_rate = int(dev_info.get("defaultSampleRate", fallback_rate))
            if fallback_rate != src_rate:
                print(
                    f"[WARN] Playback rate {src_rate} not supported by device "
                    f"{output_device_index if output_device_index is not None else 'default'}. "
                    f"Using {fallback_rate} Hz."
                )
            return fallback_rate

    def _resolve_output_device(self, pa):
        """Return a usable output device index or None for system default."""
        if SPEAKER_DEVICE_INDEX is None or SPEAKER_DEVICE_INDEX < 0:
            return None

        try:
            pa.get_device_info_by_index(SPEAKER_DEVICE_INDEX)
            return SPEAKER_DEVICE_INDEX
        except Exception as exc:
            print(
                f"[WARN] Speaker device index {SPEAKER_DEVICE_INDEX} is invalid: {exc}. "
                "Falling back to default output device."
            )
            return None

    # ------------------------------------------------------------------ speech-to-text

    def get_voice_input(self, timeout_sec=None):
        """Return the next transcribed utterance (lowercased) or ""..

        ROS-service mode forwards to ``/voice/listen``; local mode consumes
        frames from the PyAudio stream and feeds them to Vosk (the pre-refactor
        behaviour). Raises ``RuntimeError`` in both modes when listening is
        disabled or unavailable, matching the original class contract.
        """
        # --- ROS-service backend -------------------------------------------------
        if self.use_ros_service:
            if self._listen_srv is None:
                raise RuntimeError("Voice input requested but /voice/listen is unavailable.")
            req_timeout = float(timeout_sec) if timeout_sec else 0.0
            try:
                resp = self._listen_srv(timeout_sec=req_timeout, grammar="")
                if getattr(resp, "timed_out", False):
                    return ""
                return (resp.text or "").lower()
            except Exception as e:
                rospy.logwarn(f"[VoiceAssistant] listen service failed: {e}")
                return ""

        # --- Local hardware backend ---------------------------------------------
        if self.stream is None or self.recognizer is None:
            raise RuntimeError("Voice input requested but listening is disabled.")

        print("🎤 Listening... Please speak clearly.")
        start_time = time.monotonic()
        while True:
            if timeout_sec is not None and timeout_sec > 0:
                if time.monotonic() - start_time > timeout_sec:
                    return ""
            data = self.stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:
                    return text.lower()

    def get_speech_input(self):
        """async voice recognition stub (unchanged)."""
        print(" Listening for speech...")
        return "Hi"

    def get_text_input(self):
        """captures text input asynchronously."""
        text = input("Type your query: ")
        return text.strip().lower()

    # ------------------------------------------------------------------ shutdown

    def close(self):
        """Release audio resources (no-op in ROS-service mode)."""
        if self.use_ros_service:
            return
        if hasattr(self, "stream") and self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if hasattr(self, "audio_interface") and self.audio_interface is not None:
            try:
                self.audio_interface.terminate()
            except Exception:
                pass


_voice_assistant_instance = None


def get_voice_assistant(enable_listening=True, use_ros_service=True):
    """Return a process-wide shared :class:`VoiceAssistant`.

    ``use_ros_service`` controls the backend:

    - ``True``  -- talk to ``/voice/speak`` and ``/voice/listen``. Safe default
                   for every in-process consumer (door coordinator, tests, ...)
                   so the ALSA capture device has a single owner.
    - ``False`` -- own pyaudio / pyttsx3 / vosk directly. **Only**
                   ``voice_assistant_node.py`` should use this.
    The instance is created lazily on the first call.
    """
    global _voice_assistant_instance
    if _voice_assistant_instance is None:
        _voice_assistant_instance = VoiceAssistant(enable_listening=enable_listening, use_ros_service=use_ros_service)
    return _voice_assistant_instance


if __name__ == "__main__":
    # Quick smoke test path -- always exercises the LOCAL backend so we can
    # verify audio hardware without the ROS graph running.
    assistant = VoiceAssistant(enable_listening=True, use_ros_service=False)
    try:
        text_response = "Hello! I am your door navigation assistant. How can I help you today?"
        assistant.speak(text_response)
        if assistant.recognizer is None:
            print("Listening is disabled. Enable listening to use voice input.")
        else:
            while True:
                user_input = assistant.get_voice_input()
                print(f"You said: {user_input}")
                if "exit" in user_input or "quit" in user_input:
                    print("Exiting voice assistant.")
                    break
                response = f"You said: {user_input}"
                assistant.speak(response)

    finally:
        assistant.close()
