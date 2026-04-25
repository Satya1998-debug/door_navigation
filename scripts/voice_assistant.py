#!/home/ias/satya/venv38/bin/python3

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

class VoiceAssistant:
    
    """Handles voice input/output operations for the assistant."""
    
    def __init__(self, enable_listening=False):
        
        """Initializes the voice assistant and audio interfaces."""
        self.output_dir = SPEECH_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.tts_engine = pyttsx3.init('espeak')  # Use 'espeak' for better Linux compatibility
        
        # get rate and volume for espeak
        rate = self.tts_engine.getProperty('rate')
        volume = self.tts_engine.getProperty('volume')
        print(f"Initial TTS rate: {rate}, volume: {volume}")
        
        # set espeak properties (tune as needed)
        self.tts_engine.setProperty('rate', 170)  # slower rate for clarity
        self.tts_engine.setProperty('volume', 0.1)

        # Keep Vosk logs optional to reduce console noise on embedded targets.
        SetLogLevel(0 if VOSK_ENABLE_LOGS else -1)
        
        self.audio_interface = None
        self.stream = None
        self.vosk_model = None
        self.recognizer = None

        if enable_listening:
            speech_to_text_model_path = SPEECH_RECOGNITION_MODEL_PATH + SPEECH_RECOGNITION_MODEL
            print(f"Loading Vosk model from: {speech_to_text_model_path}")
            if not os.path.exists(speech_to_text_model_path): # path: 
                raise FileNotFoundError("Please download the Vosk model and place it in the working directory.")

            self.vosk_model = Model(speech_to_text_model_path)
            # vocabulary = '["Joachim", "Tür", "öffnen", "schließen", "Hallo", "Hilfe", "Danke", "Auf Wiedersehen", "Grimstad"]'
            self.recognizer = KaldiRecognizer(self.vosk_model, VOSK_RATE)

            with self._maybe_quiet_alsa():
                self.audio_interface = pyaudio.PyAudio()
                self.stream = self.audio_interface.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=VOSK_RATE, # this rate 
                    input=True,
                    input_device_index=MIC_DEVICE_INDEX,  # Add this
                    frames_per_buffer=8192
                )
            self.stream.start_stream()
            print("Voice Assistant initialized successfully.")
        else:
            print("Voice Assistant initialized in TTS-only mode.")

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

    def speak(self, text=""):
        
        """Converts text to speech and plays it. If a filename is passed, plays that audio.

        Args:
            text (str): Text to convert to speech or WAV filename to play.
        """
        print(f"Using Speak ...")
        if text.endswith(".wav"):
            self.play_wav(os.path.join(self.output_dir, text))
            print(f"Played WAV file: {text}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.wav"
            filepath = os.path.join(self.output_dir, filename)

            self.tts_engine.save_to_file(text, filepath)
            self.tts_engine.runAndWait()
            # Check if file exists and is a valid WAV before playing
            import time
            for _ in range(10):  # Wait up to 1 second for file to be written
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
        """Plays a WAV audio file using PyAudio.

        Args:
            filepath (str): The full path to the WAV file.
        """
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
                        output_device_index=output_device_index
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
                            ratecv_state
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

    def get_voice_input(self, timeout_sec=None):
        
        """Captures voice input from the user and converts it to lowercase text.

        Returns:
            str: Transcribed text from the user's speech.
        """
        
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
        """async voice recognition."""
        print(" Listening for speech...")
        return "Hi"
    
    def get_text_input(self):
        """captures text input asynchronously."""
        text = input("Type your query: ")
        return text.strip().lower()
    
    def close(self):
        
        """Close open audio streams and terminate the audio interface."""
        if hasattr(self, "stream") and self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, "audio_interface") and self.audio_interface is not None:
            self.audio_interface.terminate()

_voice_assistant_instance = None


def get_voice_assistant(enable_listening=True):
    """Return a process-wide shared VoiceAssistant instance.

    The instance is created lazily on first call so importing this module
    does not initialize audio devices or Vosk.
    """
    global _voice_assistant_instance
    if _voice_assistant_instance is None:
        _voice_assistant_instance = VoiceAssistant(enable_listening=enable_listening)
    return _voice_assistant_instance

# if __name__ == "__main__":
#     assistant = VoiceAssistant(enable_listening=True)
#     try:
#         # only play wav file for testing
#         # assistant.speak("/home/ias/satya/catkin_ws/src/door_navigation/scripts/output/dog-bark.wav")
#         text_response = "Hello! I am your door navigation assistant. How can I help you today?"
#         assistant.speak(text_response)
#         # if assistant.recognizer is None:
#         #     print("Listening is disabled. Enable listening to use voice input.")
#         # else:
#         #     while True:
#         #         user_input = assistant.get_voice_input()
#         #         print(f"You said: {user_input}")
#         #         if "exit" in user_input or "quit" in user_input:
#         #             print("Exiting voice assistant.")
#         #             break
#         #         response = f"You said: {user_input}"
#         #         assistant.speak(response)

#     finally:
#         assistant.close()
    
 
