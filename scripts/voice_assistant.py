import asyncio
import pyttsx3
# from vosk import Model, KaldiRecognizer
# import pyaudio
import wave
import json
import os
from datetime import datetime
from gtts import gTTS
# import simpleaudio as sa

class VoiceAssistant:
    
    """Handles voice input/output operations for the assistant."""
    
    def __init__(self, config, logger):
        
        """Initializes the voice assistant and audio interfaces."""
        self.logger = logger
        self.config = config
        self.output_dir = self.config.OUTPUT_DIR
        # self.tts_engine = pyttsx3.init()
        
        # speech_to_text_model_path = self.config.SPEECH_RECOGNITION_MODEL_PATH + self.config.SPEECH_RECOGNITION_MODEL
        # if not os.path.exists(speech_to_text_model_path): # path: 
        #     raise FileNotFoundError("Please download the Vosk model and place it in the working directory.")

        # self.vosk_model = Model(speech_to_text_model_path)
        # self.recognizer = KaldiRecognizer(self.vosk_model, 16000)

        # self.audio_interface = pyaudio.PyAudio()
        # self.stream = self.audio_interface.open(
        #     format=pyaudio.paInt16,
        #     channels=1,
        #     rate=16000,
        #     input=True,
        #     input_device_index=self.config.MIC_DEVICE_INDEX,  # Add this
        #     frames_per_buffer=8192
        # )
        # self.stream.start_stream()
        # self.logger.info("Voice Assistant initialized successfully.")

    def speak(self, text=""):
        
        """Converts text to speech and plays it. If a filename is passed, plays that audio.

        Args:
            text (str): Text to convert to speech or WAV filename to play.
        """
        self.logger.info(f"Using Speak ...")
        if text.endswith(".wav"):
            self.play_wav_simple(os.path.join(self.output_dir, text))
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
                    self.play_wav_simple(filepath)
                except Exception as e:
                    print(f"[WARN] Could not play generated WAV: {e}")
                    print("[INFO] Falling back to gTTS for TTS.")
                    self.speak_gtts(text)
            else:
                print(f"[ERROR] TTS did not generate a valid WAV file: {filepath}")
                print("[INFO] Falling back to gTTS for TTS.")
                self.speak_gtts(text)

    def speak_gtts(self, text=""):
        """
        Converts text to speech and plays it using gTTS and simpleaudio.

        Args:
            text (str): Text to convert to speech.
        """
        self.logger.info("Using speak with gTTS ...") 
        if text.endswith(".wav"):
            self.play_wav_simple(os.path.join(self.output_dir, text))
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"response_{timestamp}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            tts = gTTS(text)
            tts.save(filepath)
            
            # Convert MP3 to WAV for simpleaudio (which only supports WAV)
            # this will only be used in Speak_gtts method as it takes mp3 as input
            wav_filepath = filepath.replace('.mp3', '.wav')
            try:
                # Use pydub to convert MP3 to WAV
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(filepath)
                audio.export(wav_filepath, format="wav")
                self.play_wav_simple(wav_filepath)
                # Clean up files
                os.remove(filepath)  # Remove MP3
                os.remove(wav_filepath)  # Remove WAV
            except ImportError:
                print("[ERROR] pydub not installed. Cannot convert MP3 to WAV for simpleaudio.")
                print("[INFO] Please install pydub: pip install pydub")
            except Exception as e:
                print(f"[ERROR] Could not play audio with simpleaudio: {e}")

    def play_wav_simple(self, filepath):
        """Plays a WAV audio file using simpleaudio.

        Args:
            filepath (str): The full path to the WAV file.
        """
        try:
            wave_obj = sa.WaveObject.from_wave_file(filepath)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until playback is finished
        except Exception as e:
            print(f"[ERROR] Could not play WAV file with simpleaudio: {e}")

    def play_wav(self, filepath):
        """Plays a WAV audio file using PyAudio.

        Args:
            filepath (str): The full path to the WAV file.
        """
        chunk = 1024
        wf = wave.open(filepath, 'rb')
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            output_device_index=self.config.SPEAKER_DEVICE_INDEX  # Add this
        )

        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def get_voice_input(self):
        
        """Captures voice input from the user and converts it to lowercase text.

        Returns:
            str: Transcribed text from the user's speech.
        """
        
        print("🎤 Listening... Please speak clearly.")
        while True:
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
        # loop = asyncio.get_running_loop()
        # text = await loop.run_in_executor(None, input, "Type your query: ")
        text = input("Type your query: ")
        return text.strip().lower()
    
    def close(self):
        
        """Close open audio streams and terminate the audio interface."""
        self.stream.stop_stream()
        self.stream.close()
        self.audio_interface.terminate()
