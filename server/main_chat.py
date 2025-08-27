from faster_whisper import WhisperModel
from process.asr_func.asr_push_to_talk import record_and_transcribe
from process.llm_funcs.llm_scr import llm_response
from process.tts_func.sovits_ping import sovits_gen, play_audio
from pathlib import Path
import os
import time
import uuid
import soundfile as sf

# Avatar TTS and VSeeFace integration
import pyttsx3
import threading
import math
import random
from pythonosc.udp_client import SimpleUDPClient


def get_wav_duration(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


class SimpleAvatarTTS:
    """Lightweight TTS integration for your existing chat"""

    def __init__(self, vseeface_ip="192.168.0.3", vseeface_port=39539):
        self.client = SimpleUDPClient(vseeface_ip, vseeface_port)
        self.tts_engine = pyttsx3.init()
        self.is_speaking = False

        # Configure TTS
        voices = self.tts_engine.getProperty('voices')
        if len(voices) > 1:
            # Try to use a female voice if available
            for voice in voices:
                name = getattr(voice, "name", "") or ""
                if 'female' in name.lower() or 'zira' in name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)

    def animate_speaking(self, text: str):
        """Simple mouth animation during speech"""
        words = text.split()
        duration = len(words) * 0.4
        start_time = time.time()
        mouth_shapes = ['A', 'I', 'U', 'E', 'O']
        try:
            while self.is_speaking and (time.time() - start_time) < duration + 1:
                # Random mouth movement
                shape = random.choice(mouth_shapes)
                intensity = random.uniform(0.4, 0.7)
                self.client.send_message("/VMC/Ext/Blend/Val", [shape, intensity])

                # Slight head movement
                head_y = math.sin(time.time() * 6.0) * 0.08
                self.client.send_message("/VMC/Ext/Bone/Pos", [
                    "Head", 0.0, 0.0, 0.0, 0.0, head_y, 0.0, 1.0
                ])
                time.sleep(0.08)
        finally:
            # Reset mouth and head
            for shape in mouth_shapes:
                self.client.send_message("/VMC/Ext/Blend/Val", [shape, 0.0])
            self.client.send_message("/VMC/Ext/Bone/Pos", [
                "Head", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
            ])

    def speak(self, text: str):
        """Make avatar speak the text"""
        if self.is_speaking or not text:
            return
        self.is_speaking = True

        # Start animation thread
        anim_thread = threading.Thread(target=self.animate_speaking, args=(text,), daemon=True)
        anim_thread.start()

        # Speak
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False


# Initialize avatar TTS
avatar_tts = SimpleAvatarTTS()


# Helper function to speak via avatar

def make_avatar_speak(response_text: str):
    """Make the avatar speak the AI response"""
    # Clean the text for better TTS
    clean_text = response_text.replace('*', '').replace('\n', ' ').strip()
    if not clean_text:
        return

    print(f"Avatar speaking: {clean_text[:100]}...")

    # Set happy expression
    avatar_tts.client.send_message("/VMC/Ext/Blend/Val", ["Joy", 0.5])
    time.sleep(0.3)

    # Speak with animation
    avatar_tts.speak(clean_text)

    # Reset expression
    time.sleep(0.5)
    avatar_tts.client.send_message("/VMC/Ext/Blend/Val", ["Joy", 0.0])


print(' \n ========= Starting Chat... ================ \n')
whisper_model = WhisperModel("base.en", device="cpu", compute_type="float32")

while True:

    conversation_recording = output_wav_path = Path("audio") / "conversation.wav"
    conversation_recording.parent.mkdir(parents=True, exist_ok=True)

    user_spoken_text = record_and_transcribe(whisper_model, conversation_recording)

    # pass to LLM and get a LLM output.
    llm_output = llm_response(user_spoken_text)

    if "Ollama request failed" in llm_output:
        print(f"Error from LLM: {llm_output}")
        continue

    tts_read_text = llm_output

    # Print AI response and speak via avatar
    print(f"AI: {tts_read_text}")
    make_avatar_speak(tts_read_text)

    # file organization

    # 1. Generate a unique filename
    uid = uuid.uuid4().hex
    filename = f"output_{uid}.wav"
    output_wav_path = Path("audio") / filename
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # generate audio and save it to client/audio
    gen_aud_path = sovits_gen(tts_read_text, output_wav_path)

    if gen_aud_path and Path(gen_aud_path).exists():
        play_audio(gen_aud_path)

    # clean up audio files
    [fp.unlink() for fp in Path("audio").glob("*.wav") if fp.is_file()]
    # # Example
    # duration = get_wav_duration(output_wav_path)
    # print("waiting for audio to finish...")
    # time.sleep(duration)
