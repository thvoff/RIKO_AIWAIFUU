import requests
### MUST START SERVERS FIRST USING START ALL SERVER SCRIPT
import time
import soundfile as sf 
import sounddevice as sd
import yaml
from pathlib import Path

# Load YAML config robustly relative to this file's location
ROOT_DIR = Path(__file__).resolve().parents[3]  # points to waifu_project/
CONFIG_PATH = ROOT_DIR / "character_config.yaml"

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    char_config = yaml.safe_load(f)


def resolve_rel(p):
    p = Path(p)
    if not p.is_absolute():
        parts = p.parts
        # Strip leading 'waifu_project' to avoid waifu_project/waifu_project duplication
        if parts and parts[0].lower() == 'waifu_project':
            p = Path(*parts[1:]) if len(parts) > 1 else Path('.')
        p = (ROOT_DIR / p).resolve()
    return str(p)

def play_audio(path):
    data, samplerate = sf.read(path)
    sd.play(data, samplerate)
    sd.wait()  # Wait until playback is finished

def sovits_gen(in_text, output_wav_pth = "output.wav"):
    url = "http://127.0.0.1:9880/tts"

    resolved_ref = resolve_rel(char_config['sovits_ping_config']['ref_audio_path'])
    if not Path(resolved_ref).exists():
        print(f"Error: ref_audio_path not found: {resolved_ref}")
        return None

    payload = {
        "text": in_text,
        "text_lang": char_config['sovits_ping_config']['text_lang'],
        "ref_audio_path": resolved_ref,
        "prompt_text": char_config['sovits_ping_config']['prompt_text'],
        "prompt_lang": char_config['sovits_ping_config']['prompt_lang'],
        "text_split_method": "cut0",
        "media_type": "wav",
        "streaming_mode": False
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"TTS API error {response.status_code}: {response.text}")
            response.raise_for_status()  # throws if not 200

        print(response)

        # Save the response audio if it's binary
        with open(output_wav_pth, "wb") as f:
            f.write(response.content)
        # print("Audio saved as output.wav")

        return output_wav_pth

    except Exception as e:
        print("Error in sovits_gen:", e)
        return None



if __name__ == "__main__":

    start_time = time.time()
    output_wav_pth1 = "output.wav"
    path_to_aud = sovits_gen("if you hear this, that means it is set up correctly", output_wav_pth1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(path_to_aud)


