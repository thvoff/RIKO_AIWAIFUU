# OpenAI tool calling with history 
### Uses a sample function
import yaml
import gradio as gr
import json
import os
import requests
from pathlib import Path

# Resolve project root and config path robustly regardless of CWD
WAIFU_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = WAIFU_ROOT / 'character_config.yaml'
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    char_config = yaml.safe_load(f)

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"  # Hardcoded to prevent env issues

class LLMResponse:
    def __init__(self, output_text):
        self.output_text = output_text

# Constants
HISTORY_FILE = (WAIFU_ROOT / char_config['history_file'])
MODEL = char_config['model']
SYSTEM_PROMPT =  [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": char_config['presets']['default']['system_prompt']  
                }
            ]
        }
    ]

# Load/save chat history
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return SYSTEM_PROMPT

def save_history(history):
    # Keep system prompt + last 10 turns (20 messages)
    if len(history) > 21:
        history = [history[0]] + history[-20:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)



def convert_history_to_ollama_messages(messages):
    conv = []
    for m in messages:
        role = m.get("role")
        content = ""
        for c in m.get("content", []):
            t = c.get("text")
            if t:
                content += t
        if role == "system":
            conv.append({"role": "system", "content": content})
        elif role == "user":
            conv.append({"role": "user", "content": content})
        elif role == "assistant":
            conv.append({"role": "assistant", "content": content})
    return conv


def get_riko_response_no_tool(messages):
    # Build Ollama chat payload from our stored history format
    chat_messages = convert_history_to_ollama_messages(messages)

    payload = {
        "model": MODEL,
        "messages": chat_messages,
        "stream": False,
        "options": {
            "temperature": 1.0,
            "top_p": 1.0
        }
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=300
        )
        data = resp.json()
        if resp.status_code != 200:
            return LLMResponse(output_text=f"Ollama API error {resp.status_code}: {data}")

        # Extract assistant message text
        txt = ""
        if isinstance(data, dict):
            msg = data.get("message") or {}
            if isinstance(msg, dict):
                txt = msg.get("content", "")
        if not txt:
            # Fallback to raw json if unexpected shape
            txt = json.dumps(data)

        return LLMResponse(output_text=txt)
    except Exception as e:
        return LLMResponse(output_text=f"Ollama request failed: {e}")


def llm_response(user_input):

    messages = load_history()

    # Append user message to memory
    messages.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": user_input}
        ]
    })


    riko_test_response = get_riko_response_no_tool(messages)


    # just append assistant message to regular response. 
    messages.append({
    "role": "assistant",
    "content": [
        {"type": "output_text", "text": riko_test_response.output_text}
    ]
    })

    save_history(messages)
    return riko_test_response.output_text


if __name__ == "__main__":
    print('running main')