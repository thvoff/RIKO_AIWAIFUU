# RIKO_AIWAIFUU
Rikho Project Clone

Overview
This project is a voice cloning application built using the SoVITS framework, designed to generate high-quality synthetic voices. It includes a client-server architecture for processing and interaction, with support for character-based voice customization.

Features
Voice synthesis using SoVITS model (v3lora-20250228).
Client-server setup for distributed processing.
Customizable character voices via configuration files.
Audio sample generation and playback.
Project Structure


.qodo, .venv, .vscode: Development tool configurations.
GPT-SoVITS-v3lora-20250228: SoVITS model and dataset.
waifu_project: Main project folder.
audio: Contains input/output audio files (e.g., main_sample.wav, output.wav).
character_files: Character-related data.
__pycache__: Python bytecode cache.
client, server: Client and server code.
process/main_chat.py: Core processing script.
character_config.yaml: Character settings.
chat_history.json: Conversation histoY
extra-req.txt: Additional requirements.
install_reqs.sh: Dependency installation script.
requirements.txt, requirements_windows.txt: Python dependencies.
backup.txt: Backup data.



Installation
Clone the repository: git clone <repository-URL>
Navigate to the project folder: cd Rikho_PROJECT_CLONE
Install dependencies: ./install_reqs.sh or pip install -r requirements.txt
Run the main script: python waifu_project/process/main_chat.py

Usage
Configure character settings in character_config.yaml.
Use main_sample.wav as a reference audio.
Generated output is saved as output.wav.
Contributing
Feel free to fork and submit pull requests!

License
MIT

[Add license if applicable, e.g., MIT]
