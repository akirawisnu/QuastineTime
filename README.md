# QuastionTime 🎧📝
Free Qualitative tools for researcher - Fully Automated Multi-lingual Audio Transcriber and Natural Language Processing Tools using Whisper OpenAI Model

Local Audio Transcription & Text Analysis App (Standalone, No Admin Required)

QuastionTime is a local Flask-based desktop web app for audio transcription and downstream text analysis using OpenAI Whisper, NLTK, and Python.
It runs entirely offline, uses a bundled FFmpeg, and can be distributed as a single Windows .exe that requires no admin privileges.

✨ Features
🎙️ Audio transcription using OpenAI Whisper (CPU-based)
🧠 Text preprocessing and stopword handling (NLTK)

📊 Automatic generation of:
- Transcription text
- Summary files
- Statistics outputs

Local web interface (Flask), Fully offline after installation, No system FFmpeg installation required and Works on locked-down laptops (no admin access)

# Architecture Overview
User Browser
   ↓
Local Flask Server (127.0.0.1)
   ↓
Whisper Transcription Engine
   ↓
Text Processing (NLTK)
   ↓
Local Output Files (Audio / Summary / Statistics)


All components run locally on your machine. No cloud calls are made.

🚀 Running the App (End Users)
✅ Requirements

Windows 10 or newer (64-bit) (for standalone), or pure Python console and requirements installed

▶️ Run the Standalone EXE
- No admin rights required
- No Python installation required

Copy QuastionTime.exe to a writable folder, for example:
- Desktop
- Documents
- Downloads
- Double-click QuastionTime.exe
Open your browser and go to:
http://127.0.0.1:5000


That’s it. 🎉

📁 Where Files Are Stored

All runtime data is written to a user-writable location:

%LOCALAPPDATA%\QuastionTime\


Inside that folder you will find:

Audio\
Summary\
Statistics\
transcribe_text\


This ensures the app works even on restricted corporate or university laptops.

🛠️ Developer Setup (Python Source)
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/QuastionTime.git
cd QuastionTime

2️⃣ Create Virtual Environment
py -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install Dependencies
pip install -U pip setuptools wheel
pip install -r Materials/requirements_file.txt

4️⃣ Run from Source
python QuastionTime_standalone_ready.py


Then open:

http://127.0.0.1:5000

⚠️ Known Limitations

EXE size is large (expected due to Whisper + Torch)

First startup may take a few seconds

CPU-only Whisper (no GPU acceleration)

🔐 Security & Privacy

No internet connection required after build
No data is uploaded anywhere
All processing happens locally

Suitable for sensitive or confidential audio

🤝 Acknowledgements

OpenAI Whisper
PyTorch
FFmpeg
NLTK
Flask
PyInstaller

📬 Support

If you encounter issues:

Ensure you are using 127.0.0.1 and not a public IP
Ensure the app is located in a writable folder
Check console output if using a console-enabled build
