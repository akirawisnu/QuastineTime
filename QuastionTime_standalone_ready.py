# -*- coding: utf-8 -*-
"""
QuastionTime - Qualitative Interview Transcriber
A Flask-based application for transcribing audio interviews with automatic summarization,
topic modeling, and word frequency analysis.

@author: Converted from Mic Drop! by akirawisnu
"""

from flask import Flask, render_template, jsonify, request, send_file
import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import whisper
from threading import Thread, Lock
import time
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import concurrent.futures
from functools import partial


def resource_path(relative_path: str) -> str:
    """Return absolute path to a bundled resource (PyInstaller) or local file (dev)."""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base_path, relative_path)


def runtime_base_dir() -> str:
    """Return a stable writable directory for outputs.

    In PyInstaller onefile mode, __file__ points to a temp folder, so we use the
    directory that contains the executable.
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

TEMPLATE_FOLDER = resource_path('templates')
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

# Configuration
SCRIPT_DIR = runtime_base_dir()
AUDIO_LIBRARY_PATH = os.path.join(SCRIPT_DIR, "Audio")
TRANSCRIPTION_PATH = os.path.join(SCRIPT_DIR, "transcribe_text")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "Summary")
STATISTICS_PATH = os.path.join(SCRIPT_DIR, "Statistics")
SUPPORTED_FORMATS = ['.mp3', '.flac', '.wav', '.m4a', '.ogg', '.webm']

# Ensure bundled FFmpeg is available (standalone ffmpeg.exe packaged with PyInstaller)
try:
    ffmpeg_dir = resource_path(os.path.join('ThirdParty', 'ffmpeg'))
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
except Exception as _e:
    # If anything goes wrong, we fall back to system PATH.
    pass


# Create directories
for path in [AUDIO_LIBRARY_PATH, TRANSCRIPTION_PATH, SUMMARY_PATH, STATISTICS_PATH]:
    os.makedirs(path, exist_ok=True)

# Global whisper model
whisper_model = None
model_lock = Lock()

# Download NLTK data
# Point NLTK to bundled data (optional). If nltk_data is packaged, this prevents runtime downloads.
try:
    nltk_data_dir = resource_path('nltk_data')
    if os.path.isdir(nltk_data_dir):
        os.environ['NLTK_DATA'] = nltk_data_dir
except Exception:
    pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Stop words for multiple languages
STOP_WORDS = {
    'en': set(stopwords.words('english')),
    'de': set(stopwords.words('german')),
    'id': {'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 'adalah', 'ini', 'itu', 'ada', 'akan', 'atau', 'juga', 'saya', 'kami', 'mereka'},
    'zh': {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'},
}

def get_audio_files(directory):
    """Scan directory for audio files"""
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS):
                file_path = os.path.join(root, file)
                audio_files.append({
                    'filename': file,
                    'path': file_path,
                    'size': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })
    return audio_files

def format_timestamp(seconds):
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def transcribe_audio(audio_path, language='auto', model_size='base'):
    """Transcribe audio using Whisper with timestamps"""
    global whisper_model
    
    try:
        print(f"\n{'='*60}")
        print(f"Starting transcription: {os.path.basename(audio_path)}")
        print(f"Language: {language}, Model: {model_size}")
        print(f"{'='*60}\n")
        
        with model_lock:
            if whisper_model is None or getattr(whisper_model, 'model_size', None) != model_size:
                print(f"Loading Whisper model ({model_size})...")
                whisper_model = whisper.load_model(model_size)
                whisper_model.model_size = model_size
                print("Model loaded successfully!\n")
        
        print("Transcribing audio...")
        result = whisper_model.transcribe(
            audio_path,
            language=None if language == 'auto' else language,
            verbose=True
        )
        
        segments = []
        full_text = []
        
        for i, segment in enumerate(result['segments']):
            segments.append({
                'index': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
            full_text.append(segment['text'].strip())
        
        detected_language = result.get('language', 'unknown')
        print(f"\nTranscription complete!")
        print(f"Detected language: {detected_language}")
        print(f"Total segments: {len(segments)}\n")
        
        return {
            'segments': segments,
            'full_text': ' '.join(full_text),
            'language': detected_language,
            'duration': result.get('duration', 0)
        }
    except Exception as e:
        print(f"Error transcribing: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_transcription_srt(transcription, output_path):
    """Save transcription in SRT format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for segment in transcription['segments']:
            f.write(f"{segment['index']}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")

def save_transcription_txt(transcription, output_path):
    """Save transcription as plain text with timestamps"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcription\n")
        f.write(f"Language: {transcription['language']}\n")
        f.write(f"Duration: {transcription['duration']:.2f} seconds\n")
        f.write(f"{'='*60}\n\n")
        
        for segment in transcription['segments']:
            timestamp = format_timestamp(segment['start'])
            f.write(f"[{timestamp}] {segment['text']}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f"Full Text:\n\n{transcription['full_text']}\n")

def generate_summary(text, max_sentences=5):
    """Generate extractive summary using TF-IDF"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = tfidf_matrix.sum(axis=1).A1
        
        # Get top sentences
        top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
        top_indices = sorted(top_indices)
        
        summary_sentences = [sentences[i] for i in top_indices]
        return '. '.join(summary_sentences) + '.'
    except:
        # Fallback: return first few sentences
        return '. '.join(sentences[:max_sentences]) + '.'

def extract_topics(text, language='en', n_topics=5):
    """Extract topics using LDA"""
    try:
        # Simple word tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove stop words
        stop_words_set = STOP_WORDS.get(language, STOP_WORDS['en'])
        words = [w for w in words if w not in stop_words_set]
        
        if len(words) < 10:
            return []
        
        # Create document-term matrix
        text_for_lda = ' '.join(words)
        vectorizer = TfidfVectorizer(max_features=100, max_df=0.8, min_df=1)
        doc_term_matrix = vectorizer.fit_transform([text_for_lda])
        
        # LDA
        lda = LatentDirichletAllocation(n_components=min(n_topics, len(vectorizer.get_feature_names_out())), random_state=42)
        lda.fit(doc_term_matrix)
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-5:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic': f'Topic {topic_idx + 1}',
                'words': ', '.join(top_words)
            })
        
        return topics
    except Exception as e:
        print(f"Topic extraction error: {e}")
        return []

def calculate_word_frequencies(text, language='en', top_n=50):
    """Calculate word frequencies excluding stop words"""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    stop_words_set = STOP_WORDS.get(language, STOP_WORDS['en'])
    words = [w for w in words if w not in stop_words_set]
    
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

def save_summary(transcription, filename, output_dir):
    """Save summary to file"""
    summary_path = os.path.join(output_dir, f"{filename}_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Summary for: {filename}\n")
        f.write(f"{'='*60}\n\n")
        
        summary = generate_summary(transcription['full_text'])
        f.write(f"Executive Summary:\n{summary}\n\n")
        
        topics = extract_topics(transcription['full_text'], transcription['language'])
        if topics:
            f.write(f"Key Topics:\n")
            for topic in topics:
                f.write(f"  - {topic['topic']}: {topic['words']}\n")
        
        f.write(f"\n{'='*60}\n")
        f.write(f"Language: {transcription['language']}\n")
        f.write(f"Duration: {transcription['duration']:.2f} seconds\n")

def save_statistics(transcription, filename, output_dir):
    """Save word frequency statistics"""
    stats_txt = os.path.join(output_dir, f"{filename}_stats.txt")
    stats_csv = os.path.join(output_dir, f"{filename}_stats.csv")
    
    word_freqs = calculate_word_frequencies(transcription['full_text'], transcription['language'])
    
    # Save as text
    with open(stats_txt, 'w', encoding='utf-8') as f:
        f.write(f"Word Frequency Statistics for: {filename}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total words (excluding stop words): {sum(count for _, count in word_freqs)}\n")
        f.write(f"Unique words: {len(word_freqs)}\n\n")
        f.write(f"Top {len(word_freqs)} Words:\n")
        f.write(f"{'-'*40}\n")
        for word, count in word_freqs:
            f.write(f"{word:.<30} {count:>5}\n")
    
    # Save as CSV
    df = pd.DataFrame(word_freqs, columns=['Word', 'Frequency'])
    df.to_csv(stats_csv, index=False, encoding='utf-8')

def process_single_audio(audio_info, language, model_size):
    """Process a single audio file"""
    filename = Path(audio_info['filename']).stem
    
    try:
        # Check if already transcribed
        srt_path = os.path.join(TRANSCRIPTION_PATH, f"{filename}.srt")
        if os.path.exists(srt_path):
            print(f"⏭️  Skipping {filename} (already transcribed)")
            return {'status': 'skipped', 'filename': filename, 'message': 'Already transcribed'}
        
        # Transcribe
        transcription = transcribe_audio(audio_info['path'], language, model_size)
        
        if not transcription:
            return {'status': 'error', 'filename': filename, 'message': 'Transcription failed'}
        
        # Save transcription
        save_transcription_srt(transcription, srt_path)
        save_transcription_txt(transcription, os.path.join(TRANSCRIPTION_PATH, f"{filename}.txt"))
        
        # Save summary
        save_summary(transcription, filename, SUMMARY_PATH)
        
        # Save statistics
        save_statistics(transcription, filename, STATISTICS_PATH)
        
        return {
            'status': 'success',
            'filename': filename,
            'language': transcription['language'],
            'duration': transcription['duration']
        }
    except Exception as e:
        return {'status': 'error', 'filename': filename, 'message': str(e)}

@app.route('/')
def index():
    return render_template('quastion.html')

@app.route('/api/audio-files')
def get_audio_files_route():
    """Get all audio files in library"""
    files = get_audio_files(AUDIO_LIBRARY_PATH)
    return jsonify(files)

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio file(s)"""
    data = request.json
    files = data.get('files', [])
    language = data.get('language', 'auto')
    model_size = data.get('model_size', 'base')
    batch_mode = data.get('batch', False)
    
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    results = []
    
    if batch_mode and len(files) > 1:
        # Parallel processing for batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            process_func = partial(process_single_audio, language=language, model_size=model_size)
            results = list(executor.map(process_func, files))
    else:
        # Sequential processing
        for audio_info in files:
            result = process_single_audio(audio_info, language, model_size)
            results.append(result)
    
    return jsonify({'results': results})

@app.route('/api/transcriptions')
def get_transcriptions():
    """Get list of completed transcriptions"""
    transcriptions = []
    for file in os.listdir(TRANSCRIPTION_PATH):
        if file.endswith('.srt'):
            file_path = os.path.join(TRANSCRIPTION_PATH, file)
            transcriptions.append({
                'filename': file,
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            })
    return jsonify(transcriptions)

@app.route('/api/download/<path:filepath>')
def download_file(filepath):
    """Download a file"""
    return send_file(filepath, as_attachment=True)

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuastionTime - Qualitative Interview Transcriber</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 3em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1024px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        .panel {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        
        .panel h2 {
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .control-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        label {
            font-weight: bold;
            min-width: 120px;
        }
        
        select, button {
            padding: 10px 15px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            cursor: pointer;
        }
        
        select {
            background: rgba(255,255,255,0.9);
            flex: 1;
        }
        
        button {
            background: #27ae60;
            color: white;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #229954;
            transform: translateY(-2px);
        }
        
        button:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-danger {
            background: #e74c3c;
        }
        
        .btn-danger:hover {
            background: #c0392b;
        }
        
        .file-list {
            max-height: 400px;
            overflow-y: auto;
            margin-top: 15px;
        }
        
        .file-item {
            background: rgba(255,255,255,0.15);
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
        }
        
        .file-item:hover {
            background: rgba(255,255,255,0.25);
        }
        
        .file-item.selected {
            background: rgba(52, 152, 219, 0.5);
            border: 2px solid #3498db;
        }
        
        .file-info {
            flex: 1;
        }
        
        .file-name {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .file-meta {
            font-size: 0.85em;
            opacity: 0.8;
        }
        
        .checkbox {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        
        .progress-container {
            margin-top: 20px;
            display: none;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .log-container {
            background: rgba(0,0,0,0.5);
            padding: 15px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }
        
        .log-entry {
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .log-success { color: #2ecc71; }
        .log-error { color: #e74c3c; }
        .log-info { color: #3498db; }
        .log-warning { color: #f39c12; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            opacity: 0.8;
        }
        
        .batch-controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 QuastionTime</h1>
            <p class="subtitle">Qualitative Interview Transcriber with AI Analysis</p>
        </header>
        
        <div class="main-grid">
            <div class="panel">
                <h2>🎙️ Audio Files</h2>
                
                <div class="controls">
                    <div class="control-row">
                        <label>Language:</label>
                        <select id="languageSelect">
                            <option value="auto" selected>Auto-detect</option>
                            <option value="en">English</option>
                            <option value="de">German</option>
                            <option value="id">Indonesian</option>
                            <option value="zh">Mandarin Chinese</option>
                            <option value="es">Spanish</option>
                            <option value="fr">French</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                        </select>
                    </div>
                    
                    <div class="control-row">
                        <label>Model Size:</label>
                        <select id="modelSelect">
                            <option value="tiny">Tiny (fastest)</option>
                            <option value="base" selected>Base (balanced)</option>
                            <option value="small">Small (better quality)</option>
                            <option value="medium">Medium (high quality)</option>
                        </select>
                    </div>
                </div>
                
                <div class="batch-controls">
                    <button onclick="selectAll()">Select All</button>
                    <button onclick="deselectAll()">Deselect All</button>
                    <button onclick="transcribeSelected()" id="transcribeBtn">Transcribe Selected</button>
                </div>
                
                <div class="file-list" id="fileList">
                    <p style="text-align: center; opacity: 0.7;">Loading audio files...</p>
                </div>
            </div>
            
            <div class="panel">
                <h2>⚙️ Processing Status</h2>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="totalFiles">0</div>
                        <div class="stat-label">Total Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="selectedFiles">0</div>
                        <div class="stat-label">Selected</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="completedFiles">0</div>
                        <div class="stat-label">Completed</div>
                    </div>
                </div>
                
                <div class="progress-container" id="progressContainer">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill">0%</div>
                    </div>
                </div>
                
                <div class="log-container" id="logContainer">
                    <div class="log-entry log-info">Ready to transcribe audio files...</div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>📄 Completed Transcriptions</h2>
            <div class="file-list" id="transcriptionList">
                <p style="text-align: center; opacity: 0.7;">No transcriptions yet...</p>
            </div>
        </div>
    </div>
    
    <script>
        let audioFiles = [];
        let selectedFiles = new Set();
        
        async function loadAudioFiles() {
            try {
                const response = await fetch('/api/audio-files');
                audioFiles = await response.json();
                displayAudioFiles();
                updateStats();
                loadTranscriptions();
            } catch (error) {
                addLog('Error loading audio files: ' + error.message, 'error');
            }
        }
        
        function displayAudioFiles() {
            const container = document.getElementById('fileList');
            
            if (audioFiles.length === 0) {
                container.innerHTML = '<p style="text-align: center; opacity: 0.7;">No audio files found in Audio folder</p>';
                return;
            }
            
            container.innerHTML = audioFiles.map((file, index) => `
                <div class="file-item" id="file-${index}">
                    <div class="file-info">
                        <div class="file-name">🎵 ${file.filename}</div>
                        <div class="file-meta">
                            Size: ${(file.size / 1024 / 1024).toFixed(2)} MB | 
                            Modified: ${new Date(file.modified).toLocaleDateString()}
                        </div>
                    </div>
                    <input type="checkbox" class="checkbox" 
                           onchange="toggleFileSelection(${index})" 
                           id="check-${index}">
                </div>
            `).join('');
        }
        
        function toggleFileSelection(index) {
            const checkbox = document.getElementById(`check-${index}`);
            const fileItem = document.getElementById(`file-${index}`);
            
            if (checkbox.checked) {
                selectedFiles.add(index);
                fileItem.classList.add('selected');
            } else {
                selectedFiles.delete(index);
                fileItem.classList.remove('selected');
            }
            
            updateStats();
        }
        
        function selectAll() {
            audioFiles.forEach((_, index) => {
                selectedFiles.add(index);
                document.getElementById(`check-${index}`).checked = true;
                document.getElementById(`file-${index}`).classList.add('selected');
            });
            updateStats();
        }
        
        function deselectAll() {
            selectedFiles.clear();
            audioFiles.forEach((_, index) => {
                document.getElementById(`check-${index}`).checked = false;
                document.getElementById(`file-${index}`).classList.remove('selected');
            });
            updateStats();
        }
        
        function updateStats() {
            document.getElementById('totalFiles').textContent = audioFiles.length;
            document.getElementById('selectedFiles').textContent = selectedFiles.size;
        }
        
        async function transcribeSelected() {
            if (selectedFiles.size === 0) {
                addLog('Please select at least one file to transcribe', 'warning');
                return;
            }
            
            const files = Array.from(selectedFiles).map(index => audioFiles[index]);
            const language = document.getElementById('languageSelect').value;
            const modelSize = document.getElementById('modelSelect').value;
            const batchMode = files.length > 1;
            
            document.getElementById('transcribeBtn').disabled = true;
            document.getElementById('progressContainer').style.display = 'block';
            
            addLog(`Starting transcription of ${files.length} file(s)...`, 'info');
            addLog(`Language: ${language}, Model: ${modelSize}, Batch mode: ${batchMode}`, 'info');
            
            try {
                const response = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        files: files,
                        language: language,
                        model_size: modelSize,
                        batch: batchMode
                    })
                });
                
                const data = await response.json();
                
                let completed = 0;
                data.results.forEach(result => {
                    if (result.status === 'success') {
                        completed++;
                        addLog(`✅ ${result.filename}: Transcribed (${result.language}, ${result.duration.toFixed(1)}s)`, 'success');
                    } else if (result.status === 'skipped') {
                        addLog(`⏭️ ${result.filename}: ${result.message}`, 'warning');
                    } else {
                        addLog(`❌ ${result.filename}: ${result.message}`, 'error');
                    }
                    
                    const progress = ((data.results.indexOf(result) + 1) / data.results.length) * 100;
                    updateProgress(progress);
                });
                
                document.getElementById('completedFiles').textContent = completed;
                addLog(`Batch processing complete! ${completed} files transcribed.`, 'success');
                
                // Reload transcriptions list
                loadTranscriptions();
                
            } catch (error) {
                addLog('Error during transcription: ' + error.message, 'error');
            } finally {
                document.getElementById('transcribeBtn').disabled = false;
                setTimeout(() => {
                    document.getElementById('progressContainer').style.display = 'none';
                    updateProgress(0);
                }, 3000);
            }
        }
        
        function updateProgress(percent) {
            const fill = document.getElementById('progressFill');
            fill.style.width = percent + '%';
            fill.textContent = Math.round(percent) + '%';
        }
        
        function addLog(message, type = 'info') {
            const container = document.getElementById('logContainer');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.textContent = `[${timestamp}] ${message}`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }
        
        async function loadTranscriptions() {
            try {
                const response = await fetch('/api/transcriptions');
                const transcriptions = await response.json();
                
                const container = document.getElementById('transcriptionList');
                
                if (transcriptions.length === 0) {
                    container.innerHTML = '<p style="text-align: center; opacity: 0.7;">No transcriptions yet...</p>';
                    return;
                }
                
                container.innerHTML = transcriptions.map(trans => {
                    const basename = trans.filename.replace('.srt', '');
                    return `
                        <div class="file-item">
                            <div class="file-info">
                                <div class="file-name">📝 ${trans.filename}</div>
                                <div class="file-meta">Created: ${new Date(trans.modified).toLocaleString()}</div>
                            </div>
                            <div style="display: flex; gap: 5px;">
                                <button onclick="downloadFile('transcribe_text/${trans.filename}')">SRT</button>
                                <button onclick="downloadFile('transcribe_text/${basename}.txt')">TXT</button>
                                <button onclick="downloadFile('Summary/${basename}_summary.txt')">Summary</button>
                                <button onclick="downloadFile('Statistics/${basename}_stats.csv')">CSV</button>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (error) {
                addLog('Error loading transcriptions: ' + error.message, 'error');
            }
        }
        
        function downloadFile(filepath) {
            window.open(`/api/download/${filepath}`, '_blank');
        }
        
        // Initialize
        loadAudioFiles();
        setInterval(loadTranscriptions, 10000); // Refresh every 10 seconds
    </script>
</body>
</html>
'''

# Ensure the HTML template exists for development runs.
# When packaged with PyInstaller, the template should be bundled via --add-data and loaded from TEMPLATE_FOLDER.
if not getattr(sys, 'frozen', False):
    os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
    template_file = os.path.join(TEMPLATE_FOLDER, 'quastion.html')
    if not os.path.exists(template_file):
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"QuastionTime - Qualitative Interview Transcriber")
    print(f"{'='*60}")
    print(f"Audio Library: {AUDIO_LIBRARY_PATH}")
    print(f"Transcriptions: {TRANSCRIPTION_PATH}")
    print(f"Summaries: {SUMMARY_PATH}")
    print(f"Statistics: {STATISTICS_PATH}")
    print(f"\n🌐 Browse to http://localhost:5000")
    print(f"{'='*60}\n")
    app.run(debug=True, host='0.0.0.0', port=5000)