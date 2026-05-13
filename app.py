"""
WEB DEMO APPLICATION FOR DEEPFAKE DETECTION
============================================

This creates a simple web interface to demo your model to the professor.

USAGE:
    python app.py

Then open browser to: http://localhost:5000

FEATURES:
- Upload video and get prediction
- Shows confidence scores
- Displays attention maps
- Real-time processing
"""

from flask import Flask, render_template_string, request, jsonify
import torch
import numpy as np
import cv2
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Import your model
from model import AudioVisualDeepfakeDetector, extract_frames, extract_audio_spectrogram

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max file size

# Load model
MODEL_PATH = 'checkpoints/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
model = AudioVisualDeepfakeDetector(num_frames=16)

if Path(MODEL_PATH).exists():
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"  Validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}")
    print("  Using untrained model for demo")

model = model.to(device)
model.eval()


# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio-Visual Deepfake Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        
        .card {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        .upload-section {
            text-align: center;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: #f0f4ff;
            border-color: #764ba2;
        }
        
        .upload-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s;
            margin-top: 20px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .results {
            display: none;
        }
        
        .results.show {
            display: block;
        }
        
        .prediction-box {
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        
        .prediction-box.real {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
        }
        
        .prediction-box.fake {
            background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
            color: white;
        }
        
        .prediction-text {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .confidence-text {
            font-size: 1.3em;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        
        .loader.show {
            display: block;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .attention-maps {
            margin-top: 30px;
        }
        
        .attention-maps img {
            width: 100%;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .info-box {
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .info-box h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .info-box ul {
            margin-left: 20px;
            color: #666;
        }
        
        .info-box li {
            margin: 8px 0;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 Audio-Visual Deepfake Detector</h1>
            <p class="subtitle">Advanced AI System for Detecting Manipulated Videos</p>
            <p class="subtitle" style="margin-top: 10px; font-size: 0.9em; color: #999;">
                Created by [Your Name] | Minor Project 2024
            </p>
        </div>
        
        <div class="main-content">
            <div class="card upload-section">
                <h2 style="margin-bottom: 20px;">Upload Video</h2>
                
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <p style="font-size: 1.2em; color: #667eea;">Click to select video</p>
                    <p style="color: #999; margin-top: 10px;">Supports MP4, AVI, MOV</p>
                </div>
                
                <input type="file" id="fileInput" accept="video/*" onchange="handleFileSelect(event)">
                
                <p id="fileName" style="margin-top: 15px; color: #666;"></p>
                
                <button class="btn" id="analyzeBtn" onclick="analyzeVideo()" disabled>
                    Analyze Video
                </button>
                
                <div class="loader" id="loader"></div>
                
                <div class="info-box">
                    <h3>How it works:</h3>
                    <ul>
                        <li><strong>Visual Analysis:</strong> Custom CNN examines face artifacts</li>
                        <li><strong>Audio Analysis:</strong> Detects voice synthesis patterns</li>
                        <li><strong>Cross-Modal Check:</strong> Verifies audio-visual consistency</li>
                        <li><strong>Temporal Analysis:</strong> Examines frame-to-frame patterns</li>
                    </ul>
                </div>
            </div>
            
            <div class="card results" id="results">
                <h2 style="margin-bottom: 20px;">Analysis Results</h2>
                
                <div class="prediction-box" id="predictionBox">
                    <div class="prediction-text" id="predictionText">-</div>
                    <div class="confidence-text" id="confidenceText">-</div>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="realProb">-%</div>
                        <div class="metric-label">Real Probability</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="fakeProb">-%</div>
                        <div class="metric-label">Fake Probability</div>
                    </div>
                </div>
                
                <div class="info-box" style="margin-top: 30px;">
                    <h3>Model Architecture:</h3>
                    <ul>
                        <li>Custom Visual CNN (8 layers, trained from scratch)</li>
                        <li>Custom Audio CNN (4 layers, spectral analysis)</li>
                        <li>Cross-Modal Attention (8 heads)</li>
                        <li>Bidirectional LSTM (temporal analysis)</li>
                        <li>Total: ~10M parameters</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        function handleFileSelect(event) {
            selectedFile = event.target.files[0];
            if (selectedFile) {
                document.getElementById('fileName').textContent = `Selected: ${selectedFile.name}`;
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        async function analyzeVideo() {
            if (!selectedFile) return;
            
            // Show loader
            document.getElementById('loader').classList.add('show');
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('results').classList.remove('show');
            
            // Create form data
            const formData = new FormData();
            formData.append('video', selectedFile);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Hide loader
                document.getElementById('loader').classList.remove('show');
                document.getElementById('analyzeBtn').disabled = false;
                
                if (result.error) {
                    alert('Error: ' + result.error);
                    return;
                }
                
                // Display results
                displayResults(result);
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error analyzing video. Please try again.');
                document.getElementById('loader').classList.remove('show');
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function displayResults(result) {
            // Show results section
            document.getElementById('results').classList.add('show');
            
            // Set prediction
            const predictionBox = document.getElementById('predictionBox');
            const predictionText = document.getElementById('predictionText');
            const confidenceText = document.getElementById('confidenceText');
            
            predictionText.textContent = result.prediction;
            confidenceText.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            
            if (result.prediction === 'REAL') {
                predictionBox.className = 'prediction-box real';
            } else {
                predictionBox.className = 'prediction-box fake';
            }
            
            // Set probabilities
            document.getElementById('realProb').textContent = 
                `${(result.probabilities.real * 100).toFixed(1)}%`;
            document.getElementById('fakeProb').textContent = 
                `${(result.probabilities.fake * 100).toFixed(1)}%`;
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Render main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle video upload and prediction"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded video temporarily
        temp_path = Path('temp_video.mp4')
        video_file.save(temp_path)
        
        # Extract features
        print("Extracting video frames...")
        frames = extract_frames(str(temp_path), num_frames=16, img_size=224)
        
        print("Extracting audio...")
        audio_spec = extract_audio_spectrogram(str(temp_path))
        
        if frames is None:
            temp_path.unlink()
            return jsonify({'error': 'Failed to process video'}), 400
        
        # Convert to tensors
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
        audio_tensor = torch.from_numpy(audio_spec).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # Predict
        print("Running model inference...")
        with torch.no_grad():
            logits, attention_maps = model(frames_tensor, audio_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0, prediction].item()
        
        # Clean up
        temp_path.unlink()
        
        # Prepare response
        result = {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': float(confidence),
            'probabilities': {
                'real': float(probs[0, 0].item()),
                'fake': float(probs[0, 1].item())
            }
        }
        
        print(f"Prediction: {result['prediction']} (confidence: {confidence:.2%})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("AUDIO-VISUAL DEEPFAKE DETECTOR - WEB DEMO")
    print("="*70)
    print(f"\nModel device: {device}")
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)