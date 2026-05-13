"""
TEST SCRIPT - Test your model on a single video
Usage: python test.py --video path/to/video.mp4
"""

import torch
import argparse
import numpy as np
from pathlib import Path
import cv2
import librosa
import subprocess
import tempfile

# Import from your model
from model import AudioVisualDeepfakeDetector

def extract_frames(video_path, num_frames=16, img_size=224):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3)))
    
    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames

def extract_audio_spectrogram(video_path, sr=16000, n_mels=128):
    """Extract audio spectrogram"""
    try:
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        subprocess.run([
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sr), '-ac', '1',
            temp_audio_path, '-y'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        
        audio, _ = librosa.load(temp_audio_path, sr=sr, duration=3.0)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        Path(temp_audio_path).unlink()
        return mel_spec_db
    except:
        return np.zeros((n_mels, 128))

def test_video(video_path, model_path):
    """Test model on a single video"""
    print("\n" + "="*70)
    print("TESTING VIDEO")
    print("="*70)
    
    # Check if files exist
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first: python train.py")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    model = AudioVisualDeepfakeDetector(num_frames=16)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Extract features
    print(f"\nAnalyzing: {video_path}")
    print("Extracting frames...")
    frames = extract_frames(video_path, num_frames=16, img_size=224)
    
    if frames is None:
        print("❌ Failed to extract frames")
        return
    
    print("Extracting audio...")
    audio_spec = extract_audio_spectrogram(video_path)
    
    # Convert to tensors
    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    audio_tensor = torch.from_numpy(audio_spec).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Predict
    print("Running inference...")
    with torch.no_grad():
        logits, _ = model(frames_tensor, audio_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0, prediction].item()
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Prediction: {'🔴 FAKE' if prediction == 1 else '🟢 REAL'}")
    print(f"Confidence: {confidence:.1%}")
    print(f"\nProbability Breakdown:")
    print(f"  Real: {probs[0, 0].item():.1%}")
    print(f"  Fake: {probs[0, 1].item():.1%}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test deepfake detection model')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to model')
    args = parser.parse_args()
    
    test_video(args.video, args.model)