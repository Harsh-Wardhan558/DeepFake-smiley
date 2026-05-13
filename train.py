"""
TRAINING SCRIPT FOR AUDIO-VISUAL DEEPFAKE DETECTOR
===================================================

This script trains your custom model from scratch.

USAGE:
    python train.py --data_dir dataset --epochs 30 --batch_size 4

QUICK START (3 days to project):
    1. Get 100-200 videos (50-100 real + 50-100 fake)
    2. Organize in dataset/train/real and dataset/train/fake
    3. Run: python train.py --epochs 20 --batch_size 2
    4. Wait 4-8 hours (on GPU)
    5. Model saved as checkpoints/best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import librosa
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import json
from datetime import datetime

# Import your model
from model import (
    AudioVisualDeepfakeDetector,
    extract_frames,
    extract_audio_spectrogram
)


# ============================================================
# DATASET CLASS
# ============================================================

class DeepfakeDataset(Dataset):
    """Dataset for loading videos with audio-visual features"""
    def __init__(self, video_paths, labels, num_frames=16, img_size=224, 
                 augment=False):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.augment = augment
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract frames
            frames = extract_frames(video_path, self.num_frames, self.img_size)
            if frames is None:
                frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3))
            
            # Extract audio
            audio_spec = extract_audio_spectrogram(video_path)
            
            # Convert to tensors
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            audio_spec = torch.from_numpy(audio_spec).unsqueeze(0).float()
            
            return frames, audio_spec, label
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # Return dummy data
            frames = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
            audio_spec = torch.zeros(1, 128, 128)
            return frames, audio_spec, label


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(data_dir, split='train'):
    """Load video paths and labels"""
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    video_paths = []
    labels = []
    
    # Load real videos
    real_dir = split_dir / 'real'
    if real_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov']:
            for video in real_dir.glob(ext):
                video_paths.append(str(video))
                labels.append(0)  # 0 = REAL
    
    # Load fake videos
    fake_dir = split_dir / 'fake'
    if fake_dir.exists():
        for ext in ['*.mp4', '*.avi', '*.mov']:
            for video in fake_dir.glob(ext):
                video_paths.append(str(video))
                labels.append(1)  # 1 = FAKE
    
    print(f"\n{split.upper()} SET:")
    print(f"  Total videos: {len(video_paths)}")
    print(f"  Real: {labels.count(0)}")
    print(f"  Fake: {labels.count(1)}")
    
    return video_paths, labels


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (frames, audio, labels) in enumerate(pbar):
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(frames, audio)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, labels in tqdm(val_loader, desc='Validating'):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            logits, _ = model(frames, audio)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'predictions': all_preds,
        'labels': all_labels
    }


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(args):
    """Main training function"""
    print("\n" + "="*70)
    print("STARTING TRAINING: AUDIO-VISUAL DEEPFAKE DETECTOR")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be very slow!")
        print("   Recommend using Google Colab with GPU")
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load dataset
    print("\n" + "-"*70)
    print("LOADING DATASET")
    print("-"*70)
    
    train_videos, train_labels = load_dataset(args.data_dir, 'train')
    val_videos, val_labels = load_dataset(args.data_dir, 'val')
    
    if len(train_videos) == 0:
        print("\n❌ ERROR: No training data found!")
        print("Please organize your dataset as:")
        print("  dataset/")
        print("    ├── train/")
        print("    │   ├── real/")
        print("    │   └── fake/")
        print("    └── val/")
        print("        ├── real/")
        print("        └── fake/")
        return
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        train_videos, train_labels,
        num_frames=args.num_frames,
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = DeepfakeDataset(
        val_videos, val_labels,
        num_frames=args.num_frames,
        img_size=args.img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("\n" + "-"*70)
    print("INITIALIZING MODEL")
    print("-"*70)
    
    model = AudioVisualDeepfakeDetector(num_frames=args.num_frames)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5,
    )
    
    # Training loop
    print("\n" + "-"*70)
    print("TRAINING")
    print("-"*70)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    start_time = datetime.now()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_results = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_results['accuracy'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_precision'].append(val_results['precision'])
        history['val_recall'].append(val_results['recall'])
        history['val_f1'].append(val_results['f1'])
        
        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_results['loss']:.4f} | Val Acc:   {val_results['accuracy']:.4f}")
        print(f"  Val Precision: {val_results['precision']:.4f} | Val Recall: {val_results['recall']:.4f}")
        print(f"  Val F1: {val_results['f1']:.4f}")
        
        # Save best model
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_results['accuracy'],
                'val_f1': val_results['f1'],
            }, 'checkpoints/best_model.pth')
            print(f"  ✓ Best model saved! (Acc: {val_results['accuracy']:.4f})")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
    
    # Training complete
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Training Time: {training_time}")
    
    # Save training history
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Create confusion matrix
    plot_confusion_matrix(val_results['labels'], val_results['predictions'])
    
    print("\nResults saved to 'results/' directory")
    print("Best model saved to 'checkpoints/best_model.pth'")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_curves(history):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    axes[1, 0].plot(history['val_precision'], label='Precision', linewidth=2)
    axes[1, 0].plot(history['val_recall'], label='Recall', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='F1-Score', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overfitting check
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].set_title('Overfitting Check (Train - Val Accuracy)')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    print("Training curves saved to 'results/training_curves.png'")


def plot_confusion_matrix(labels, predictions):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to 'results/confusion_matrix.png'")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Audio-Visual Deepfake Detector')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract from each video')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for frames')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    train(args)