"""
evaluate.py

Minimal evaluation utilities for HDC classifier.
"""

import os
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

from config import *
from features import FeatureExtractor, get_image_transform
from projection import generate_random_projection, encode_hypervector


def evaluate_classifier(classifier, test_data_dir, feature_extractor, projection_matrix, transform, device):
    """Evaluate HDC classifier with minimal metrics."""
    
    y_true = []
    y_pred = []
    inference_times = []
    class_names = list(CLASS_MAP.keys())
    
    # Collect predictions
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(test_data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = feature_extractor(x).cpu().numpy().squeeze()
            
            hv = encode_hypervector(features, projection_matrix)
            
            # Time inference
            start_time = time.time()
            pred = classifier.predict(hv)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            y_true.append(class_id)
            y_pred.append(pred)
            inference_times.append(inference_time)
    
    # Calculate metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Model size
    model_size = classifier.centroids.nbytes / 1024  # KB
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm,
        'avg_inference_time': np.mean(inference_times),
        'model_size_kb': model_size,
        'class_names': class_names
    }


def print_metrics(metrics):
    """Print minimal evaluation metrics."""
    
    print("\n" + "="*40)
    print("HDC EVALUATION RESULTS")
    print("="*40)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(metrics['class_names']):
        acc = metrics['per_class_accuracy'][i]
        print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"{'':>8}", end="")
    for name in metrics['class_names']:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(metrics['class_names']):
        print(f"{name:>8}", end="")
        for j in range(len(metrics['class_names'])):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    print(f"\nAvg Inference Time: {metrics['avg_inference_time']:.2f} ms")
    print(f"Model Size: {metrics['model_size_kb']:.2f} KB")