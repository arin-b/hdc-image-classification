"""
simple_evaluate.py

Simple evaluation script with only 5 metrics:
- Overall Accuracy
- Per-Class Accuracy  
- Confusion Matrix
- Avg Inference Time (ms)
- Model Size (KB)
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('hdc')

from hdc.config import *
from hdc.features import FeatureExtractor, get_image_transform
from hdc.projection import generate_random_projection, encode_hypervector
from hdc.train import train


def evaluate_simple(data_dir):
    """Simple evaluation with 5 metrics only."""
    
    print("Training classifier...")
    classifier = train(data_dir)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor(INPUT_DIM).to(device).eval()
    projection_matrix = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
    transform = get_image_transform()
    
    y_true = []
    y_pred = []
    inference_times = []
    
    print("Evaluating...")
    
    # Test on all images
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)
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
            inference_time = (time.time() - start_time) * 1000
            
            y_true.append(class_id)
            y_pred.append(pred)
            inference_times.append(inference_time)
    
    # Calculate metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    avg_inference_time = np.mean(inference_times)
    model_size_kb = classifier.centroids.nbytes / 1024
    
    # Print results
    print("\n" + "="*40)
    print("HDC EVALUATION RESULTS")
    print("="*40)
    
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    print("\nPer-Class Accuracy:")
    class_names = list(CLASS_MAP.keys())
    for i, class_name in enumerate(class_names):
        acc = per_class_accuracy[i]
        print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    print(f"{'':>8}", end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, name in enumerate(class_names):
        print(f"{name:>8}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    print(f"\nAvg Inference Time: {avg_inference_time:.2f} ms")
    print(f"Model Size: {model_size_kb:.2f} KB")


if __name__ == "__main__":
    data_dir = "../hdc_dtaset"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found!")
        sys.exit(1)
    
    evaluate_simple(data_dir)