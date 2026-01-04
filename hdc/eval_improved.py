"""
eval_improved.py

Improved evaluation with train/test split and augmented data.
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

from config import *
from features import FeatureExtractor, get_image_transform
from projection import generate_random_projection, encode_hypervector
from classifier import HDCClassifier


def train_hdc(train_dir):
    """Train HDC classifier on training data."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor(INPUT_DIM).to(device).eval()
    R = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
    classifier = HDCClassifier(num_classes=len(CLASS_MAP), hd_dim=HD_DIM)
    transform = get_image_transform()
    
    print("Training on augmented data...")
    
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        count = 0
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = feature_extractor(x).cpu().numpy().squeeze()
            
            h = encode_hypervector(feat, R)
            classifier.add_sample(class_id, h)
            count += 1
        
        print(f"  {class_name}: {count} training samples")
    
    return classifier, feature_extractor, R, transform, device


def evaluate_hdc(classifier, test_dir, feature_extractor, R, transform, device):
    """Evaluate HDC classifier on test data."""
    
    y_true = []
    y_pred = []
    inference_times = []
    
    print("Evaluating on test data...")
    
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(test_dir, class_name)
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
            
            hv = encode_hypervector(features, R)
            
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
    train_dir = r"D:\hdc-image-classification-main\hdc_split\train"
    test_dir = r"D:\hdc-image-classification-main\hdc_split\test"
    
    if not os.path.exists(train_dir):
        print("Train directory not found. Run augment_data.py and split_data.py first.")
        exit()
    
    classifier, feature_extractor, R, transform, device = train_hdc(train_dir)
    evaluate_hdc(classifier, test_dir, feature_extractor, R, transform, device)