"""
evaluate_model.py

Standalone evaluation script for trained HDC classifier.
Run this after training to get comprehensive metrics.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add hdc directory to path
sys.path.append('hdc')

from hdc.config import *
from hdc.features import FeatureExtractor, get_image_transform
from hdc.projection import generate_random_projection, encode_hypervector
from hdc.classifier import HDCClassifier
from hdc.train import train
from hdc.evaluate import evaluate_classifier, print_metrics, plot_confusion_matrix


def quick_evaluate(data_dir):
    """Quick evaluation of HDC classifier on dataset."""
    
    print("=== HDC Classifier Evaluation ===")
    
    # Train classifier
    print("Training classifier...")
    classifier = train(data_dir)
    
    # Setup evaluation components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FeatureExtractor(INPUT_DIM).to(device).eval()
    projection_matrix = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)
    transform = get_image_transform()
    
    # Evaluate
    print("Evaluating...")
    metrics = evaluate_classifier(
        classifier, data_dir, feature_extractor,
        projection_matrix, transform, device
    )
    
    # Print results
    print_metrics(metrics)
    
    # Save confusion matrix
    try:
        plot_confusion_matrix(metrics, save_path="hdc_confusion_matrix.png")
    except:
        print("Could not save confusion matrix plot")
    
    return metrics


if __name__ == "__main__":
    data_dir = "hdc_dtaset"  # Update this path
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory '{data_dir}' not found!")
        print("Usage: python evaluate_model.py [dataset_path]")
        sys.exit(1)
    
    metrics = quick_evaluate(data_dir)
    
    print(f"\n🎯 Final Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"📊 Total samples evaluated: {len(metrics['predictions'])}")
    print(f"📁 Classes: {', '.join(metrics['class_names'])}")