"""
evaluate.py

Evaluation utilities for HDC classifier.
Reports accuracy, per-class accuracy, and confusion matrix.
"""

import os
import time
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

from config import *
from features import FeatureExtractor, get_image_transform
from projection import encode_hypervector


def evaluate_classifier(
    classifier,
    data_dir,
    feature_extractor,
    projection_matrix,
    transform,
    device
):
    """
    Evaluate an HDC classifier.
    """

    y_true, y_pred = [], []
    inference_times = []

    feature_extractor.eval()

    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img = Image.open(os.path.join(class_dir, img_file)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            start = time.time()

            with torch.no_grad():
                feat = feature_extractor(x).cpu().numpy().squeeze()

            hv = encode_hypervector(feat, projection_matrix)
            pred = classifier.predict(hv)

            end = time.time()

            inference_times.append(end - start)
            y_true.append(class_id)
            y_pred.append(pred)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_accuracy,
        "class_names": list(CLASS_MAP.keys()),
        "avg_inference_time_ms": np.mean(inference_times) * 1000
    }


def print_metrics(metrics):
    print("\n========== HDC EVALUATION RESULTS ==========")
    print(f"Overall Accuracy      : {metrics['accuracy']*100:.2f}%")
    print(f"Avg Inference Time    : {metrics['avg_inference_time_ms']:.2f} ms/image")

    print("\nPer-Class Accuracy:")
    for cls, acc in zip(metrics["class_names"], metrics["per_class_accuracy"]):
        print(f"  {cls:<10}: {acc*100:.2f}%")

    print("\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    print("True \\ Pred →", metrics["class_names"])
    for i, row in enumerate(cm):
        print(f"{metrics['class_names'][i]:<10} {row}")
