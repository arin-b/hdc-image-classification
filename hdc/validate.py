"""
validate.py

Scientific validation and benchmarking script for the
Hyperdimensional Computing (HDC) image classification pipeline.

Purpose:
- Verify correctness of the HDC pipeline on a standard dataset
- Measure classification accuracy
- Inspect centroid separation (sanity check)
- Demonstrate expected statistical behavior

IMPORTANT DISTINCTION:
- train.py  → deployment / production pipeline
- validate.py → controlled experiment / benchmarking script

This file is intentionally self-contained and explicit.
"""

import torch
import numpy as np
import torchvision
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

# -----------------------------
# Project-specific imports
# -----------------------------
from features import FeatureExtractor, get_image_transform
from projection import generate_random_projection, encode_hypervector


# ============================================================
# Local HDC Classifier (self-contained for validation)
# ============================================================
class HDCClassifier:
    """
    Minimal HDC classifier used only for validation.

    Re-defined locally to:
    - keep this script self-contained
    - make experimental logic explicit
    - avoid dependency on production code paths
    """

    def __init__(self, num_classes, hd_dim):
        """
        Initialize classifier with fixed number of classes.

        Parameters
        ----------
        num_classes : int
            Number of classes in the experiment.
        hd_dim : int
            Hypervector dimensionality.
        """

        self.hd_dim = hd_dim
        self.num_classes = num_classes

        # Centroids are accumulated hypervectors
        self.centroids = np.zeros(
            (num_classes, hd_dim), dtype=np.int32
        )

        # Optional: track how many samples per class were seen
        self.counts = np.zeros(num_classes, dtype=np.int32)

    def add_sample(self, label, hv):
        """
        Add a single training example to the corresponding centroid.

        Learning rule:
            C_label ← C_label + h(x)
        """
        self.centroids[label] += hv
        self.counts[label] += 1

    def predict(self, hv):
        """
        Predict class label using cosine similarity.

        ŷ = argmax_k cos(hv, C_k)

        Query norm is omitted in denominator since it is constant
        for bipolar hypervectors and does not affect argmax.
        """

        # Dot product between query and each centroid
        dot_products = np.dot(self.centroids, hv)

        # Norms of centroids (depend on number of samples)
        centroid_norms = np.linalg.norm(
            self.centroids, axis=1
        ) + 1e-8

        # Cosine similarity scores
        similarities = dot_products / centroid_norms

        return np.argmax(similarities)


# ============================================================
# Experiment configuration
# ============================================================

# Two visually distinct CIFAR-10 classes
CLASSES = [0, 8]  # 0 = airplane, 8 = ship

HD_DIM = 4096            # Hypervector dimensionality
CNN_OUTPUT_DIM = 128     # Feature dimension from CNN
SAMPLES_TRAIN = 1000     # Samples per class for training
SAMPLES_TEST = 200       # Samples per class for testing
SEED = 42                # Random seed for reproducibility


def run_experiment():
    """
    Run the full validation experiment:
    1. Train HDC classifier
    2. Inspect centroid separation
    3. Evaluate test accuracy
    """

    print("--- STARTING HDC VALIDATION ---")
    print(f"Dimensions: {HD_DIM}")
    print(f"Samples per class: {SAMPLES_TRAIN}")
    print("Encoding: Bipolar (-1, +1)")

    # ========================================================
    # 1. Model and projection setup
    # ========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing frozen feature extractor...")
    model = FeatureExtractor(output_dim=CNN_OUTPUT_DIM)
    model.to(device).eval()

    print("Generating fixed random projection matrix...")
    proj_matrix = generate_random_projection(
        CNN_OUTPUT_DIM, HD_DIM, SEED
    )

    classifier = HDCClassifier(
        num_classes=len(CLASSES),
        hd_dim=HD_DIM
    )

    # ========================================================
    # 2. Dataset preparation (CIFAR-10)
    # ========================================================
    transform = get_image_transform()

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    def get_loader(dataset, limit_per_class):
        """
        Build a DataLoader that includes only selected classes
        and limits samples per class.
        """
        indices = []
        counts = {k: 0 for k in CLASSES}

        for idx, (_, label) in enumerate(dataset):
            if label in CLASSES and counts[label] < limit_per_class:
                indices.append(idx)
                counts[label] += 1

        return DataLoader(
            Subset(dataset, indices),
            batch_size=1,
            shuffle=True
        )

    train_loader = get_loader(train_set, SAMPLES_TRAIN)
    test_loader = get_loader(test_set, SAMPLES_TEST)

    # Map CIFAR labels → local contiguous IDs
    label_map = {
        original: local
        for local, original in enumerate(CLASSES)
    }

    # ========================================================
    # 3. Training phase (single pass)
    # ========================================================
    print(f"\n[Phase 1] Training on {len(train_loader)} images...")

    with torch.no_grad():
        for img, label in tqdm(train_loader):
            img = img.to(device)
            target = label_map[label.item()]

            # Feature extraction
            features = model(img).cpu().numpy().flatten()

            # HDC encoding
            hv = encode_hypervector(features, proj_matrix)

            # Learning by accumulation
            classifier.add_sample(target, hv)

    # ========================================================
    # 4. Centroid similarity diagnostics
    # ========================================================
    print("\n[Phase 2] Checking centroid separation...")

    c0 = classifier.centroids[0]
    c1 = classifier.centroids[1]

    sim = np.dot(c0, c1) / (
        np.linalg.norm(c0) * np.linalg.norm(c1)
    )

    print(f"Cosine similarity between centroids: {sim:.4f}")

    if sim < 0.2:
        print("✅ Centroids are well separated")
    else:
        print("⚠️ Centroids are too similar (expected with small N)")

    # ========================================================
    # 5. Testing phase
    # ========================================================
    print(f"\n[Phase 3] Testing on {len(test_loader)} images...")

    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            target = label_map[label.item()]

            features = model(img).cpu().numpy().flatten()
            hv = encode_hypervector(features, proj_matrix)

            pred = classifier.predict(hv)

            correct += int(pred == target)
            total += 1

    acc = (correct / total) * 100
    print(f"\nFinal Accuracy: {acc:.2f}%")

    if acc >= 75.0:
        print("🚀 PASSED: System behaves as expected")
    else:
        print("❌ FAILED: Accuracy below expected regime")


if __name__ == "__main__":
    run_experiment()

