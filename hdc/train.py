"""
train.py

End-to-end training pipeline for Hyperdimensional Computing (HDC)
based image classification.

This script:
1. Loads images from a folder-structured dataset
2. Extracts frozen CNN features (MobileNetV2-based)
3. Projects features into a high-dimensional bipolar space
4. Learns one centroid per class via simple accumulation
5. (Optionally) compresses centroids for federated / low-bandwidth use

IMPORTANT:
- There is NO backpropagation here.
- Learning = vector addition.
- This file is dataset-agnostic and intended for deployment / edge use.
"""

import os
import torch
import numpy as np
from PIL import Image

from config import *
from features import FeatureExtractor, get_image_transform
from projection import generate_random_projection, encode_hypervector
from classifier import HDCClassifier
from compression import generate_class_keys, compress_centroids


def train(data_dir):
    """
    Train an HDC classifier on an image dataset stored on disk.

    Expected dataset structure:
        data_dir/
        ├── class_name_0/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── class_name_1/
        │   └── ...
        └── ...

    The mapping from class_name -> class_id is defined in config.py
    via CLASS_MAP.

    Parameters
    ----------
    data_dir : str
        Path to the root dataset directory.

    Returns
    -------
    classifier : HDCClassifier
        Trained HDC classifier containing one centroid per class.
    """


    # DEVICE SELECTION
    # Use GPU if available (mostly affects CNN feature extraction which is not too expensive as it is).
    # HDC operations themselves are NumPy-based and lightweight.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # FEATURE EXTRACTOR
    # MobileNetV2-based feature extractor that outputs INPUT_DIM features.
    # All parameters are frozen: this is NOT trained.
    feature_extractor = FeatureExtractor(INPUT_DIM)
    feature_extractor.to(device).eval()


    # RANDOM MATRIX GENERATOR
    ## Fixed bipolar random matrix R ∈ {-1, +1}^{HD_DIM × INPUT_DIM}.
    # This MUST be fixed across all samples and all clients.
    R = generate_random_projection(INPUT_DIM, HD_DIM, RANDOM_SEED)


    # INITIALIZE HDC CLASSIFIER
    # Creates one centroid per class.
    # Centroids are stored as int32 to safely accumulate many samples.
    classifier = HDCClassifier(
                    num_classes=len(CLASS_MAP),
                    hd_dim=HD_DIM
                 )


    # IMAGE PREPROCESSING (to input into MobileNetV2)
    transform = get_image_transform()


    # TRAINING LOOP (no epochs)
    # We loop once over all images.
    # Learning = addition of hypervectors into class centroids.
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)

        for img_file in os.listdir(class_dir):
            img = Image.open(
                os.path.join(class_dir, img_file)
            ).convert("RGB")

            x = transform(img).unsqueeze(0).to(device)

            # FEATURE EXTRACTION
            # Forward pass through frozen CNN.
            # Output shape: (1, INPUT_DIM)
            with torch.no_grad():
                feat = feature_extractor(x)

            feat = feat.cpu().numpy().squeeze()

            # HDC ENCODING
            # Encode features into a bipolar hypervector:
            #   h(x) = sign(R @ feat)
            h = encode_hypervector(feat, R)

            # LEARNING STEP (ACCUMULATION)
            # Add the hypervector to the centroid of its class.
            classifier.add_sample(class_id, h)

    return classifier


if __name__ == "__main__":
    clf = train(r"D:\hdc-image-classification-main\hdc_dtaset")


    # OPTIONAL: HDC compression for federated upload
    # This step is used when the centroids need to be
    # transmitted over a low-bandwidth link (e.g., federated learning).
    # It is NOT required for local inference.
    class_keys = generate_class_keys(
        num_classes=len(CLASS_MAP),
        hd_dim=HD_DIM,
        seed=RANDOM_SEED
    )

      
    # Compress all class centroids into a single hypervector
    compressed = compress_centroids(
        clf.centroids,
        class_keys
    )

    print("Training complete.")
    print("Compressed vector shape:", compressed.shape)
