"""
features.py

This file defines the CNN-based feature extraction frontend.

Purpose:
- Convert raw images into compact, semantically meaningful
  feature vectors.
- These features are then fed into the HDC pipeline.

Design choice:
- The CNN is FROZEN (no training).
- All learning happens in the HDC space, not in the CNN.
"""


import torch
import torch.nn as nn
from torchvision import models, transforms


class FeatureExtractor(nn.Module):
    """
    Frozen CNN feature extractor based on MobileNetV2.

    Architecture:
    - MobileNetV2 convolutional backbone (pretrained on ImageNet)
    - Global average pooling
    - Linear projection to a fixed output dimension

    IMPORTANT:
    - No parameters in this module are trained.
    - This acts purely as a feature encoder.
    """

    def __init__(self, output_dim: int):
        super().__init__()

        backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")
       
        # Keep only the convolutional feature extractor
        self.features = backbone.features

        # Freeze all backbone parameters
        for p in self.features.parameters():
            p.requires_grad = False

        # Global average pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Linear projection layer to reduce dimensionality
        # Bias is disabled for simplicity and stability
        self.proj = nn.Linear(1280, output_dim, bias=False)

        # Freeze projection layers as well
        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the feature extractor.

        Input:
            x : torch.Tensor
                Shape (B, 3, H, W), where B is batch size.

        Output:
            features : torch.Tensor
                Shape (B, output_dim)

        Steps:
        1. CNN convolutional feature extraction
        2. Global average pooling
        3. Linear projection to output_dim
        """

        # Extract convolution features
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)

        # Project to desired feature dimension
        return self.proj(x)


def get_image_transform():
    """
    Return the image preprocessing pipeline compatible
    with MobileNetV2.

    Operations:
    1. Resize image to 224×224
    2. Convert PIL image to PyTorch tensor
    3. Normalize using ImageNet mean and std

    This transform MUST be consistent between
    training and inference.

    Returns
    -------
    transform : torchvision.transforms.Compose
        Image preprocessing pipeline.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
