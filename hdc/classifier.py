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

import numpy as np

class HDCClassifier:
    """
    Hyperdimensional classifier based on centroid accumulation.

    Each class k is represented by a centroid:
        C_k = sum_{x in class k} h(x)

    where h(x) is a bipolar hypervector.
    """

    def __init__(self, num_classes, hd_dim):
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        
        # dtype=int32 is used because:
        # - individual hypervectors are int8
        # - accumulation may involve hundreds or thousands of additions
        # - int8 would overflow
        self.centroids = np.zeros(
            (num_classes, hd_dim), dtype=np.int32
        )

    def add_sample(self, class_id, hypervector):
        """
        Update the centroid of a class using a single sample.

        Learning rule:
            C_class_id ← C_class_id + h(x)

        Parameters
        ----------
        class_id : int
            Integer ID of the class (0 ... num_classes-1).
        hypervector : np.ndarray
            Bipolar hypervector of shape (hd_dim,).
        """

        self.centroids[class_id] += hypervector

    def predict(self, hypervector):
        """
        Predict the class of a query hypervector.

        Inference rule:
            ŷ = argmax_k cosine(h, C_k)

        Cosine similarity is used instead of raw dot product
        to remove bias from different centroid magnitudes
        (which depend on number of training samples).

        Parameters
        ----------
        hypervector : np.ndarray
            Bipolar query hypervector of shape (hd_dim,).

        Returns
        -------
        int
            Predicted class ID.
        """

        # Norm of query hypervector (constant for bipolar vectors,
        # but included for clarity and numerical stability)
        hv_norm = np.linalg.norm(hypervector) + 1e-8
        centroid_norms = np.linalg.norm(self.centroids, axis=1) + 1e-8

        # Compute cosine similarity for all classes at once
        scores = (self.centroids @ hypervector) / (
            centroid_norms * hv_norm
        )

        # Return class with maximum similarity
        return int(np.argmax(scores))
