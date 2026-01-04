"""
compression.py

Implements optional Hyperdimensional Computing (HDC) compression
using Holographic Reduced Representations (HRR).

Purpose:
- Combine multiple class centroids into a single hypervector
- Enable extremely low-bandwidth transmission (e.g., federated learning)
- Approximate recovery of individual centroids when needed

IMPORTANT:
- Compression is OPTIONAL.
- It is NOT required for local inference.
- It is mainly used for communication efficiency.
"""

import numpy as np


def generate_class_keys(num_classes, hd_dim, seed):
    """
    Generates random bipolar keys for each class.
    """
    rng = np.random.default_rng(seed)
    keys = {
        k: rng.choice([-1, 1], size=hd_dim).astype(np.int8)
        for k in range(num_classes)
    }
    return keys



def circular_convolution(a, b):
    """
    Perform circular convolution between two vectors.

    Circular convolution is the binding operation used in
    Holographic Reduced Representations (HRR).

    Mathematically:
        a ⊛ b = IFFT( FFT(a) ⊙ FFT(b) )

    Properties:
    - Approximately invertible
    - Distributes information across all dimensions
    - Noise increases with number of superposed items

    Parameters
    ----------
    a, b : np.ndarray
        Input vectors of equal length.

    Returns
    -------
    np.ndarray
        Result of circular convolution (real-valued).
    """

    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    return np.real(np.fft.ifft(fa * fb))


def compress_centroids(centroids, class_keys):
    """
    Compress all class centroids into a single hypervector.

    Compression rule:
        W = Σ_k ( K_k ⊛ C_k )

    Where:
    - C_k is the centroid of class k
    - K_k is the random key for class k
    - ⊛ denotes circular convolution

    Parameters
    ----------
    centroids : np.ndarray
        Array of shape (num_classes, hd_dim) containing class centroids.
    class_keys : dict[int, np.ndarray]
        Dictionary mapping class_id -> bipolar key.

    Returns
    -------
    W : np.ndarray
        Compressed hypervector containing all class information.
    """

    hd_dim = centroids.shape[1]
    W = np.zeros(hd_dim)

    for k in range(centroids.shape[0]):
        Ck = centroids[k]
        bound = circular_convolution(class_keys[k], Ck)
        W += bound

    return W


def decompress_centroid(W, class_key):
    """
    Approximate recovery of a single class centroid from
    the compressed hypervector.

    Recovery rule:
        Ĉ_k ≈ W ⊛ K_k^{-1}

    For HRR, the inverse of a key is its reversed vector.

    NOTE:
    - Recovery is approximate.
    - Noise increases with number of classes.
    - Accuracy improves with higher hd_dim.

    Parameters
    ----------
    W : np.ndarray
        Compressed hypervector.
    class_key : np.ndarray
        Bipolar key corresponding to the desired class.

    Returns
    -------
    np.ndarray
        Approximate recovered centroid.
    """

    # Inverse of HRR key = reversed vector
    key_inv = class_key[::-1]
    return circular_convolution(W, key_inv)
