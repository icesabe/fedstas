import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple

def compress_gradient(gradient: np.ndarray, d_prime: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress a gradient vector into d_prime cluster centroids using KMeans.

    Args:
        gradient (np.ndarray): The original gradient vector of shape (d,)
        d_prime (int): Number of clusters (compressed dimensions)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - centroids: array of shape (d_prime,)
            - indices: array of shape (d,) with cluster assignment for each dimension
    """
    d = gradient.shape[0]
    gradient = gradient.reshape(-1, 1) # shape: (d, 1)

    # Fit KMeans
    kmeans = KMeans(n_clusters=d_prime, n_init='auto', random_state=42)
    kmeans.fit(gradient)

    centroids = kmeans.cluster_centers_.flatten() # shape: (d_prime,)
    indices = kmeans.labels_ # shape: (d,)

    return centroids, indices

def compress_gradient(centroids: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Reconstruct the gradient vector from centroids and cluster assignments.

    Args:
        centroids (np.ndarray): Compressed gradient values of shape (d_prime,)
        indices (np.ndarray): Cluster assignment for each original dimension (d,)

    Returns:
        np.ndarray: Reconstructed gradient of shape (d,)
    """
    return centroids[indices]