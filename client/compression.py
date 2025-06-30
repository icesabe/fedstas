import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple

def compress_gradient(gradient: np.ndarray, d_prime: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compress a 1D gradient vector using the Information Squeeze algorithm (custom KMeans).

    Args:
        gradient (np.ndarray): The original gradient vector of shape (d,)
        d_prime (int): Number of groups (compressed dimensions)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - centroids: array of shape (d_prime,) — the compressed values
            - indices: array of shape (d,) — cluster assignment for each dimension
    """
    d = len(gradient)
    assert d_prime <= d, "d_prime must be less than or equal to the gradient dimension"

    # Step 1: Randomly initialize centroids (choose d_prime unique entries)
    rng = np.random.default_rng(seed=42)
    initial_idx = rng.choice(d, size=d_prime, replace=False)
    centroids = gradient[initial_idx].copy()

    indices = np.zeros(d, dtype=int)
    converged = False

    while not converged:
        # Step 2: Assign each g_i to the nearest centroid
        groups = {j: [] for j in range(d_prime)}
        for i in range(d):
            distances = np.abs(gradient[i] - centroids) # scalar distances
            cluster_id = np.argmin(distances)
            indices[i] = cluster_id
            groups[cluster_id].append(gradient[i])

        # Step 3: Update centroids
        new_centroids = centroids.copy()
        for j in range(d_prime):
            if groups[j]:  # avoid empty clusters
                new_centroids[j] = np.mean(groups[j])

        # Step 4: Check convergence
        converged = np.allclose(new_centroids, centroids)
        centroids = new_centroids
    
    return centroids, indices

def decompress_gradient(centroids: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Reconstruct the gradient vector from centroids and cluster assignments.

    Args:
        centroids (np.ndarray): Compressed gradient values of shape (d_prime,)
        indices (np.ndarray): Cluster assignment for each original dimension (d,)

    Returns:
        np.ndarray: Reconstructed gradient of shape (d,)
    """
    return centroids[indices]