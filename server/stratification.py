import numpy as np
from typing import List, Dict

def stratify_clients(gradients: List[np.ndarray], H: int, max_iter: int = 100) -> Dict[int, List[int]]:
    """
    Cluster clients into H strata using custom client stratification algorithm (Algorithm 2).
    
    Args:
        gradients (List[np.ndarray]): Reconstructed gradients Z_k for each client (each shape (d,))
        H (int): Number of strata (groups)
        max_iter (int): Maximum number of iterations for convergence

    Returns:
        Dict[int, List[int]]: Dictionary mapping stratum index to list of client indices
    """

    N = len(gradients)
    d =  gradients[0].shape[0]

    # Step 1: Initialize mu_i as H random client gradients
    rng = np.random.default_rng(seed=42)
    initial_indices = rng.choice(N, size=H, replace=False)
    mu = [gradients[i].copy() for i in initial_indices]

    # Initialize clusters
    strata = {h: [] for h in range(H)}
    converged = False
    iter_count = 0

    while not converged and iter_count < max_iter:
        iter_count += 1

        # Clear current clusters
        strata = {h: [] for h in range(H)}

        # Step 2: Assign each client to nearest cluster
        for k, Z_k in enumerate(gradients):
            distances = [np.linalg.norm(Z_k - mu_i) for mu_i in mu]
            h = int(np.argmin(distances)) # nearest stratum index
            strata[h].append(k)

        # Step 3: Update centers
        new_mu = []
        for h in range(H):
            if strata[h]: # avoid empty strata
                Z_h = np.stack([gradients[k] for k in strata[h]])
                new_mu_h = np.mean(Z_h, axis=0)
                new_mu.append(new_mu_h)
            else:
                # reinitialize to a random client if empty
                new_mu.append(gradients[rng.choice(N)].copy())

        # Step 4: Convergence check
        converged = all(np.allclose(mu[h], new_mu[h]) for h in range(H))
        mu = new_mu
    
    return strata