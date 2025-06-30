import numpy as np
from torch.utils.data import Dataset, Subset
from typing import Union

def sample_uniform_data(dataset: Dataset, target_fraction: float, seed: Union[int, None] = None) -> Subset:
    """
    Uniformly subsample a dataset given a target fraction of data.

    Args:
        dataset (Dataset): PyTorch Dataset object
        target_fraction (float): Fraction to retain (e.g., n_star / n_tilde)
        seed (int or None): Optional random seed for reproducibility

    Returns:
        Subset: A torch.utils.data.Subset with sampled indices
    """
    n = len(dataset)
    sample_size = max(1, int(np.floor(n * target_fraction)))
    sample_size = min(sample_size, n)  # ensure we don't oversample
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=sample_size, replace=False)
    return Subset(dataset, indices)