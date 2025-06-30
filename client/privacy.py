import numpy as np

def clip_and_fake(n_k: int, M: int, alpha: float) -> int:
    """
    Locally privatize a client's sample size using randomized response.

    Args:
        n_k (int): True sample size
        M (int): Privacy budget upper bound (client reports <= M-1)
        alpha (float): Probability of reporting the true (clipped) value

    Returns:
        int: Reported value r_k
    """
    n_c = min(n_k, M - 1)
    fake = np.random.randint(1, M)
    x = np.random.binomial(1, alpha)
    return x * n_c + (1 - x) * fake