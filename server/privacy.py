def estimate_total_sample_size(responses: list[int], alpha: float, M: int) -> float:
    """
    Estimate the total sample size from privatized responses.

    Args:
        responses (list[int]): List of reported sample sizes from clients
        alpha (float): Probability of true response
        M (int): Privacy upper bound (clients report in [1, M-1])

    Returns:
        float: Estimated total sample size (n_tilde)
    """
    m = len(responses)
    total = sum(responses)
    adjustment = (1 - alpha) * (M * m / 2)
    n_tilde = (total - adjustment) / alpha
    return max(n_tilde, 1.0) # ensure positivity