"""Mathematical and numerical stability utilities for Bayesian mixture models."""

import numpy as np


def logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Compute the log of the sum of exponentials in a numerically stable manner.

    Why this is required:
        Direct calculation of sum(exp(a_i)) risks floating-point overflow when a_i >> 0
        or catastrophic underflow to 0.0 when all a_i << 0.
        The identity log(sum(exp(a))) = c + log(sum(exp(a - c))), where c = max(a),
        ensures that the largest exponent is 0 (exp(0) = 1), preventing overflow and
        preserving relative precision.

    Args:
        a: Input array of log-probabilities or logits.
        axis: Axis along which to compute the log-sum-exp.
        keepdims: Whether to keep the reduced dimension.

    Returns:
        The log-sum-exp result array.
    """
    max_val = np.amax(a, axis=axis, keepdims=True)
    
    # Check for -inf in max_val (e.g. all inputs were -inf)
    if not np.isfinite(max_val).all():
        max_val = np.where(np.isneginf(max_val), 0.0, max_val)

    tmp = np.exp(a - max_val)
    sum_val = np.sum(tmp, axis=axis, keepdims=keepdims)
    out = np.log(sum_val)

    if not keepdims:
        max_val = np.squeeze(max_val, axis=axis)
    
    return out + max_val


def stick_breaking_numpy(v: np.ndarray) -> np.ndarray:
    """Compute Dirichlet Process mixture weights from Beta fractions via stick-breaking.

    Mathematical formulation:
        Given Beta draws v_k ~ Beta(1, alpha), the weights w_k are given by:
        w_1 = v_1
        w_k = v_k * prod_{j=1}^{k-1} (1 - v_j)   for k > 1
        
        To ensure sum(w_k) = 1 for a truncated approximation of size K, the final
        fraction v_K is typically set to 1.0 (consuming the entire remaining stick).

    Args:
        v: 1D array of shape (K,) or 2D array of shape (..., K) containing Beta fractions in [0, 1].

    Returns:
        Array of weights summing to 1 along the last axis.
    """
    v = np.asarray(v)
    # Remaining stick fractions: 1 - v
    remaining = 1.0 - v
    
    # Cumulative product of remaining stick before each index
    # We prepend 1 to the cumulative product along the last axis
    shape = list(v.shape)
    shape[-1] = 1
    ones = np.ones(shape, dtype=v.dtype)
    cum_remaining = np.cumprod(np.concatenate([ones, remaining[..., :-1]], axis=-1), axis=-1)
    
    weights = v * cum_remaining
    # Normalize last axis to guard against minor numerical drift
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights
