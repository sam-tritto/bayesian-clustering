"""Evaluation metrics and cluster analysis utilities."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score


def filter_active_clusters(
    weights: np.ndarray,
    threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out empty/dead clusters below a specified mixture weight threshold.

    In a Dirichlet Process Mixture Model with a truncation limit K, components that do
    not receive support from the data have their posterior weights pushed toward 0 by
    the prior concentration parameter alpha.

    Args:
        weights: Mixture component weights summing to 1, shape (K,).
        threshold: Minimum weight threshold to consider a cluster active (e.g., 0.01).

    Returns:
        Tuple of (active_indices, normalized_active_weights).
    """
    weights = np.asarray(weights)
    active_indices = np.where(weights >= threshold)[0]
    raw_active_weights = weights[active_indices]
    normalized_weights = raw_active_weights / np.sum(raw_active_weights)
    return active_indices, normalized_weights


def evaluate_clustering(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray
) -> Dict[str, float]:
    """Compute standard clustering agreement metrics against ground truth.

    Metrics:
        - Adjusted Rand Index (ARI): Corrected-for-chance measure of agreement between clusterings.
          Range [-1, 1], with 1.0 being perfect match and 0.0 being chance.
        - Adjusted Mutual Information (AMI): Information-theoretic agreement measure adjusted for chance.
        - V-Measure: Harmonic mean of homogeneity and completeness.

    Args:
        true_labels: Ground truth class labels (integer or string).
        predicted_labels: Cluster assignments assigned by the model.

    Returns:
        Dictionary containing ARI, AMI, and V-measure scores.
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, predicted_labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(true_labels, predicted_labels)),
        "v_measure": float(v_measure_score(true_labels, predicted_labels)),
    }
