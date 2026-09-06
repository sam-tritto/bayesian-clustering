"""Continuous Dirichlet Process Gaussian Mixture Model (DP-GMM) using Scikit-Learn.

Phase 2 of the Non-Parametric Bayesian Clustering project.
"""

import json
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)

from src.config import PipelineConfig, default_config, DATA_DIR, ARTIFACTS_DIR
from src.utils.eval_utils import filter_active_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fit_sklearn_dpgmm(
    X: np.ndarray,
    config: PipelineConfig = default_config
) -> BayesianGaussianMixture:
    """Fit a Dirichlet Process Gaussian Mixture Model via Variational Inference.

    Mathematical and Numerical Rationale:
        1. Non-Parametric Stick-Breaking: Setting weight_concentration_prior_type='dirichlet_process'
           implements a truncated variational stick-breaking approximation. By choosing a high
           truncation limit (K=30), the variational algorithm automatically prunes excess
           components whose posterior weights approach zero.
        2. Spherical Covariance: Using covariance_type='spherical' restricts each cluster's
           covariance matrix to Sigma_k = sigma_k^2 * I. In 12-dimensional UMAP space, this
           avoids inverting a full D x D matrix (which risks singularity and numerical instability
           if a cluster has few data points), while remaining isotropic and computationally efficient.

    Args:
        X: Feature matrix of shape (N, D), typically 12D UMAP embeddings.
        config: Configuration containing truncation limit, concentration prior, and seed.

    Returns:
        Fitted BayesianGaussianMixture estimator.
    """
    logger.info(
        "Fitting BayesianGaussianMixture (K_trunc=%d, alpha=%.2f, cov=%s)...",
        config.truncation_k,
        config.weight_concentration_prior,
        config.covariance_type,
    )
    dpgmm = BayesianGaussianMixture(
        n_components=config.truncation_k,
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=config.weight_concentration_prior,
        covariance_type=config.covariance_type,
        max_iter=300,
        random_state=config.random_seed,
        verbose=0,
    )
    dpgmm.fit(X)
    logger.info("DP-GMM converged in %d iterations (lower bound: %.4f).", dpgmm.n_iter_, dpgmm.lower_bound_)
    return dpgmm


def get_active_cluster_assignments(
    dpgmm: BayesianGaussianMixture,
    X: np.ndarray,
    threshold: float = 0.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prune inactive clusters and assign each point to the most probable active cluster.

    Args:
        dpgmm: Fitted BayesianGaussianMixture model.
        X: Feature matrix of shape (N, D).
        threshold: Minimum mixture weight to retain a component.

    Returns:
        Tuple of:
            - active_indices: Original component indices kept.
            - active_weights: Normalized weights of active components.
            - active_means: Centroids of active components, shape (K_active, D).
            - point_assignments: Integer active cluster IDs [0, K_active - 1] for each point.
    """
    weights = dpgmm.weights_
    active_indices, active_weights = filter_active_clusters(weights, threshold=threshold)
    logger.info(
        "Filtered dead clusters (weight < %.3f): %d / %d components active.",
        threshold,
        len(active_indices),
        len(weights),
    )

    # Compute full responsibilities (posterior probabilities p(z_i = k | x_i))
    responsibilities = dpgmm.predict_proba(X)  # shape: (N, K)

    # Restrict to active components and re-normalize across active components
    active_resp = responsibilities[:, active_indices]  # shape: (N, K_active)
    active_resp = active_resp / np.maximum(active_resp.sum(axis=1, keepdims=True), 1e-12)

    # MAP assignment over active clusters
    point_assignments = np.argmax(active_resp, axis=1)
    active_means = dpgmm.means_[active_indices]

    return active_indices, active_weights, active_means, point_assignments


def build_agglomerative_hierarchy(
    centroids: np.ndarray,
    n_parent_clusters: int = 3,
    linkage_method: str = "ward"
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform agglomerative hierarchical clustering on active DP-GMM centroids.

    Rationale:
        The Dirichlet Process GMM discovers fine-grained Gaussian leaf clusters.
        By performing agglomerative clustering on the resulting active cluster centroids,
        we reconstruct the latent semantic tree (e.g. comp, rec, sci macro-classes).

    Args:
        centroids: Active cluster centroids, shape (K_active, D).
        n_parent_clusters: Number of macro-clusters to cut (e.g., 3 for comp, rec, sci).
        linkage_method: Linkage criterion ('ward', 'average', etc.).

    Returns:
        Tuple of:
            - Z: Linkage matrix for plotting dendrogram.
            - centroid_parent_labels: Macro-cluster label for each centroid (1 to n_parent_clusters).
    """
    logger.info(
        "Building Agglomerative Hierarchy over %d centroids (target parent clusters=%d, linkage=%s)...",
        len(centroids),
        n_parent_clusters,
        linkage_method,
    )
    Z = linkage(centroids, method=linkage_method)
    centroid_parent_labels = fcluster(Z, t=n_parent_clusters, criterion="maxclust")
    return Z, centroid_parent_labels


def evaluate_discovered_hierarchy(
    fine_labels: np.ndarray,
    parent_labels: np.ndarray,
    point_cluster_assignments: np.ndarray,
    centroid_parent_labels: np.ndarray,
    active_indices: np.ndarray,
    fine_names: List[str],
    parent_names: List[str],
) -> Dict[str, Any]:
    """Quantitatively compare discovered clusters and discovered hierarchy against ground truth."""
    # 1. Evaluate leaf-level cluster assignments against 9 ground-truth subcategories
    leaf_metrics = {
        "ari": float(adjusted_rand_score(fine_labels, point_cluster_assignments)),
        "ami": float(adjusted_mutual_info_score(fine_labels, point_cluster_assignments)),
        "v_measure": float(v_measure_score(fine_labels, point_cluster_assignments)),
        "homogeneity": float(homogeneity_score(fine_labels, point_cluster_assignments)),
        "completeness": float(completeness_score(fine_labels, point_cluster_assignments)),
    }

    # 2. Map data points to discovered parent clusters via their centroid's parent label
    point_parent_assignments = centroid_parent_labels[point_cluster_assignments]

    # Evaluate discovered parent macro-clusters against 3 ground-truth parent categories
    parent_metrics = {
        "ari": float(adjusted_rand_score(parent_labels, point_parent_assignments)),
        "ami": float(adjusted_mutual_info_score(parent_labels, point_parent_assignments)),
        "v_measure": float(v_measure_score(parent_labels, point_parent_assignments)),
        "homogeneity": float(homogeneity_score(parent_labels, point_parent_assignments)),
        "completeness": float(completeness_score(parent_labels, point_parent_assignments)),
    }

    # 3. Determine dominant ground-truth category for each active cluster
    cluster_profiles = []
    for active_idx in range(len(active_indices)):
        mask = (point_cluster_assignments == active_idx)
        if np.sum(mask) == 0:
            continue
        cluster_fine = fine_labels[mask]
        cluster_parent = parent_labels[mask]

        fine_bincount = np.bincount(cluster_fine, minlength=len(fine_names))
        dom_fine_idx = int(np.argmax(fine_bincount))
        fine_purity = float(fine_bincount[dom_fine_idx] / len(cluster_fine))

        parent_bincount = np.bincount(cluster_parent, minlength=len(parent_names))
        dom_parent_idx = int(np.argmax(parent_bincount))
        parent_purity = float(parent_bincount[dom_parent_idx] / len(cluster_parent))

        cluster_profiles.append({
            "active_cluster_id": int(active_idx),
            "original_component_id": int(active_indices[active_idx]),
            "size": int(np.sum(mask)),
            "dominant_subcategory": fine_names[dom_fine_idx],
            "subcategory_purity": fine_purity,
            "dominant_parent": parent_names[dom_parent_idx],
            "parent_purity": parent_purity,
        })

    return {
        "leaf_metrics": leaf_metrics,
        "parent_metrics": parent_metrics,
        "cluster_profiles": cluster_profiles,
        "point_parent_assignments": point_parent_assignments,
    }


def plot_phase2_diagnostics(
    all_weights: np.ndarray,
    threshold: float,
    active_indices: np.ndarray,
    Z: np.ndarray,
    cluster_profiles: List[Dict[str, Any]],
    umap_2d: np.ndarray,
    point_cluster_assignments: np.ndarray,
    fine_labels: np.ndarray,
    parent_labels: np.ndarray,
    fine_names: List[str],
    parent_names: List[str],
) -> None:
    """Generate comprehensive visual diagnostics for Phase 2."""
    # Figure 1: Stick-breaking mixture weights
    plt.figure(figsize=(10, 4))
    bars = plt.bar(np.arange(len(all_weights)), all_weights, color="steelblue", edgecolor="black", alpha=0.85)
    for idx in active_indices:
        bars[idx].set_color("forestgreen")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Active Threshold ({threshold})")
    plt.title(f"DP-GMM Component Weights (Active={len(active_indices)} / {len(all_weights)})", fontsize=12, fontweight="bold")
    plt.xlabel("Component Index (Truncation K=30)")
    plt.ylabel("Mixture Weight $w_k$")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase2_dpgmm_weights.png", dpi=200)
    plt.close()

    # Figure 2: Dendrogram over active centroids
    plt.figure(figsize=(12, 6))
    leaf_labels = [
        f"C{p['active_cluster_id']}: {p['dominant_subcategory'][:12]} ({p['size']} docs)"
        for p in cluster_profiles
    ]
    dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=30,
        leaf_font_size=10,
        color_threshold=Z[-2, 2] if len(Z) >= 2 else None,
    )
    plt.title("Agglomerative Hierarchy Over Discovered DP-GMM Centroids", fontsize=13, fontweight="bold")
    plt.ylabel("Ward Linkage Distance")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase2_hierarchy_dendrogram.png", dpi=200)
    plt.close()

    # Figure 3: Discovered Active Clusters vs Ground Truth on 2D UMAP
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Ground Truth Parent
    scatter0 = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=parent_labels, cmap="tab10", alpha=0.5, s=12)
    axes[0].set_title("Ground Truth Parent Classes", fontsize=12, fontweight="bold")
    handles0, _ = scatter0.legend_elements()
    axes[0].legend(handles0, parent_names, title="Parent", loc="best", framealpha=0.8)

    # Discovered DP-GMM Clusters
    scatter1 = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c=point_cluster_assignments, cmap="tab20", alpha=0.5, s=12)
    axes[1].set_title(f"Discovered DP-GMM Clusters ($K={len(active_indices)}$)", fontsize=12, fontweight="bold")

    # Ground Truth Subcategories
    scatter2 = axes[2].scatter(umap_2d[:, 0], umap_2d[:, 1], c=fine_labels, cmap="tab20", alpha=0.5, s=12)
    axes[2].set_title("Ground Truth Subcategories (9 Leaf Topics)", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase2_clusters_vs_truth.png", dpi=200)
    plt.close()


def run_phase_2(
    config: PipelineConfig = default_config
) -> Dict[str, Any]:
    """Execute Phase 2: DP-GMM fitting, active cluster pruning, and hierarchy discovery."""
    logger.info("=== Starting Phase 2: Continuous DP-GMM (Scikit-Learn Baseline) ===")

    # 1. Load Phase 1 cached artifacts
    cache_file = DATA_DIR / "phase1_processed.npz"
    metadata_file = DATA_DIR / "phase1_metadata.json"
    if not cache_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"Phase 1 artifacts not found at {cache_file}. Run Phase 1 first."
        )

    with np.load(cache_file) as data:
        umap_embeddings = data["umap_embeddings"]
        umap_2d = data["umap_2d"]
        fine_labels = data["fine_labels"]
        parent_labels = data["parent_labels"]

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fine_names = meta["fine_names"]
    parent_names = meta["parent_names"]

    # 2. Fit BayesianGaussianMixture with dirichlet_process prior
    dpgmm = fit_sklearn_dpgmm(umap_embeddings, config)

    # 3. Filter dead clusters
    active_indices, active_weights, active_means, point_assignments = get_active_cluster_assignments(
        dpgmm,
        umap_embeddings,
        threshold=config.weight_threshold,
    )

    # 4. Build agglomerative hierarchy on active centroids
    n_parents = len(parent_names)  # 3 parents: comp, rec, sci
    Z, centroid_parent_labels = build_agglomerative_hierarchy(
        active_means,
        n_parent_clusters=n_parents,
        linkage_method="ward",
    )

    # 5. Evaluate against ground truth
    eval_results = evaluate_discovered_hierarchy(
        fine_labels=fine_labels,
        parent_labels=parent_labels,
        point_cluster_assignments=point_assignments,
        centroid_parent_labels=centroid_parent_labels,
        active_indices=active_indices,
        fine_names=fine_names,
        parent_names=parent_names,
    )

    # 6. Generate diagnostic plots
    plot_phase2_diagnostics(
        all_weights=dpgmm.weights_,
        threshold=config.weight_threshold,
        active_indices=active_indices,
        Z=Z,
        cluster_profiles=eval_results["cluster_profiles"],
        umap_2d=umap_2d,
        point_cluster_assignments=point_assignments,
        fine_labels=fine_labels,
        parent_labels=parent_labels,
        fine_names=fine_names,
        parent_names=parent_names,
    )

    # 7. Save results
    results_summary = {
        "truncation_k": config.truncation_k,
        "active_clusters_discovered": int(len(active_indices)),
        "active_weights": [float(w) for w in active_weights],
        "leaf_evaluation": eval_results["leaf_metrics"],
        "parent_hierarchy_evaluation": eval_results["parent_metrics"],
        "cluster_profiles": eval_results["cluster_profiles"],
    }
    with open(ARTIFACTS_DIR / "phase2_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("=== Phase 2 Results Summary ===")
    logger.info("Active Clusters Discovered: %d (Ground Truth Leaves: %d)", len(active_indices), len(fine_names))
    logger.info("Leaf Subcategory V-Measure: %.4f | ARI: %.4f", eval_results["leaf_metrics"]["v_measure"], eval_results["leaf_metrics"]["ari"])
    logger.info("Parent Macro-Hierarchy V-Measure: %.4f | ARI: %.4f", eval_results["parent_metrics"]["v_measure"], eval_results["parent_metrics"]["ari"])
    logger.info("=== Phase 2 Completed Successfully ===")

    return results_summary


if __name__ == "__main__":
    run_phase_2()
