"""Continuous Dirichlet Process Gaussian Mixture Model (DP-GMM) implemented natively in PyMC.

Phase 3 of the Non-Parametric Bayesian Clustering project.
"""

import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)

from src.config import PipelineConfig, default_config, DATA_DIR, ARTIFACTS_DIR
from src.utils.math_utils import logsumexp
from src.utils.eval_utils import filter_active_clusters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_pymc_dpgmm_model(
    X: np.ndarray,
    K: int = 30,
    alpha_prior_rate: float = 1.0,
    mu_prior_sigma: float = 5.0,
    sigma_prior_scale: float = 2.0,
) -> pm.Model:
    """Construct a native PyMC model for Dirichlet Process Gaussian Mixture Model (DP-GMM).

    Mathematical Formulation:
        1. Base Measure and Concentration:
           alpha ~ Gamma(alpha=1.0, beta=alpha_prior_rate)
           alpha governs the rate of cluster generation under the Dirichlet process prior.
        2. Stick-Breaking Construction:
           v_k ~ Beta(1.0, alpha)  for k = 0, ..., K-2
           w_0 = v_0
           w_k = v_k * prod_{j < k} (1 - v_j)   for 0 < k < K-1
           w_{K-1} = prod_{j < K-1} (1 - v_j)   [ensures sum(w) = 1]
        3. Spherical Component Likelihoods:
           mu_k ~ Normal(0, mu_prior_sigma^2 * I_D)
           sigma_k ~ HalfNormal(sigma_prior_scale)
           comp_k = Normal(mu_k, sigma_k^2 * I_D)
        4. Mixture Observation Likelihood:
           y_i ~ sum_{k=0}^{K-1} w_k * comp_k(y_i)

    Args:
        X: Feature matrix of shape (N, D), typically 12D UMAP embeddings.
        K: Truncation limit for the stick-breaking process (default: 30).
        alpha_prior_rate: Rate parameter for Gamma prior on concentration alpha.
        mu_prior_sigma: Standard deviation for Normal prior on cluster centroids.
        sigma_prior_scale: Scale parameter for HalfNormal prior on cluster spreads.

    Returns:
        Configured pm.Model instance.
    """
    N, D = X.shape
    logger.info("Building PyMC DP-GMM model (N=%d, D=%d, K_trunc=%d)...", N, D, K)

    with pm.Model() as model:
        # Concentration parameter alpha governing cluster dispersion
        alpha = pm.Gamma("alpha", alpha=1.0, beta=alpha_prior_rate)

        # Stick fractions v ~ Beta(1, alpha)
        v = pm.Beta("v", alpha=1.0, beta=alpha, shape=K - 1)

        # Deterministic stick-breaking weights
        one_minus_v = 1.0 - v
        cumprod = pt.cumprod(one_minus_v)
        w_0 = v[0:1]
        w_rest = v[1:] * cumprod[:-1]
        w_last = cumprod[-1:]
        w = pm.Deterministic("w", pt.concatenate([w_0, w_rest, w_last]))

        # Empirical center and spread of the data coordinates
        x_mean = np.mean(X, axis=0)
        x_std = np.std(X, axis=0)

        # Component centroids: centered at data mean with broad prior variance
        mu = pm.Normal("mu", mu=x_mean, sigma=x_std * 2.5, shape=(K, D))

        # Component standard deviations (spherical covariance): shape (K,)
        sigma = pm.HalfNormal("sigma", sigma=sigma_prior_scale, shape=K)

        # List of K spherical Normal distributions in D dimensions
        comp_dists = [
            pm.Normal.dist(mu=mu[k], sigma=sigma[k], shape=(D,))
            for k in range(K)
        ]

        # Mixture likelihood over observations
        pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=X)

    return model


def fit_dpgmm_advi(
    model: pm.Model,
    X: np.ndarray,
    n_iterations: int = 2500,
    draws: int = 500,
    use_kmeans_init: bool = True,
    random_seed: int = 42,
) -> Tuple[az.InferenceData, np.ndarray]:
    """Fit DP-GMM using Automatic Differentiation Variational Inference (ADVI).

    Why ADVI is optimal for large continuous mixtures:
        - Exact MCMC sampling (NUTS) on mixture models suffers from label switching
          and non-identifiable permutation modes, causing poor mixing across chains.
        - ADVI fits a mean-field variational approximation to the posterior in seconds,
          providing smooth convergence and stable posterior point estimates.
        - Initializing component centroids with KMeans centers (as in scikit-learn's DP-GMM)
          breaks symmetry between the K components, preventing all components from collapsing
          to the global empirical mean.

    Args:
        model: Configured PyMC model.
        X: Feature matrix of shape (N, D).
        n_iterations: Optimization steps for the Evidence Lower Bound (ELBO).
        draws: Number of posterior draws to sample from the fitted approximation.
        use_kmeans_init: Whether to initialize centroid variational means with KMeans centers.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (InferenceData trace, ELBO history array).
    """
    start = None
    if use_kmeans_init:
        from sklearn.cluster import KMeans
        K = default_config.truncation_k
        logger.info("Initializing ADVI component centroids using KMeans (K=%d)...", K)
        kmeans = KMeans(n_clusters=K, n_init=1, random_state=random_seed).fit(X)
        start = {"mu": kmeans.cluster_centers_}

    logger.info("Fitting PyMC DP-GMM with ADVI (%d iterations)...", n_iterations)
    with model:
        approx = pm.fit(
            n=n_iterations,
            method="advi",
            start=start,
            random_seed=random_seed,
            progressbar=True,
        )
        trace = approx.sample(draws=draws, random_seed=random_seed)

    elbo_history = -np.array(approx.hist)  # approx.hist records negative ELBO
    logger.info("ADVI optimization completed. Final ELBO: %.2f", elbo_history[-1])
    return trace, elbo_history


def compute_posthoc_cluster_responsibilities(
    X: np.ndarray,
    trace: az.InferenceData,
    threshold: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute post-hoc posterior cluster probabilities using the log-sum-exp trick.

    Mathematical and Numerical Safeguards:
        For observation x_i in R^D and component k:
        log p(x_i, z_i = k) = log w_k + log N(x_i | mu_k, sigma_k^2 * I_D)
        log N(x_i | mu_k, sigma_k^2 * I_D) = -0.5 * D * log(2pi) - D * log(sigma_k) - 0.5 / (sigma_k^2) * ||x_i - mu_k||^2

        To avoid floating-point underflow when normalizing:
        log p(z_i = k | x_i) = log p(x_i, z_i = k) - logsumexp_j(log p(x_i, z_i = j))
        gamma_{ik} = exp(log p(z_i = k | x_i))

        This prevents underflow to 0.0 when distances are non-zero, ensuring reliable MAP assignments.

    Args:
        X: Observation matrix of shape (N, D).
        trace: Posterior InferenceData containing 'w', 'mu', 'sigma'.
        threshold: Minimum mixture weight to consider a cluster active.

    Returns:
        Tuple of:
            - responsibilities: Soft assignment matrix of shape (N, K_active).
            - point_assignments: Integer MAP cluster assignments [0, K_active - 1].
            - active_indices: Indices of active components.
            - active_weights: Normalized posterior mean weights of active components.
            - active_means: Posterior mean centroids of active components.
    """
    N, D = X.shape

    # Extract posterior mean parameters across posterior draws
    post_w = np.mean(trace.posterior["w"].values.reshape(-1, trace.posterior["w"].shape[-1]), axis=0)
    post_mu = np.mean(trace.posterior["mu"].values.reshape(-1, *trace.posterior["mu"].shape[-2:]), axis=0)
    post_sigma = np.mean(trace.posterior["sigma"].values.reshape(-1, trace.posterior["sigma"].shape[-1]), axis=0)

    K = len(post_w)
    logger.info("Extracted posterior means: K=%d, D=%d", K, D)

    # Filter active components based on posterior weight
    active_indices, active_weights = filter_active_clusters(post_w, threshold=threshold)
    K_active = len(active_indices)
    logger.info(
        "Active components (weight >= %.3f): %d / %d active.",
        threshold,
        K_active,
        K,
    )

    act_w = post_w[active_indices]
    act_mu = post_mu[active_indices]        # (K_act, D)
    act_sigma = post_sigma[active_indices]  # (K_act,)

    # Compute log joint: log p(x_i, z_i = k)
    log_w = np.log(np.maximum(act_w, 1e-15))  # (K_act,)

    # Squared Euclidean distances: ||x_i - mu_k||^2, shape (N, K_act)
    diff = X[:, None, :] - act_mu[None, :, :]  # (N, K_act, D)
    sq_dist = np.sum(diff ** 2, axis=-1)       # (N, K_act)

    # Spherical log normal likelihood
    log_norm_const = -0.5 * D * np.log(2.0 * np.pi) - D * np.log(act_sigma[None, :])
    log_lik = log_norm_const - 0.5 * sq_dist / (act_sigma[None, :] ** 2)  # (N, K_act)

    # Unnormalized log joint
    log_joint = log_w[None, :] + log_lik  # (N, K_act)

    # Stable normalization using log-sum-exp trick
    log_marginal = logsumexp(log_joint, axis=1, keepdims=True)  # (N, 1)
    log_responsibilities = log_joint - log_marginal
    responsibilities = np.exp(log_responsibilities)

    # MAP assignment
    point_assignments = np.argmax(responsibilities, axis=1)

    return responsibilities, point_assignments, active_indices, active_weights, act_mu


def plot_phase3_diagnostics(
    elbo_history: np.ndarray,
    all_weights: np.ndarray,
    active_indices: np.ndarray,
    threshold: float,
    umap_2d: np.ndarray,
    point_assignments: np.ndarray,
    fine_labels: np.ndarray,
    fine_names: List[str],
) -> None:
    """Generate diagnostic plots for Phase 3 PyMC DP-GMM."""
    # Plot 1: ELBO Optimization curve
    plt.figure(figsize=(8, 4))
    plt.plot(elbo_history, color="navy", linewidth=1.5)
    plt.title("PyMC ADVI Evidence Lower Bound (ELBO) Optimization", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase3_advi_elbo.png", dpi=200)
    plt.close()

    # Plot 2: Posterior stick-breaking weights
    plt.figure(figsize=(10, 4))
    bars = plt.bar(np.arange(len(all_weights)), all_weights, color="steelblue", edgecolor="black", alpha=0.85)
    for idx in active_indices:
        bars[idx].set_color("darkorange")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Active Threshold ({threshold})")
    plt.title(f"PyMC Posterior Component Weights (Active={len(active_indices)} / {len(all_weights)})", fontsize=12, fontweight="bold")
    plt.xlabel("Component Index (Truncation K=30)")
    plt.ylabel("Posterior Weight $w_k$")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase3_pymc_weights.png", dpi=200)
    plt.close()

    # Plot 3: 2D UMAP Inferred Clusters vs Ground Truth Subcategories
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    scatter_pymc = axes[0].scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=point_assignments,
        cmap="tab20",
        alpha=0.6,
        s=15,
    )
    axes[0].set_title(f"PyMC DP-GMM Discovered Clusters (K={len(active_indices)})", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    scatter_gt = axes[1].scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=fine_labels,
        cmap="tab20",
        alpha=0.6,
        s=15,
    )
    axes[1].set_title("Ground Truth Subcategories (9 Leaf Topics)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    handles_gt, _ = scatter_gt.legend_elements()
    axes[1].legend(handles_gt, fine_names, bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase3_pymc_clusters.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_phase_3(
    config: PipelineConfig = default_config,
    advi_iterations: int = 2500,
) -> Dict[str, Any]:
    """Execute complete Phase 3: PyMC model building, ADVI fitting, log-sum-exp post-hoc assignment."""
    logger.info("=== Starting Phase 3: Continuous DP-GMM (PyMC Native Implementation) ===")

    # 1. Load Phase 1 processed data
    cache_file = DATA_DIR / "phase1_processed.npz"
    metadata_file = DATA_DIR / "phase1_metadata.json"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached data at {cache_file}. Run Phase 1 first.")

    with np.load(cache_file) as data:
        umap_embeddings = data["umap_embeddings"]
        umap_2d = data["umap_2d"]
        fine_labels = data["fine_labels"]
        parent_labels = data["parent_labels"]

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    fine_names = meta["fine_names"]
    parent_names = meta["parent_names"]

    # 2. Build PyMC DP-GMM model
    model = build_pymc_dpgmm_model(
        X=umap_embeddings,
        K=config.truncation_k,
    )

    # 3. Fit with ADVI
    trace, elbo_history = fit_dpgmm_advi(
        model=model,
        X=umap_embeddings,
        n_iterations=advi_iterations,
        draws=500,
        use_kmeans_init=True,
        random_seed=config.random_seed,
    )

    # 4. Extract posterior mean weights
    post_w = np.mean(trace.posterior["w"].values.reshape(-1, config.truncation_k), axis=0)

    # 5. Compute post-hoc cluster assignments via log-sum-exp
    responsibilities, point_assignments, active_indices, active_weights, act_mu = (
        compute_posthoc_cluster_responsibilities(
            X=umap_embeddings,
            trace=trace,
            threshold=config.weight_threshold,
        )
    )

    # 6. Quantitative Evaluation against ground truth
    leaf_metrics = {
        "ari": float(adjusted_rand_score(fine_labels, point_assignments)),
        "ami": float(adjusted_mutual_info_score(fine_labels, point_assignments)),
        "v_measure": float(v_measure_score(fine_labels, point_assignments)),
        "homogeneity": float(homogeneity_score(fine_labels, point_assignments)),
        "completeness": float(completeness_score(fine_labels, point_assignments)),
    }

    # 7. Generate diagnostic plots
    plot_phase3_diagnostics(
        elbo_history=elbo_history,
        all_weights=post_w,
        active_indices=active_indices,
        threshold=config.weight_threshold,
        umap_2d=umap_2d,
        point_assignments=point_assignments,
        fine_labels=fine_labels,
        fine_names=fine_names,
    )

    # 8. Save results summary
    results_summary = {
        "truncation_k": config.truncation_k,
        "advi_iterations": advi_iterations,
        "active_clusters_discovered": int(len(active_indices)),
        "active_weights": [float(w) for w in active_weights],
        "leaf_evaluation": leaf_metrics,
    }
    with open(ARTIFACTS_DIR / "phase3_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("=== Phase 3 Results Summary ===")
    logger.info("Active Clusters Discovered: %d", len(active_indices))
    logger.info("Leaf Subcategory V-Measure: %.4f | ARI: %.4f", leaf_metrics["v_measure"], leaf_metrics["ari"])
    logger.info("=== Phase 3 Completed Successfully ===")

    return results_summary


if __name__ == "__main__":
    run_phase_3()
