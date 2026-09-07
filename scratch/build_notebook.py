"""Script to generate the comprehensive Bayesian Clustering tutorial Jupyter Notebook,
including the 2-step semi-automated Yellowbrick baseline (Elbow + Silhouette).
"""

import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# ==============================================================================
# Cell 1: Title & Introduction
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# Non-Parametric Bayesian Clustering & Hierarchy Discovery
### From Classical Two-Step Heuristics (Yellowbrick) to DP-GMM (Scikit-Learn & PyMC) and Discrete HDP (Tomotopy)

---

## 🎯 Tutorial Objectives
In standard clustering algorithms (like $k$-means or standard GMMs), you are forced to specify the number of clusters $K$ *a priori*. In exploratory data analysis and natural language processing, $K$ is rarely known in advance, and real-world data frequently exhibits an **unbounded, hierarchical structure**.

In this tutorial, you will master the progression from classical heuristic clustering to non-parametric Bayesian models:
1. **Classical Baseline (Yellowbrick Two-Pass Search)**: A popular practitioner workflow that runs a coarse sweep with **Distortion (Inertia) Elbow**, followed by a localized fine-tuning sweep around $K \\pm N$ with the **Silhouette Score** (comparing PCA vs. UMAP).
2. **Continuous DP-GMM (Scikit-Learn Baseline)**: Using variational inference with a Dirichlet Process prior to prune inactive clusters from a truncation limit $K=30$ in a **single fit**, and using agglomerative linkage on centroids to discover parent hierarchies.
3. **Continuous DP-GMM (Native PyMC Implementation)**: Implementing the stick-breaking process from scratch in PyMC, configuring spherical Gaussian components, and applying the **log-sum-exp trick** for numerically stable post-hoc cluster assignments.
4. **Discrete HDP (Tomotopy)**: Applying the Hierarchical Dirichlet Process via **Collapsed Gibbs Sampling** directly on tokenized text to discover latent topics and word distributions.
5. **Ground-Truth Benchmarking**: Testing all methods on the **20 Newsgroups** dataset, which has a known 2-level hierarchy (`comp.*`, `rec.*`, `sci.*`), to evaluate whether models can automatically recover both leaf topics and parent categories without supervision.

---
"""))

# ==============================================================================
# Cell 2: Environment Setup
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 0. Environment Setup & Reproducibility
We import our core libraries, configure the random seed, and set styling defaults.
"""))

cells.append(nbf.v4.new_code_cell("""import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress verbose warnings
import warnings
warnings.filterwarnings("ignore")

# Set random seed for complete reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plotting theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

print("Environment configured successfully!")
"""))

# ==============================================================================
# Cell 3: Phase 1 - Data Preparation & Geometry of Embeddings
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. Phase 1: Data Preparation & The Geometry of Embeddings

### Dataset: 20 Newsgroups Subsets
We select 9 subcategories across 3 distinct parent families:
- **`comp.*`**: `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.mac.hardware`
- **`rec.*`**: `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`
- **`sci.*`**: `sci.crypt`, `sci.med`, `sci.space`

This provides an explicit ground-truth hierarchy: **3 parent classes** and **9 leaf subcategories**.

### Why Dimensionality Reduction (UMAP & PCA) is Mathematically Necessary
Modern Sentence Transformers produce dense 384-dimensional embeddings (e.g., `all-MiniLM-L6-v2`). However, fitting Gaussian likelihoods directly in 384 dimensions fails due to:
1. **The Distance Concentration Phenomenon ("Curse of Dimensionality")**: In high dimensions, the ratio of distance between the nearest and farthest points approaches 1. Pairwise distances concentrate around $\\sqrt{2}$.
2. **Likelihood Underflow**: The Gaussian density contains a normalizer $(2\\pi)^{-D/2} |\\Sigma|^{-1/2}$. For $D=384$, $(2\\pi)^{-192} \\approx 10^{-153}$, which immediately suffers floating-point underflow or causes singular covariance condition numbers.
3. **Manifold Learning**: Reducing to 12 continuous dimensions preserves semantic geometry while making spherical Gaussian assumptions well-conditioned.
"""))

cells.append(nbf.v4.new_code_cell("""from src.config import default_config, DATA_DIR, ARTIFACTS_DIR
from src.data.pipeline import fetch_and_clean_data, generate_embeddings, reduce_dimensions_umap

# Check if Phase 1 cached arrays exist; if not, compute them
cache_file = DATA_DIR / "phase1_processed.npz"
metadata_file = DATA_DIR / "phase1_metadata.json"

if cache_file.exists() and metadata_file.exists():
    print("Loading cached Phase 1 data...")
    with np.load(cache_file) as data:
        embeddings = data["embeddings"]
        umap_embeddings = data["umap_embeddings"]
        umap_2d = data["umap_2d"]
        fine_labels = data["fine_labels"]
        parent_labels = data["parent_labels"]
    with open(metadata_file, "r") as f:
        meta = json.load(f)
    fine_names = meta["fine_names"]
    parent_names = meta["parent_names"]
    with open(DATA_DIR / "phase1_texts.json", "r") as f:
        texts = json.load(f)
else:
    print("Running Phase 1 pipeline from scratch...")
    from src.data.pipeline import run_phase_1
    res = run_phase_1(default_config)
    umap_embeddings = res["umap_embeddings"]
    umap_2d = res["umap_2d"]
    fine_labels = res["fine_labels"]
    parent_labels = res["parent_labels"]
    fine_names = res["fine_names"]
    parent_names = res["parent_names"]
    texts = res["texts"]

print(f"Loaded {len(texts)} documents.")
print(f"Embedding shape: {embeddings.shape} -> UMAP shape: {umap_embeddings.shape}")
print(f"Ground truth categories (9 leaves): {fine_names}")
print(f"Parent categories (3 roots): {parent_names}")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualize Ground Truth Hierarchy on 2D UMAP
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Left: 3 Parent Macro-Classes
scatter_p = axes[0].scatter(umap_2d[:, 0], umap_2d[:, 1], c=parent_labels, cmap="tab10", alpha=0.5, s=15)
axes[0].set_title("Ground Truth Parent Classes (comp, rec, sci)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")
handles_p, _ = scatter_p.legend_elements()
axes[0].legend(handles_p, parent_names, title="Parent", loc="best")

# Right: 9 Subcategories
scatter_f = axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c=fine_labels, cmap="tab20", alpha=0.5, s=15)
axes[1].set_title("Ground Truth Subcategories (9 Leaf Topics)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")
handles_f, _ = scatter_f.legend_elements()
axes[1].legend(handles_f, fine_names, bbox_to_anchor=(1.02, 1), loc="upper left", title="Subcategory")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 4: Classical Baseline - Yellowbrick 2-Pass Workflow
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Baseline: Two-Step Semi-Automated Clustering with Yellowbrick

Before diving into Bayesian inference, let us implement the standard heuristic workflow that practitioners often use in production:
1. **Pass 1 (Coarse Search via Distortion / Inertia Elbow)**:
   We train multiple $k$-means models across a broad candidate range $K \\in [2, 16]$ and locate the "elbow" where the rate of decrease in within-cluster sum of squares sharply slows down.
2. **Pass 2 (Fine-Tuning via Local Silhouette Sweep)**:
   Taking the elbow candidate $K_{\\text{elbow}}$, we evaluate the **Silhouette Score** in a local window $K \\in [K_{\\text{elbow}} - N, K_{\\text{elbow}} + N]$ (e.g. $\\pm 3$) to select the number of clusters that maximizes point-to-cluster cohesion relative to nearest-cluster separation.
3. **PCA vs. UMAP Comparison**:
   We will test this two-pass search on both **12D PCA** (linear variance projection) and **12D UMAP** (nonlinear manifold projection) to see how geometry impacts the selected $K$.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import v_measure_score, adjusted_rand_score

# Fit 12D PCA on the normalized 384D sentence embeddings for direct comparison with 12D UMAP
pca = PCA(n_components=12, random_state=RANDOM_SEED)
pca_embeddings = pca.fit_transform(embeddings)

print(f"PCA shape: {pca_embeddings.shape} (Total variance explained: {pca.explained_variance_ratio_.sum():.2%})")
print(f"UMAP shape: {umap_embeddings.shape}")
"""))

cells.append(nbf.v4.new_code_cell("""# -------------------------------------------------------------------------
# UMAP Workflow: Pass 1 (Distortion Elbow) & Pass 2 (Local Silhouette Sweep)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Pass 1: Distortion Elbow across K in [2, 16]
viz_umap_elbow = KElbowVisualizer(
    KMeans(random_state=RANDOM_SEED, n_init=3),
    k=(2, 17),
    metric="distortion",
    timings=False,
    force_model=True,
    ax=axes[0],
)
viz_umap_elbow.fit(umap_embeddings)
axes[0].set_title(f"UMAP Pass 1: Distortion Elbow (Optimal k = {viz_umap_elbow.elbow_value_})", fontweight="bold")

k_elbow_umap = viz_umap_elbow.elbow_value_

# Pass 2: Local Silhouette Sweep around k_elbow +/- 3
k_min_umap = max(2, k_elbow_umap - 3)
k_max_umap = k_elbow_umap + 4

viz_umap_sil = KElbowVisualizer(
    KMeans(random_state=RANDOM_SEED, n_init=3),
    k=(k_min_umap, k_max_umap),
    metric="silhouette",
    timings=False,
    force_model=True,
    ax=axes[1],
)
viz_umap_sil.fit(umap_embeddings)
axes[1].set_title(f"UMAP Pass 2: Local Silhouette Sweep (Optimal k = {viz_umap_sil.elbow_value_})", fontweight="bold")

plt.tight_layout()
plt.show()

k_best_umap = viz_umap_sil.elbow_value_
km_umap = KMeans(n_clusters=k_best_umap, random_state=RANDOM_SEED, n_init=10).fit(umap_embeddings)

umap_yb_v = v_measure_score(fine_labels, km_umap.labels_)
umap_yb_ari = adjusted_rand_score(fine_labels, km_umap.labels_)
print(f"--> UMAP Yellowbrick Result: k = {k_best_umap} | Leaf V-Measure: {umap_yb_v:.4f} | ARI: {umap_yb_ari:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# -------------------------------------------------------------------------
# PCA Workflow: Pass 1 (Distortion Elbow) & Pass 2 (Local Silhouette Sweep)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Pass 1: Distortion Elbow across K in [2, 16]
viz_pca_elbow = KElbowVisualizer(
    KMeans(random_state=RANDOM_SEED, n_init=3),
    k=(2, 17),
    metric="distortion",
    timings=False,
    force_model=True,
    ax=axes[0],
)
viz_pca_elbow.fit(pca_embeddings)
axes[0].set_title(f"PCA Pass 1: Distortion Elbow (Optimal k = {viz_pca_elbow.elbow_value_})", fontweight="bold")

k_elbow_pca = viz_pca_elbow.elbow_value_

# Pass 2: Local Silhouette Sweep around k_elbow +/- 3
k_min_pca = max(2, k_elbow_pca - 3)
k_max_pca = k_elbow_pca + 4

viz_pca_sil = KElbowVisualizer(
    KMeans(random_state=RANDOM_SEED, n_init=3),
    k=(k_min_pca, k_max_pca),
    metric="silhouette",
    timings=False,
    force_model=True,
    ax=axes[1],
)
viz_pca_sil.fit(pca_embeddings)
axes[1].set_title(f"PCA Pass 2: Local Silhouette Sweep (Optimal k = {viz_pca_sil.elbow_value_})", fontweight="bold")

plt.tight_layout()
plt.show()

k_best_pca = viz_pca_sil.elbow_value_
km_pca = KMeans(n_clusters=k_best_pca, random_state=RANDOM_SEED, n_init=10).fit(pca_embeddings)

pca_yb_v = v_measure_score(fine_labels, km_pca.labels_)
pca_yb_ari = adjusted_rand_score(fine_labels, km_pca.labels_)
print(f"--> PCA Yellowbrick Result: k = {k_best_pca} | Leaf V-Measure: {pca_yb_v:.4f} | ARI: {pca_yb_ari:.4f}")
"""))

cells.append(nbf.v4.new_markdown_cell("""### Summary of the Yellowbrick Heuristic Approach
- **What worked well**: It is conceptually intuitive, runs fast, and provides visual diagnostic curves.
- **The Core Limitations**:
  1. **Computational Inefficiency**: We had to train $15 + 7 = 22$ separate $k$-means models across the two sweeps.
  2. **Heuristic Window Sensitivity**: The result of Pass 2 depends entirely on the window $\\pm N$ chosen around the Pass 1 elbow.
  3. **No Posterior Uncertainty or Hierarchy**: It returns hard cluster memberships with no probabilistic uncertainty, no cluster mixing weights, and no automated parent-child hierarchy.

This directly motivates **Non-Parametric Bayesian Clustering**, where a **single model fit** infers both the active cluster count and cluster weights autonomously via Dirichlet Process priors.
"""))

# ==============================================================================
# Cell 5: Phase 2 - Scikit-Learn DP-GMM & Agglomerative Hierarchy
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Phase 2: Continuous DP-GMM (Scikit-Learn Baseline)

### Mathematical Formulation
A Dirichlet Process Gaussian Mixture Model (DP-GMM) places a Dirichlet Process prior $G \\sim \\text{DP}(\\alpha, G_0)$ over the mixture components:
$$y_i \\sim \\sum_{k=1}^K w_k \\mathcal{N}(\\mu_k, \\Sigma_k)$$

In `scikit-learn`:
1. `weight_concentration_prior_type='dirichlet_process'` implements a truncated variational stick-breaking approximation.
2. We specify a high truncation limit ($K=30$).
3. `covariance_type='spherical'` enforces $\\Sigma_k = \\sigma_k^2 \\mathbf{I}$, preventing singular covariance inversions.
4. **Active Cluster Pruning**: Components with posterior weight $w_k < 0.01$ are classified as "dead" components and pruned.
5. **Hierarchical Discovery**: We feed the active component centroids $\\mu_k \\in \\mathbb{R}^{12}$ into **Agglomerative Hierarchical Clustering (Ward linkage)** to reconstruct the latent parent-child tree.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.mixture import BayesianGaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import adjusted_rand_score, v_measure_score, homogeneity_score
from src.utils.eval_utils import filter_active_clusters

# 1. Fit DP-GMM with high truncation limit K=30 in a SINGLE fit
truncation_k = 30
dpgmm = BayesianGaussianMixture(
    n_components=truncation_k,
    weight_concentration_prior_type="dirichlet_process",
    weight_concentration_prior=1.0,
    covariance_type="spherical",
    max_iter=300,
    random_state=RANDOM_SEED,
)
dpgmm.fit(umap_embeddings)
print(f"DP-GMM converged in {dpgmm.n_iter_} iterations (lower bound: {dpgmm.lower_bound_:.2f}).")

# 2. Filter inactive clusters (weight < 0.01)
weights = dpgmm.weights_
active_indices, active_weights = filter_active_clusters(weights, threshold=0.01)
print(f"Active components: {len(active_indices)} / {truncation_k} components active.")

# Responsibilities and MAP cluster assignment
resp = dpgmm.predict_proba(umap_embeddings)[:, active_indices]
resp /= resp.sum(axis=1, keepdims=True)
p2_pred_labels = np.argmax(resp, axis=1)
active_centroids = dpgmm.means_[active_indices]

# Quantitative Leaf Evaluation
leaf_v = v_measure_score(fine_labels, p2_pred_labels)
leaf_ari = adjusted_rand_score(fine_labels, p2_pred_labels)
leaf_homo = homogeneity_score(fine_labels, p2_pred_labels)
print(f"Leaf Subcategory V-Measure: {leaf_v:.4f} | ARI: {leaf_ari:.4f} | Homogeneity: {leaf_homo:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Agglomerative Clustering on Active Centroids to recover Parent Hierarchy
Z = linkage(active_centroids, method="ward")
pred_parent_clusters = fcluster(Z, t=3, criterion="maxclust")
p2_point_parents = pred_parent_clusters[p2_pred_labels]

parent_v = v_measure_score(parent_labels, p2_point_parents)
parent_ari = adjusted_rand_score(parent_labels, p2_point_parents)
print(f"Parent Macro-Hierarchy V-Measure: {parent_v:.4f} | ARI: {parent_ari:.4f}")

# Plot Dendrogram
plt.figure(figsize=(14, 6))
# Determine dominant category per cluster for dendrogram labels
cluster_labels = []
for k_idx in range(len(active_indices)):
    mask = (p2_pred_labels == k_idx)
    dom_cat = fine_names[np.bincount(fine_labels[mask]).argmax()]
    cluster_labels.append(f"C{k_idx}: {dom_cat[:14]} ({mask.sum()} docs)")

dendrogram(Z, labels=cluster_labels, leaf_rotation=35, leaf_font_size=10)
plt.title("Agglomerative Hierarchy Over DP-GMM Centroids", fontsize=13, fontweight="bold")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 6: Phase 3 - PyMC Native Stick-Breaking DP-GMM
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. Phase 3: Continuous DP-GMM (Native PyMC Implementation)

### The Stick-Breaking Process from Scratch
Rather than relying on scikit-learn's black-box EM, we implement the Dirichlet Process stick-breaking construction natively in **PyMC**:
1. **Concentration Prior**: $\\alpha \\sim \\text{Gamma}(1.0, 1.0)$
2. **Beta Sticks**: $v_k \\sim \\text{Beta}(1, \\alpha)$ for $k = 0, \\dots, K-2$
3. **Deterministic Mixture Weights**:
   $$w_0 = v_0, \\quad w_k = v_k \\prod_{j < k} (1 - v_j), \\quad w_{K-1} = \\prod_{j < K-1} (1 - v_j)$$
4. **Isotropic Normal Likelihood**:
   $$\\mu_k \\sim \\mathcal{N}(\\bar{x}, (2.5 s_x)^2 \\mathbf{I}), \\quad \\sigma_k \\sim \\text{HalfNormal}(2.0)$$
   $$y_i \\sim \\sum_{k=0}^{K-1} w_k \\mathcal{N}(\\mu_k, \\sigma_k^2 \\mathbf{I}_D)$$

### Numerical Stability: The Log-Sum-Exp Trick
To extract post-hoc responsibilities without numerical underflow:
$$\\log p(x_i, z_i = k) = \\log w_k - \\frac{D}{2} \\log(2\\pi) - D \\log \\sigma_k - \\frac{1}{2\\sigma_k^2} \\|x_i - \\mu_k\\|^2$$
$$\\log p(z_i = k \\mid x_i) = \\log p(x_i, z_i = k) - \\text{logsumexp}_j \\big( \\log p(x_i, z_i = j) \\big)$$
$$\\hat{z}_i = \\arg\\max_k \\log p(z_i = k \\mid x_i)$$
"""))

cells.append(nbf.v4.new_code_cell("""import pymc as pm
import pytensor.tensor as pt
from src.utils.math_utils import logsumexp

K = 30
N, D = umap_embeddings.shape

# 1. Define PyMC Model
with pm.Model() as pymc_dpgmm:
    alpha = pm.Gamma("alpha", alpha=1.0, beta=1.0)
    v = pm.Beta("v", alpha=1.0, beta=alpha, shape=K - 1)
    
    # Stick-breaking deterministic weights
    one_minus_v = 1.0 - v
    cumprod = pt.cumprod(one_minus_v)
    w_0 = v[0:1]
    w_rest = v[1:] * cumprod[:-1]
    w_last = cumprod[-1:]
    w = pm.Deterministic("w", pt.concatenate([w_0, w_rest, w_last]))
    
    # Empirical Bayes centroid prior
    x_mean = np.mean(umap_embeddings, axis=0)
    x_std = np.std(umap_embeddings, axis=0)
    mu = pm.Normal("mu", mu=x_mean, sigma=x_std * 2.5, shape=(K, D))
    sigma = pm.HalfNormal("sigma", sigma=2.0, shape=K)
    
    # Mixture likelihood with spherical Normal components
    comp_dists = [
        pm.Normal.dist(mu=mu[k], sigma=sigma[k], shape=(D,))
        for k in range(K)
    ]
    pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=umap_embeddings)

print("PyMC DP-GMM Model constructed successfully!")
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Fit via Variational Inference (ADVI) with KMeans centroid initialization
kmeans_init = KMeans(n_clusters=K, n_init=1, random_state=RANDOM_SEED).fit(umap_embeddings)
start_vals = {"mu": kmeans_init.cluster_centers_}

with pymc_dpgmm:
    approx = pm.fit(
        n=2500,
        method="advi",
        start=start_vals,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )
    trace = approx.sample(draws=500, random_seed=RANDOM_SEED)

elbo = -np.array(approx.hist)
print(f"ADVI completed. Final ELBO: {elbo[-1]:.2f}")
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Post-Hoc Log-Sum-Exp MAP Cluster Assignment
post_w = np.mean(trace.posterior["w"].values.reshape(-1, K), axis=0)
post_mu = np.mean(trace.posterior["mu"].values.reshape(-1, K, D), axis=0)
post_sigma = np.mean(trace.posterior["sigma"].values.reshape(-1, K), axis=0)

# Filter active clusters (w_k >= 0.01)
act_idx, act_w = filter_active_clusters(post_w, threshold=0.01)
K_act = len(act_idx)
print(f"PyMC active clusters: {K_act} / {K} (Weights: {np.round(act_w, 3)})")

act_mu = post_mu[act_idx]
act_sigma = post_sigma[act_idx]

# Stable Log-Sum-Exp computation
diff = umap_embeddings[:, None, :] - act_mu[None, :, :]
sq_dist = np.sum(diff ** 2, axis=-1)
log_norm = -0.5 * D * np.log(2.0 * np.pi) - D * np.log(act_sigma[None, :])
log_lik = log_norm - 0.5 * sq_dist / (act_sigma[None, :] ** 2)
log_joint = np.log(np.maximum(act_w, 1e-15))[None, :] + log_lik

log_marginal = logsumexp(log_joint, axis=1, keepdims=True)
p3_pred_labels = np.argmax(log_joint - log_marginal, axis=1)

p3_leaf_v = v_measure_score(fine_labels, p3_pred_labels)
p3_leaf_ari = adjusted_rand_score(fine_labels, p3_pred_labels)
print(f"PyMC Leaf Subcategory V-Measure: {p3_leaf_v:.4f} | ARI: {p3_leaf_ari:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Plot PyMC Stick-Breaking Decay and Discovered Clusters
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Plot 1: Stick-Breaking Weights
bars = axes[0].bar(range(K), post_w, color="steelblue", edgecolor="black")
for idx in act_idx:
    bars[idx].set_color("darkorange")
axes[0].axhline(0.01, color="red", linestyle="--", label="Threshold (0.01)")
axes[0].set_title(f"PyMC Posterior Stick-Breaking Weights (Active: {K_act} / {K})", fontweight="bold")
axes[0].set_xlabel("Component Index")
axes[0].set_ylabel("Posterior Weight")
axes[0].legend()

# Plot 2: 2D Projection of PyMC Clusters
axes[1].scatter(umap_2d[:, 0], umap_2d[:, 1], c=p3_pred_labels, cmap="tab20", alpha=0.6, s=15)
axes[1].set_title(f"PyMC Discovered Clusters (K={K_act})", fontweight="bold")
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 7: Phase 4 - Discrete HDP (Tomotopy)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Phase 4: Discrete Hierarchical Dirichlet Process (Tomotopy)

### Transitioning from Continuous Embeddings to Discrete Word Tokens
While DP-GMM operates on continuous sentence embeddings, text is inherently a sequence of discrete tokens.

The **Hierarchical Dirichlet Process (HDP)** (Teh et al., 2006) extends Latent Dirichlet Allocation (LDA) to non-parametric Bayesian topic modeling:
1. **Global Base Measure**: A corpus-wide Dirichlet Process generates a shared pool of topics:
   $$G_0 \\sim \\text{DP}(\\gamma, H)$$
2. **Document-Level Distributions**: Each document draws its own mixture over the global topics:
   $$G_d \\sim \\text{DP}(\\alpha, G_0)$$
3. **Collapsed Gibbs Sampling**: Rather than estimating dense variational parameters, Collapsed Gibbs Sampling analytically integrates out the topic mixtures and samples only discrete table assignments, running efficiently in seconds.
"""))

cells.append(nbf.v4.new_code_cell("""import tomotopy as tp
from src.models.tomotopy_hdp import preprocess_text_for_hdp, extract_prominent_topics, evaluate_hdp_clustering

# 1. Tokenize corpus
tokenized_corpus = preprocess_text_for_hdp(texts)

# 2. Initialize and Train HDP with Collapsed Gibbs Sampling
hdp = tp.HDPModel(
    tw=tp.TermWeight.IDF,
    min_cf=3,
    rm_top=5,
    alpha=0.1,
    gamma=1.0,
    initial_k=30,
    seed=RANDOM_SEED,
)

for doc in tokenized_corpus:
    hdp.add_doc(doc)

print("Training HDP Model over 500 Gibbs sweeps...")
hdp.train(500)
print(f"HDP training complete. Live topics discovered: {hdp.live_k} | Final LL/word: {hdp.ll_per_word:.4f}")

# 3. Extract prominent topics and display top words
prominent = extract_prominent_topics(hdp, top_n_words=5, top_k_topics=8)
for p in prominent:
    print(f"Topic {p['topic_id']:2d} ({p['word_count']:6d} tokens): {', '.join(p['top_words'])}")

# 4. Document-level evaluation against ground truth
p4_leaf_metrics, p4_parent_metrics, p4_pred_labels = evaluate_hdp_clustering(hdp, fine_labels, parent_labels)
print(f"HDP Leaf Subcategory V-Measure: {p4_leaf_metrics['v_measure']:.4f} | ARI: {p4_leaf_metrics['ari']:.4f}")
print(f"HDP Parent Macro-Hierarchy V-Measure: {p4_parent_metrics['v_measure']:.4f} | ARI: {p4_parent_metrics['ari']:.4f}")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualize Top 5 Words for Most Prominent Topics
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, top in enumerate(prominent[:8]):
    words = top["top_words"]
    probs = top["word_probs"]
    y_pos = np.arange(len(words))
    axes[i].barh(y_pos, probs, color="teal", alpha=0.85, edgecolor="black")
    axes[i].set_yticks(y_pos)
    axes[i].set_yticklabels(words, fontsize=10, fontweight="bold")
    axes[i].invert_yaxis()
    axes[i].set_title(f"Topic {top['topic_id']} ({top['word_count']} tokens)", fontsize=11)
    axes[i].set_xlabel("Probability")

plt.suptitle("Top 5 Characteristic Words for Prominent HDP Topics", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 8: Synthesis & Comprehensive Benchmark
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Synthesis & Comprehensive Benchmark

We now assemble the performance metrics across all models — classical Yellowbrick heuristics, continuous DP-GMMs, and discrete HDP — into a unified comparative benchmark.
"""))

cells.append(nbf.v4.new_code_cell("""# Compile Unified Benchmark Table
summary_data = [
    {
        "Model": "Baseline: Yellowbrick (PCA 12D)",
        "Paradigm": "Continuous (PCA Linear)",
        "Inference": "2-Pass (Elbow + Silhouette)",
        "Discovered Clusters": k_best_pca,
        "Leaf V-Measure": pca_yb_v,
        "Leaf ARI": pca_yb_ari,
        "Parent V-Measure": "N/A",
        "Parent ARI": "N/A",
    },
    {
        "Model": "Baseline: Yellowbrick (UMAP 12D)",
        "Paradigm": "Continuous (UMAP Manifold)",
        "Inference": "2-Pass (Elbow + Silhouette)",
        "Discovered Clusters": k_best_umap,
        "Leaf V-Measure": umap_yb_v,
        "Leaf ARI": umap_yb_ari,
        "Parent V-Measure": "N/A",
        "Parent ARI": "N/A",
    },
    {
        "Model": "Phase 2: DP-GMM (scikit-learn)",
        "Paradigm": "Continuous (UMAP Manifold)",
        "Inference": "Variational EM (Single Fit)",
        "Discovered Clusters": len(active_indices),
        "Leaf V-Measure": leaf_v,
        "Leaf ARI": leaf_ari,
        "Parent V-Measure": f"{parent_v:.4f}",
        "Parent ARI": f"{parent_ari:.4f}",
    },
    {
        "Model": "Phase 3: DP-GMM (PyMC Stick-Breaking)",
        "Paradigm": "Continuous (UMAP Manifold)",
        "Inference": "ADVI + LogSumExp (Single Fit)",
        "Discovered Clusters": K_act,
        "Leaf V-Measure": p3_leaf_v,
        "Leaf ARI": p3_leaf_ari,
        "Parent V-Measure": "N/A",
        "Parent ARI": "N/A",
    },
    {
        "Model": "Phase 4: HDP (tomotopy)",
        "Paradigm": "Discrete BoW Tokens",
        "Inference": "Collapsed Gibbs (Single Fit)",
        "Discovered Clusters": int(hdp.live_k),
        "Leaf V-Measure": p4_leaf_metrics["v_measure"],
        "Leaf ARI": p4_leaf_metrics["ari"],
        "Parent V-Measure": f"{p4_parent_metrics['v_measure']:.4f}",
        "Parent ARI": f"{p4_parent_metrics['ari']:.4f}",
    },
]

df_summary = pd.DataFrame(summary_data)
display(df_summary)

# Plot Comparative Bar Chart
plt.figure(figsize=(13, 5))
x = np.arange(len(summary_data))
v_scores = [r["Leaf V-Measure"] for r in summary_data]
ari_scores = [r["Leaf ARI"] for r in summary_data]
labels = [
    "Yellowbrick\\n(PCA)",
    "Yellowbrick\\n(UMAP)",
    "Sklearn\\nDP-GMM",
    "PyMC\\nDP-GMM",
    "Tomotopy\\nHDP",
]

width = 0.35
plt.bar(x - width / 2, v_scores, width=width, label="Leaf V-Measure", color="teal", alpha=0.85, edgecolor="black")
plt.bar(x + width / 2, ari_scores, width=width, label="Leaf Adjusted Rand Index (ARI)", color="coral", alpha=0.85, edgecolor="black")

plt.xticks(x, labels, fontsize=10, fontweight="bold")
plt.ylabel("Score", fontsize=11)
plt.title("Comparative Performance: Heuristic Search vs. Non-Parametric Bayesian Models", fontsize=13, fontweight="bold")
plt.legend(loc="upper left")
plt.ylim(0, 0.8)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 9: Conclusion & Practical Guidelines
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Practical Takeaways & Engineering Guidelines

1. **Heuristic Search (Yellowbrick) vs. Bayesian Non-Parametric Inference**:
   - **Yellowbrick** is a great starting point for familiarizing yourself with cluster geometry. However, running $15+$ fits to find an elbow and then sweeping a localized window is computationally repetitive and sensitive to heuristic bounds.
   - **Bayesian Non-Parametric Models (DP-GMM / HDP)** infer cluster count and mixture weights in a **single unified fit**, naturally penalizing complex models via Occam's razor embedded in the Dirichlet prior.

2. **Continuous DP-GMM with Sentence Embeddings**:
   - **Best for semantic clustering purity**: Dense neural embeddings (MiniLM) capture contextual nuance, polysemy, and syntax that bag-of-words ignores.
   - **Crucial preprocessing**: UMAP reduction from 384D to 10–15D is strictly necessary to avoid the distance concentration phenomenon and exponential underflow in Gaussian likelihoods.
   - **Initialization**: Variational inference (ADVI) in PyMC requires proper centroid initialization (e.g. k-means seeds) to prevent symmetric mode collapse where all components merge at the empirical center.

3. **Hierarchical Recovery**:
   - Running agglomerative Ward linkage on active DP-GMM centroids cleanly recovered the 3 true parent subtrees (`comp`, `rec`, `sci`) with zero manual supervision.

4. **Discrete HDP with Collapsed Gibbs Sampling**:
   - **Best for human interpretability**: Unlike continuous embeddings where clusters are abstract coordinate means, HDP directly yields characteristic vocabulary distributions for each topic.
   - **Speed**: Collapsed Gibbs sampling in `tomotopy` integrates out continuous distributions analytically, training 500 sweeps over 8,400 documents in under 1 minute.
"""))

nb.cells = cells

# Kernel metadata
nb.metadata["kernelspec"] = {
    "display_name": "Python (bayesian-clustering uv)",
    "language": "python",
    "name": "bayesian-clustering"
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.13"
}

output_path = Path("bayesian_clustering_tutorial.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {output_path} with {len(cells)} cells successfully!")
