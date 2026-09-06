"""Data preparation, embedding, and UMAP dimensionality reduction pipeline.

Phase 1 of the Non-Parametric Bayesian Clustering project.
"""

import json
import logging
from dataclasses import asdict
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import umap

from src.config import PipelineConfig, default_config, DATA_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_and_clean_data(
    config: PipelineConfig = default_config
) -> Dict[str, Any]:
    """Fetch and clean the 20 Newsgroups dataset subset.

    Removes headers, footers, and quotes to prevent models from learning trivial
    metadata artifacts (such as email domain names or poster names) rather than
    underlying semantic topics. Drops empty or trivial posts.

    Args:
        config: Configuration instance containing category list and cleaning options.

    Returns:
        Dictionary containing cleaned texts, fine category names and target IDs,
        and parent category names and target IDs.
    """
    logger.info("Fetching 20 Newsgroups subset for categories: %s", config.categories)
    raw_data = fetch_20newsgroups(
        subset="all",
        categories=config.categories,
        remove=config.remove_sections,
        shuffle=True,
        random_state=config.random_seed,
    )

    cleaned_texts: List[str] = []
    fine_labels: List[int] = []
    parent_labels: List[int] = []

    # Map unique parent categories to integers
    unique_parents = sorted(list(set(config.parent_mapping.values())))
    parent_to_id = {p: idx for idx, p in enumerate(unique_parents)}

    for text, label_idx in zip(raw_data.data, raw_data.target):
        # Normalize whitespace and strip
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) < config.min_doc_length_chars:
            continue

        cat_name = raw_data.target_names[label_idx]
        parent_name = config.parent_mapping[cat_name]

        cleaned_texts.append(cleaned)
        fine_labels.append(label_idx)
        parent_labels.append(parent_to_id[parent_name])

    logger.info(
        "Data cleaned: %d valid documents retained across %d categories and %d parents.",
        len(cleaned_texts),
        len(raw_data.target_names),
        len(unique_parents),
    )

    return {
        "texts": cleaned_texts,
        "fine_labels": np.array(fine_labels, dtype=np.int32),
        "fine_names": list(raw_data.target_names),
        "parent_labels": np.array(parent_labels, dtype=np.int32),
        "parent_names": unique_parents,
    }


def generate_embeddings(
    texts: List[str],
    config: PipelineConfig = default_config
) -> np.ndarray:
    """Generate dense semantic representations using a lightweight Transformer.

    Args:
        texts: List of preprocessed text documents.
        config: Configuration containing model name and normalization setting.

    Returns:
        Dense embedding matrix of shape (N, 384).
    """
    logger.info("Generating sentence embeddings using '%s'...", config.embedding_model_name)
    model = SentenceTransformer(config.embedding_model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
    )
    logger.info("Embeddings generated with shape: %s", embeddings.shape)
    return embeddings


def reduce_dimensions_umap(
    embeddings: np.ndarray,
    n_components: int = 12,
    config: PipelineConfig = default_config
) -> np.ndarray:
    """Project high-dimensional embeddings into a lower-dimensional Riemannian manifold.

    Why UMAP is mathematically necessary before Bayesian Gaussian Mixtures (PyMC & sklearn):
        1. Curse of Dimensionality: In 384 dimensions, data points become equidistant under
           the distance concentration phenomenon. Gaussian mixture probability densities:
           p(x | mu_k, Sigma_k) = (2pi)^{-D/2} |Sigma_k|^{-1/2} exp(-0.5 (x - mu_k)^T Sigma_k^{-1} (x - mu_k))
           suffer severe floating-point underflow due to the (2pi)^{-D/2} factor and extreme quadratic distances.
        2. Covariance Condition Number: Estimating even spherical covariance scales poorly with high D,
           often leading to near-singular variance estimates and unstable posterior updates.
        3. Manifold Preservation: UMAP preserves both local semantic neighborhoods (high-dimensional cosine
           similarities) and global topological cluster groupings into 10-15 continuous dimensions,
           where spherical Gaussian likelihood assumptions remain well-conditioned.

    Args:
        embeddings: Normalized embedding array of shape (N, D_orig).
        n_components: Target dimensionality (typically 10-15).
        config: Configuration containing metric, n_neighbors, and random_seed.

    Returns:
        Reduced coordinates of shape (N, n_components).
    """
    logger.info(
        "Applying UMAP reduction: %d dimensions -> %d dimensions (metric=%s)...",
        embeddings.shape[1],
        n_components,
        config.umap_metric,
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        metric=config.umap_metric,
        random_state=config.random_seed,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete. Output shape: %s", reduced.shape)
    return reduced


def plot_and_save_projections(
    umap_2d: np.ndarray,
    fine_labels: np.ndarray,
    fine_names: List[str],
    parent_labels: np.ndarray,
    parent_names: List[str],
    save_path: str,
) -> None:
    """Save 2D diagnostic visualization showing ground truth subcategories and parent categories."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Parent categories (top-level hierarchy)
    scatter_parent = axes[0].scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=parent_labels,
        cmap="tab10",
        alpha=0.6,
        s=15,
    )
    axes[0].set_title("Ground Truth Parent Categories (comp.*, rec.*, sci.*)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    handles_p, _ = scatter_parent.legend_elements()
    axes[0].legend(handles_p, parent_names, title="Parent Category", loc="best", framealpha=0.8)

    # Plot 2: Fine-grained subcategories
    scatter_fine = axes[1].scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=fine_labels,
        cmap="tab20",
        alpha=0.6,
        s=15,
    )
    axes[1].set_title("Ground Truth Fine-Grained Subcategories (9 Leaf Topics)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    handles_f, _ = scatter_fine.legend_elements()
    axes[1].legend(handles_f, fine_names, title="Subcategory", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved 2D diagnostic visualization to %s", save_path)


def run_phase_1(
    config: PipelineConfig = default_config,
    save_cache: bool = True
) -> Dict[str, Any]:
    """Execute complete Phase 1 pipeline: fetch, clean, embed, reduce, and cache.

    Args:
        config: Configuration parameters.
        save_cache: If True, caches embeddings and coordinates to disk.

    Returns:
        Dictionary containing all processed outputs.
    """
    logger.info("=== Starting Phase 1: Data Preparation & Embeddings Pipeline ===")
    
    # 1. Fetch & clean
    data_dict = fetch_and_clean_data(config)
    texts = data_dict["texts"]
    fine_labels = data_dict["fine_labels"]
    fine_names = data_dict["fine_names"]
    parent_labels = data_dict["parent_labels"]
    parent_names = data_dict["parent_names"]

    # 2. Dense embeddings (384D)
    embeddings = generate_embeddings(texts, config)

    # 3. UMAP reduction for Gaussian likelihoods (~12D)
    umap_embeddings = reduce_dimensions_umap(
        embeddings,
        n_components=config.umap_n_components,
        config=config,
    )

    # 4. 2D projection for visualization
    umap_2d = reduce_dimensions_umap(
        embeddings,
        n_components=2,
        config=config,
    )

    # 5. Diagnostic plot
    plot_path = str(ARTIFACTS_DIR / "phase1_umap_ground_truth.png")
    plot_and_save_projections(
        umap_2d,
        fine_labels,
        fine_names,
        parent_labels,
        parent_names,
        plot_path,
    )

    # 6. Cache to disk
    if save_cache:
        cache_file = DATA_DIR / "phase1_processed.npz"
        np.savez_compressed(
            cache_file,
            embeddings=embeddings,
            umap_embeddings=umap_embeddings,
            umap_2d=umap_2d,
            fine_labels=fine_labels,
            parent_labels=parent_labels,
        )
        metadata = {
            "num_documents": len(texts),
            "fine_names": fine_names,
            "parent_names": parent_names,
            "categories": config.categories,
            "umap_n_components": config.umap_n_components,
            "embedding_dim": embeddings.shape[1],
            "embedding_model": config.embedding_model_name,
        }
        with open(DATA_DIR / "phase1_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
        with open(DATA_DIR / "phase1_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f)
            
        logger.info("Cached processed dataset to %s and metadata to %s", cache_file, DATA_DIR / "phase1_metadata.json")

    logger.info("=== Phase 1 Pipeline Completed Successfully ===")
    return {
        "texts": texts,
        "fine_labels": fine_labels,
        "fine_names": fine_names,
        "parent_labels": parent_labels,
        "parent_names": parent_names,
        "embeddings": embeddings,
        "umap_embeddings": umap_embeddings,
        "umap_2d": umap_2d,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_phase_1()
