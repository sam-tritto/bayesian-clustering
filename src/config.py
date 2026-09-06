"""Configuration settings for Bayesian clustering experiments."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# Workspace root and data directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Central configuration for datasets, embedding models, and clustering models."""
    
    # 20 Newsgroups subset with explicit 2-level parent-child hierarchy
    categories: List[str] = field(
        default_factory=lambda: [
            # Parent: comp
            "comp.graphics",
            "comp.os.ms-windows.misc",
            "comp.sys.mac.hardware",
            # Parent: rec
            "rec.autos",
            "rec.motorcycles",
            "rec.sport.baseball",
            # Parent: sci
            "sci.crypt",
            "sci.med",
            "sci.space",
        ]
    )
    
    # Hierarchy mapping from child topic to parent category
    parent_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "comp.graphics": "comp",
            "comp.os.ms-windows.misc": "comp",
            "comp.sys.mac.hardware": "comp",
            "rec.autos": "rec",
            "rec.motorcycles": "rec",
            "rec.sport.baseball": "rec",
            "sci.crypt": "sci",
            "sci.med": "sci",
            "sci.space": "sci",
        }
    )
    
    # Text cleaning
    remove_sections: tuple = ("headers", "footers", "quotes")
    min_doc_length_chars: int = 40
    
    # Embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    
    # Dimensionality Reduction (UMAP)
    # Reducing 384D -> 12D avoids curse of dimensionality in Gaussian mixture likelihoods
    umap_n_components: int = 12
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    
    # Bayesian Gaussian Mixture (DP-GMM) defaults
    truncation_k: int = 30
    weight_concentration_prior: float = 1.0  # Dirichlet process alpha
    weight_threshold: float = 0.01          # Minimum weight to consider cluster active
    covariance_type: str = "spherical"      # Enforce spherical covariance for numerical stability
    
    # Random reproducibility
    random_seed: int = 42


default_config = PipelineConfig()
