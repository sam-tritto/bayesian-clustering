"""Unified execution and benchmarking pipeline for all Bayesian clustering phases.

Compares:
    - Phase 1: Embedding generation & UMAP dimensionality reduction
    - Phase 2: Continuous DP-GMM baseline (scikit-learn) + Agglomerative Hierarchy
    - Phase 3: Continuous DP-GMM native PyMC stick-breaking + Log-Sum-Exp
    - Phase 4: Discrete Hierarchical Dirichlet Process (tomotopy Collapsed Gibbs)
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from src.config import default_config, ARTIFACTS_DIR
from src.data.pipeline import run_phase_1
from src.models.sklearn_dpgmm import run_phase_2
from src.models.pymc_dpgmm import run_phase_3
from src.models.tomotopy_hdp import run_phase_4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_comparative_summary():
    """Load results from all phases and compile a unified comparative markdown table and plot."""
    p2_file = ARTIFACTS_DIR / "phase2_results.json"
    p3_file = ARTIFACTS_DIR / "phase3_results.json"
    p4_file = ARTIFACTS_DIR / "phase4_results.json"

    records = []
    if p2_file.exists():
        with open(p2_file, "r") as f:
            p2 = json.load(f)
        records.append({
            "Model": "Phase 2: DP-GMM (scikit-learn)",
            "Paradigm": "Continuous (12D UMAP)",
            "Inference": "Variational EM",
            "Discovered Clusters": p2["active_clusters_discovered"],
            "Leaf V-Measure": p2["leaf_evaluation"]["v_measure"],
            "Leaf ARI": p2["leaf_evaluation"]["ari"],
            "Parent V-Measure": p2["parent_hierarchy_evaluation"]["v_measure"],
            "Parent ARI": p2["parent_hierarchy_evaluation"]["ari"],
        })

    if p3_file.exists():
        with open(p3_file, "r") as f:
            p3 = json.load(f)
        records.append({
            "Model": "Phase 3: DP-GMM (PyMC Stick-Breaking)",
            "Paradigm": "Continuous (12D UMAP)",
            "Inference": "PyMC ADVI + LogSumExp",
            "Discovered Clusters": p3["active_clusters_discovered"],
            "Leaf V-Measure": p3["leaf_evaluation"]["v_measure"],
            "Leaf ARI": p3["leaf_evaluation"]["ari"],
            "Parent V-Measure": "N/A (Flat Mixture)",
            "Parent ARI": "N/A (Flat Mixture)",
        })

    if p4_file.exists():
        with open(p4_file, "r") as f:
            p4 = json.load(f)
        records.append({
            "Model": "Phase 4: HDP (tomotopy)",
            "Paradigm": "Discrete BoW Tokens",
            "Inference": "Collapsed Gibbs Sampling",
            "Discovered Clusters": p4["total_discovered_live_topics"],
            "Leaf V-Measure": p4["leaf_evaluation"]["v_measure"],
            "Leaf ARI": p4["leaf_evaluation"]["ari"],
            "Parent V-Measure": p4["parent_hierarchy_evaluation"]["v_measure"],
            "Parent ARI": p4["parent_hierarchy_evaluation"]["ari"],
        })

    df = pd.DataFrame(records)
    logger.info("\n%s", df.to_string(index=False))

    # Save summary table
    df.to_json(ARTIFACTS_DIR / "model_comparison_summary.json", orient="records", indent=2)

    # Plot comparative metrics
    if len(records) > 0:
        plt.figure(figsize=(10, 5))
        x = range(len(records))
        v_scores = [r["Leaf V-Measure"] for r in records]
        ari_scores = [r["Leaf ARI"] for r in records]
        models = [r["Model"].split(":")[0] for r in records]

        width = 0.35
        plt.bar([i - width / 2 for i in x], v_scores, width=width, label="Leaf V-Measure", color="teal", alpha=0.85, edgecolor="black")
        plt.bar([i + width / 2 for i in x], ari_scores, width=width, label="Leaf Adjusted Rand Index (ARI)", color="coral", alpha=0.85, edgecolor="black")

        plt.xticks(x, models, fontsize=11, fontweight="bold")
        plt.ylabel("Score", fontsize=11)
        plt.title("Comparative Performance: Non-Parametric Bayesian Clustering", fontsize=13, fontweight="bold")
        plt.legend(loc="upper left")
        plt.ylim(0, 0.8)
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "all_phases_comparison.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Clustering Experiments")
    parser.add_argument("--phase", type=str, default="summary", choices=["1", "2", "3", "4", "all", "summary"], help="Phase to execute")
    args = parser.parse_args()

    if args.phase in ["1", "all"]:
        run_phase_1()
    if args.phase in ["2", "all"]:
        run_phase_2()
    if args.phase in ["3", "all"]:
        run_phase_3()
    if args.phase in ["4", "all"]:
        run_phase_4()

    generate_comparative_summary()


if __name__ == "__main__":
    main()
