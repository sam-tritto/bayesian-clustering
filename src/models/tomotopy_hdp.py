"""Discrete Hierarchical Dirichlet Process (HDP) using Collapsed Gibbs Sampling (tomotopy).

Phase 4 of the Non-Parametric Bayesian Clustering project.
"""

import re
import json
import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import tomotopy as tp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)

from src.config import PipelineConfig, default_config, DATA_DIR, ARTIFACTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def preprocess_text_for_hdp(
    texts: List[str],
    min_word_len: int = 3,
    min_doc_tokens: int = 5,
) -> List[List[str]]:
    """Tokenize and clean raw documents for discrete topic modeling.

    Steps:
        1. Extract alphabetic tokens (length >= min_word_len) to eliminate numbers,
           punctuation, and encoding artifacts.
        2. Filter standard English stopwords.
        3. Exclude trivial docs with fewer than min_doc_tokens.

    Args:
        texts: List of cleaned text documents from Phase 1.
        min_word_len: Minimum word character length.
        min_doc_tokens: Minimum tokens per document.

    Returns:
        List of tokenized documents (list of word strings).
    """
    logger.info("Tokenizing %d documents for HDP topic modeling...", len(texts))
    token_pattern = re.compile(r"\b[a-zA-Z]{%d,}\b" % min_word_len)
    stop_set = set(ENGLISH_STOP_WORDS).union({
        "subject", "lines", "organization", "writes", "article",
        "nntp", "posting", "host", "university", "reply", "thanks",
    })

    tokenized_corpus: List[List[str]] = []
    for doc in texts:
        words = token_pattern.findall(doc.lower())
        meaningful = [w for w in words if w not in stop_set]
        tokenized_corpus.append(meaningful if len(meaningful) >= min_doc_tokens else ["empty"])

    logger.info("Tokenization complete. %d documents prepared.", len(tokenized_corpus))
    return tokenized_corpus


def train_hdp_gibbs(
    tokenized_corpus: List[List[str]],
    initial_k: int = 30,
    alpha: float = 0.1,
    gamma: float = 1.0,
    total_iterations: int = 500,
    step_size: int = 25,
    random_seed: int = 42,
) -> Tuple[tp.HDPModel, List[Dict[str, float]]]:
    """Train Hierarchical Dirichlet Process model using Collapsed Gibbs Sampling.

    Mathematical Formulation:
        - The Hierarchical Dirichlet Process (Teh et al., 2006) couples multiple Dirichlet
          Processes across documents through a shared global base measure G_0 ~ DP(gamma, H).
        - Document-specific base measures G_d ~ DP(alpha, G_0) share topic atoms across the corpus,
          enabling both the discovery of an unbounded number of global topics and document-level
          topic proportions without setting a fixed number of topics K.
        - Collapsed Gibbs Sampling analytically integrates out the continuous topic distributions
          (Dirichlet components) and samples only the discrete table and topic assignment variables,
          achieving faster convergence and avoiding local variational mode collapse.

    Args:
        tokenized_corpus: List of word lists.
        initial_k: Initial number of candidate topics.
        alpha: Concentration parameter for document-level Dirichlet process.
        gamma: Concentration parameter for corpus-level global Dirichlet process.
        total_iterations: Total Gibbs sampling sweeps.
        step_size: Interval for logging likelihood and live topic count.
        random_seed: Seed for sampling reproducibility.

    Returns:
        Tuple of (trained HDPModel, training_history).
    """
    logger.info(
        "Initializing tomotopy.HDPModel (initial_k=%d, alpha=%.2f, gamma=%.2f)...",
        initial_k,
        alpha,
        gamma,
    )
    hdp = tp.HDPModel(
        tw=tp.TermWeight.IDF,
        min_cf=3,
        rm_top=5,
        alpha=alpha,
        gamma=gamma,
        initial_k=initial_k,
        seed=random_seed,
    )

    for doc in tokenized_corpus:
        hdp.add_doc(doc)

    logger.info("Added %d documents into HDP corpus. Starting Collapsed Gibbs Sampling...", len(hdp.docs))

    history: List[Dict[str, float]] = []
    for i in range(0, total_iterations, step_size):
        hdp.train(step_size)
        ll = float(hdp.ll_per_word)
        live_k = int(hdp.live_k)
        history.append({
            "iteration": i + step_size,
            "ll_per_word": ll,
            "live_topics": live_k,
        })
        if (i + step_size) % 100 == 0 or (i + step_size) == total_iterations:
            logger.info("Gibbs Iteration %4d / %d | LL/word: %.4f | Live Topics: %d", i + step_size, total_iterations, ll, live_k)

    return hdp, history


def extract_prominent_topics(
    hdp: tp.HDPModel,
    top_n_words: int = 5,
    top_k_topics: int = 15,
) -> List[Dict[str, Any]]:
    """Extract and rank active topics by document/word assignment volume."""
    counts = hdp.get_count_by_topics()
    live_topics = [k for k in range(hdp.k) if hdp.is_live_topic(k)]
    # Sort live topics by count descending
    live_topics_sorted = sorted(live_topics, key=lambda k: counts[k], reverse=True)

    topic_summaries = []
    for rank, k in enumerate(live_topics_sorted[:top_k_topics]):
        top_words = hdp.get_topic_words(k, top_n=top_n_words)
        topic_summaries.append({
            "rank": rank + 1,
            "topic_id": int(k),
            "word_count": int(counts[k]),
            "top_words": [word for word, prob in top_words],
            "word_probs": [float(prob) for word, prob in top_words],
        })

    return topic_summaries


def evaluate_hdp_clustering(
    hdp: tp.HDPModel,
    fine_labels: np.ndarray,
    parent_labels: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    """Evaluate document topic assignments against ground truth hierarchies."""
    doc_topics = []
    for doc in hdp.docs:
        topic_dist = doc.get_topic_dist()
        doc_topics.append(int(np.argmax(topic_dist)))

    pred_labels = np.array(doc_topics, dtype=np.int32)

    leaf_metrics = {
        "ari": float(adjusted_rand_score(fine_labels, pred_labels)),
        "ami": float(adjusted_mutual_info_score(fine_labels, pred_labels)),
        "v_measure": float(v_measure_score(fine_labels, pred_labels)),
        "homogeneity": float(homogeneity_score(fine_labels, pred_labels)),
        "completeness": float(completeness_score(fine_labels, pred_labels)),
    }

    parent_metrics = {
        "ari": float(adjusted_rand_score(parent_labels, pred_labels)),
        "ami": float(adjusted_mutual_info_score(parent_labels, pred_labels)),
        "v_measure": float(v_measure_score(parent_labels, pred_labels)),
        "homogeneity": float(homogeneity_score(parent_labels, pred_labels)),
        "completeness": float(completeness_score(parent_labels, pred_labels)),
    }

    return leaf_metrics, parent_metrics, pred_labels


def plot_phase4_diagnostics(
    history: List[Dict[str, float]],
    prominent_topics: List[Dict[str, Any]],
) -> None:
    """Generate diagnostic plots for Phase 4 HDP training and topics."""
    # Plot 1: Training convergence (Log-Likelihood & Live Topics)
    iters = [h["iteration"] for h in history]
    lls = [h["ll_per_word"] for h in history]
    live_ks = [h["live_topics"] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    color = "tab:blue"
    ax1.set_xlabel("Collapsed Gibbs Sampling Iterations", fontsize=11)
    ax1.set_ylabel("Log-Likelihood per Word", color=color, fontsize=11)
    ax1.plot(iters, lls, color=color, linewidth=2, marker="o", markersize=4)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color = "tab:orange"
    ax2.set_ylabel("Discovered Live Topics (live_k)", color=color, fontsize=11)
    ax2.plot(iters, live_ks, color=color, linewidth=2, linestyle="--", marker="s", markersize=4)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("HDP Training Dynamics (Collapsed Gibbs Sampling)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase4_hdp_training_curves.png", dpi=200)
    plt.close()

    # Plot 2: Prominent Topics Top Words Barplot
    n_display = min(8, len(prominent_topics))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for i in range(n_display):
        top = prominent_topics[i]
        words = top["top_words"]
        probs = top["word_probs"]
        y_pos = np.arange(len(words))
        axes[i].barh(y_pos, probs, color="teal", alpha=0.8, edgecolor="black")
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(words, fontsize=10, fontweight="bold")
        axes[i].invert_yaxis()
        axes[i].set_title(f"Topic {top['topic_id']} ({top['word_count']} tokens)", fontsize=11)
        axes[i].set_xlabel("Probability")
        axes[i].grid(axis="x", linestyle=":", alpha=0.5)

    for j in range(n_display, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Top 5 Words for Most Prominent HDP Topics", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "phase4_hdp_prominent_topics.png", dpi=200)
    plt.close()


def run_phase_4(
    config: PipelineConfig = default_config,
    gibbs_iterations: int = 500,
) -> Dict[str, Any]:
    """Execute complete Phase 4: Tokenization, tomotopy HDP training, topic interpretation."""
    logger.info("=== Starting Phase 4: Discrete Hierarchical Dirichlet Process (Tomotopy) ===")

    # 1. Load Phase 1 processed data & raw texts
    texts_file = DATA_DIR / "phase1_texts.json"
    cache_file = DATA_DIR / "phase1_processed.npz"
    metadata_file = DATA_DIR / "phase1_metadata.json"

    with open(texts_file, "r", encoding="utf-8") as f:
        raw_texts = json.load(f)
    with np.load(cache_file) as data:
        fine_labels = data["fine_labels"]
        parent_labels = data["parent_labels"]
    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2. Tokenize corpus
    tokenized_docs = preprocess_text_for_hdp(raw_texts)

    # 3. Train HDP via Collapsed Gibbs Sampling
    hdp, history = train_hdp_gibbs(
        tokenized_corpus=tokenized_docs,
        initial_k=config.truncation_k,
        total_iterations=gibbs_iterations,
        random_seed=config.random_seed,
    )

    # 4. Extract prominent topics and top 5 words
    prominent = extract_prominent_topics(hdp, top_n_words=5, top_k_topics=12)

    # 5. Evaluate clustering alignment against ground truth
    leaf_metrics, parent_metrics, pred_labels = evaluate_hdp_clustering(
        hdp=hdp,
        fine_labels=fine_labels,
        parent_labels=parent_labels,
    )

    # 6. Generate diagnostic plots
    plot_phase4_diagnostics(history, prominent)

    # 7. Save results summary
    results_summary = {
        "gibbs_iterations": gibbs_iterations,
        "total_discovered_live_topics": int(hdp.live_k),
        "prominent_topics": prominent,
        "leaf_evaluation": leaf_metrics,
        "parent_hierarchy_evaluation": parent_metrics,
    }
    with open(ARTIFACTS_DIR / "phase4_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    logger.info("=== Phase 4 Results Summary ===")
    logger.info("Total Discovered Live Topics: %d", hdp.live_k)
    logger.info("Leaf Subcategory V-Measure: %.4f | ARI: %.4f", leaf_metrics["v_measure"], leaf_metrics["ari"])
    logger.info("Parent Macro-Hierarchy V-Measure: %.4f | ARI: %.4f", parent_metrics["v_measure"], parent_metrics["ari"])
    for top in prominent[:6]:
        logger.info("Topic %2d (%5d tokens): %s", top["topic_id"], top["word_count"], ", ".join(top["top_words"]))
    logger.info("=== Phase 4 Completed Successfully ===")

    return results_summary


if __name__ == "__main__":
    run_phase_4()
