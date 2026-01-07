"""
Experiment 1: Clustering Quality Test

Measures how well embeddings cluster similar questions together.
RL needs embeddings that group similar problems for consistent decision-making.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import json
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedders import BaseEmbedder
from embedding_utils import load_questions_from_datasets, get_tool_labels
from config import DATASETS, MAX_SAMPLES_PER_DATASET, EXPERIMENT_SETTINGS, TARGET_EMBEDDING_DIM


def run_clustering_experiment(embedder: BaseEmbedder, embedder_name: str) -> Dict:
    """
    Test clustering quality of embeddings.
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print(f"Experiment 1: Clustering Quality - {embedder_name}")
    print(f"{'='*60}")
    
    # Load questions
    questions, answers, dataset_labels = load_questions_from_datasets(
        DATASETS, 
        max_samples=MAX_SAMPLES_PER_DATASET
    )
    
    if len(questions) < 10:
        print("  Warning: Not enough samples for clustering")
        return {"error": "Insufficient samples"}
    
    print(f"\nEmbedding {len(questions)} questions...")
    
    # Generate embeddings with progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("  (Install tqdm for progress bars: pip install tqdm)")
    
    embeddings = []
    iterator = tqdm(questions, desc="  Embedding") if use_tqdm else questions
    for i, q in enumerate(iterator):
        try:
            emb = embedder.embed(q)
            embeddings.append(emb)
        except Exception as e:
            if use_tqdm:
                tqdm.write(f"  Error embedding question {i}: {e}")
            else:
                print(f"  Error embedding question {i}: {e}")
            # Use zero vector as fallback
            embeddings.append(np.zeros(embedder.get_dimension()))
    
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Create ground truth labels (dataset + tool type)
    tool_labels = get_tool_labels(questions, answers)
    combined_labels = [f"{d}_{t}" for d, t in zip(dataset_labels, tool_labels)]
    
    # Get unique labels for ARI
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    true_labels = le.fit_transform(combined_labels)
    n_clusters = min(EXPERIMENT_SETTINGS["clustering"]["n_clusters"], len(np.unique(true_labels)))
    
    # K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=EXPERIMENT_SETTINGS["clustering"]["random_state"],
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    
    # Intra-cluster similarity (average cosine similarity within clusters)
    intra_cluster_sims = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        if np.sum(cluster_mask) < 2:
            continue
        cluster_embeds = embeddings[cluster_mask]
        # Compute average pairwise similarity within cluster
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(cluster_embeds)
        # Exclude diagonal (self-similarity)
        mask = ~np.eye(len(sims), dtype=bool)
        intra_cluster_sims.append(sims[mask].mean())
    
    avg_intra_cluster_sim = np.mean(intra_cluster_sims) if intra_cluster_sims else 0.0
    
    # Inter-cluster separation (average distance between cluster centroids)
    centroids = kmeans.cluster_centers_
    from sklearn.metrics.pairwise import cosine_similarity
    centroid_sims = cosine_similarity(centroids)
    mask = ~np.eye(len(centroid_sims), dtype=bool)
    avg_inter_cluster_sim = centroid_sims[mask].mean()
    # Lower inter-cluster similarity = better separation
    
    results = {
        "embedder": embedder_name,
        "n_samples": len(questions),
        "n_clusters": n_clusters,
        "silhouette_score": float(silhouette),
        "adjusted_rand_index": float(ari),
        "avg_intra_cluster_similarity": float(avg_intra_cluster_sim),
        "avg_inter_cluster_similarity": float(avg_inter_cluster_sim),
        "cluster_separation": float(avg_intra_cluster_sim - avg_inter_cluster_sim),  # Higher is better
    }
    
    print(f"\nResults:")
    print(f"  Silhouette Score: {silhouette:.4f} (higher = better)")
    print(f"  Adjusted Rand Index: {ari:.4f} (higher = better, vs ground truth)")
    print(f"  Intra-cluster Similarity: {avg_intra_cluster_sim:.4f} (higher = tighter clusters)")
    print(f"  Inter-cluster Similarity: {avg_inter_cluster_sim:.4f} (lower = better separation)")
    print(f"  Cluster Separation: {results['cluster_separation']:.4f} (higher = better)")
    
    return results


if __name__ == "__main__":
    # Test with a single embedder
    from embedders import create_embedder
    
    embedder = create_embedder("sentence-all-MiniLM-L6-v2", target_dim=TARGET_EMBEDDING_DIM)
    results = run_clustering_experiment(embedder, "all-MiniLM-L6-v2")
    print("\nResults:", json.dumps(results, indent=2))

