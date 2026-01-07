"""
Experiment 3: Complexity Ranking Test

Tests if embeddings capture problem complexity/difficulty.
RL needs this to choose appropriate workflows and budgets.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import json
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedders import BaseEmbedder
from embedding_utils import load_questions_from_datasets, get_complexity_labels
from config import DATASETS, MAX_SAMPLES_PER_DATASET, EXPERIMENT_SETTINGS, TARGET_EMBEDDING_DIM


def run_complexity_experiment(embedder: BaseEmbedder, embedder_name: str) -> Dict:
    """
    Test if embeddings can predict problem complexity.
    
    Returns:
        Dictionary with ranking and regression metrics
    """
    print(f"\n{'='*60}")
    print(f"Experiment 3: Complexity Ranking - {embedder_name}")
    print(f"{'='*60}")
    
    # Load questions
    questions, answers, _ = load_questions_from_datasets(
        DATASETS, 
        max_samples=MAX_SAMPLES_PER_DATASET
    )
    
    if len(questions) < 20:
        print("  Warning: Not enough samples for complexity regression")
        return {"error": "Insufficient samples"}
    
    print(f"\nEmbedding {len(questions)} questions...")
    
    # Generate embeddings with progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
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
            embeddings.append(np.zeros(embedder.get_dimension()))
    
    embeddings = np.array(embeddings)
    
    # Get complexity labels
    complexity_scores = get_complexity_labels(questions)
    
    print(f"\nComplexity scores - Min: {complexity_scores.min():.3f}, Max: {complexity_scores.max():.3f}, Mean: {complexity_scores.mean():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, complexity_scores,
        test_size=EXPERIMENT_SETTINGS["complexity"]["test_size"],
        random_state=EXPERIMENT_SETTINGS["complexity"]["random_state"]
    )
    
    # Train regressor
    regressor = Ridge(alpha=1.0, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    # Regression metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Ranking metrics (Spearman correlation)
    spearman_train = spearmanr(y_train, y_pred_train)[0]
    spearman_test = spearmanr(y_test, y_pred_test)[0]
    pearson_test = pearsonr(y_test, y_pred_test)[0]
    
    # Ranking accuracy: % of pairs correctly ranked
    def ranking_accuracy(true_scores, pred_scores):
        n = len(true_scores)
        if n < 2:
            return 0.0
        correct = 0
        total = 0
        for i in range(n):
            for j in range(i+1, n):
                true_order = true_scores[i] > true_scores[j]
                pred_order = pred_scores[i] > pred_scores[j]
                if true_order == pred_order:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0
    
    rank_acc_train = ranking_accuracy(y_train, y_pred_train)
    rank_acc_test = ranking_accuracy(y_test, y_pred_test)
    
    results = {
        "embedder": embedder_name,
        "n_samples": len(questions),
        "regression": {
            "train_mse": float(train_mse),
            "test_mse": float(test_mse),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
        },
        "ranking": {
            "spearman_train": float(spearman_train) if not np.isnan(spearman_train) else 0.0,
            "spearman_test": float(spearman_test) if not np.isnan(spearman_test) else 0.0,
            "pearson_test": float(pearson_test) if not np.isnan(pearson_test) else 0.0,
            "ranking_accuracy_train": float(rank_acc_train),
            "ranking_accuracy_test": float(rank_acc_test),
        },
        # Combined score (higher is better)
        "complexity_score": float((spearman_test + test_r2) / 2),
    }
    
    print(f"\nRegression Results:")
    print(f"  Test MSE: {test_mse:.4f} (lower = better)")
    print(f"  Test R²: {test_r2:.4f} (higher = better)")
    
    print(f"\nRanking Results:")
    print(f"  Spearman Correlation (test): {spearman_test:.4f} (higher = better ranking)")
    print(f"  Pearson Correlation (test): {pearson_test:.4f}")
    print(f"  Ranking Accuracy (test): {rank_acc_test:.4f} (higher = better)")
    print(f"  Combined Complexity Score: {results['complexity_score']:.4f}")
    
    return results


if __name__ == "__main__":
    from embedders import create_embedder
    
    embedder = create_embedder("sentence-all-MiniLM-L6-v2", target_dim=TARGET_EMBEDDING_DIM)
    results = run_complexity_experiment(embedder, "all-MiniLM-L6-v2")
    print("\nResults:", json.dumps(results, indent=2))

