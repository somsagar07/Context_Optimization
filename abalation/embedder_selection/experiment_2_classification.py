"""
Experiment 2: Question Type Classification Test

Tests if embeddings can distinguish problem types (domain, tool requirements).
RL needs embeddings that capture these distinctions to make appropriate decisions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedders import BaseEmbedder
from embedding_utils import load_questions_from_datasets, get_tool_labels
from config import DATASETS, MAX_SAMPLES_PER_DATASET, EXPERIMENT_SETTINGS, TARGET_EMBEDDING_DIM


def run_classification_experiment(embedder: BaseEmbedder, embedder_name: str) -> Dict:
    """
    Test classification accuracy on embeddings.
    
    Classification tasks:
    1. Dataset classification (gsm8k, hotpotqa, etc.)
    2. Tool requirement classification (calculator, web_search, python, none)
    """
    print(f"\n{'='*60}")
    print(f"Experiment 2: Classification - {embedder_name}")
    print(f"{'='*60}")
    
    # Load questions
    questions, answers, dataset_labels = load_questions_from_datasets(
        DATASETS, 
        max_samples=MAX_SAMPLES_PER_DATASET
    )
    
    if len(questions) < 20:
        print("  Warning: Not enough samples for classification")
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
    
    # Task 1: Dataset classification
    print(f"\nTask 1: Dataset Classification")
    le_dataset = LabelEncoder()
    dataset_labels_encoded = le_dataset.fit_transform(dataset_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, dataset_labels_encoded,
        test_size=EXPERIMENT_SETTINGS["classification"]["test_size"],
        random_state=EXPERIMENT_SETTINGS["classification"]["random_state"],
        stratify=dataset_labels_encoded
    )
    
    clf_dataset = LogisticRegression(max_iter=1000, random_state=42)
    clf_dataset.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(
        clf_dataset, embeddings, dataset_labels_encoded,
        cv=EXPERIMENT_SETTINGS["classification"]["cv_folds"],
        scoring='accuracy'
    )
    
    y_pred = clf_dataset.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    
    # Task 2: Tool classification
    print(f"\nTask 2: Tool Requirement Classification")
    tool_labels = get_tool_labels(questions, answers)
    le_tool = LabelEncoder()
    tool_labels_encoded = le_tool.fit_transform(tool_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, tool_labels_encoded,
        test_size=EXPERIMENT_SETTINGS["classification"]["test_size"],
        random_state=EXPERIMENT_SETTINGS["classification"]["random_state"],
        stratify=tool_labels_encoded
    )
    
    clf_tool = LogisticRegression(max_iter=1000, random_state=42)
    clf_tool.fit(X_train, y_train)
    
    cv_scores_tool = cross_val_score(
        clf_tool, embeddings, tool_labels_encoded,
        cv=EXPERIMENT_SETTINGS["classification"]["cv_folds"],
        scoring='accuracy'
    )
    
    y_pred_tool = clf_tool.predict(X_test)
    test_accuracy_tool = accuracy_score(y_test, y_pred_tool)
    test_f1_tool = f1_score(y_test, y_pred_tool, average='weighted')
    
    print(f"  CV Accuracy: {cv_scores_tool.mean():.4f} (+/- {cv_scores_tool.std() * 2:.4f})")
    print(f"  Test Accuracy: {test_accuracy_tool:.4f}")
    print(f"  Test F1: {test_f1_tool:.4f}")
    
    results = {
        "embedder": embedder_name,
        "n_samples": len(questions),
        "dataset_classification": {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(test_accuracy),
            "test_f1": float(test_f1),
        },
        "tool_classification": {
            "cv_accuracy_mean": float(cv_scores_tool.mean()),
            "cv_accuracy_std": float(cv_scores_tool.std()),
            "test_accuracy": float(test_accuracy_tool),
            "test_f1": float(test_f1_tool),
        },
        "average_accuracy": float((test_accuracy + test_accuracy_tool) / 2),
        "average_f1": float((test_f1 + test_f1_tool) / 2),
    }
    
    return results


if __name__ == "__main__":
    from embedders import create_embedder
    
    embedder = create_embedder("sentence-all-MiniLM-L6-v2", target_dim=TARGET_EMBEDDING_DIM)
    results = run_classification_experiment(embedder, "all-MiniLM-L6-v2")
    print("\nResults:", json.dumps(results, indent=2))

