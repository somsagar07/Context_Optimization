"""
Experiment 5: Decision Prediction Test

Tests if embeddings can predict optimal structure decisions (workflow, tools, budget).
This directly tests what the RL policy needs to learn.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import json
from typing import Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedders import BaseEmbedder
from embedding_utils import load_questions_from_datasets, get_tool_labels, get_workflow_labels
from config import DATASETS, MAX_SAMPLES_PER_DATASET, EXPERIMENT_SETTINGS, TARGET_EMBEDDING_DIM


class DecisionPredictor(nn.Module):
    """MLP to predict structure decisions from embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, workflow_dim: int, tool_dim: int, budget_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.workflow_head = nn.Linear(hidden_dim, workflow_dim)
        self.tool_head = nn.Linear(hidden_dim, tool_dim)
        self.budget_head = nn.Linear(hidden_dim, budget_dim)
    
    def forward(self, x):
        features = self.network(x)
        workflow_logits = self.workflow_head(features)
        tool_logits = self.tool_head(features)
        budget_logits = self.budget_head(features)
        return workflow_logits, tool_logits, budget_logits


def run_decision_prediction_experiment(embedder: BaseEmbedder, embedder_name: str) -> Dict:
    """
    Test if embeddings can predict optimal structure decisions.
    """
    print(f"\n{'='*60}")
    print(f"Experiment 5: Decision Prediction - {embedder_name}")
    print(f"{'='*60}")
    
    # Load questions
    questions, answers, _ = load_questions_from_datasets(
        DATASETS, 
        max_samples=MAX_SAMPLES_PER_DATASET
    )
    
    if len(questions) < 50:
        print("  Warning: Not enough samples for decision prediction")
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
    
    # Generate heuristic decision labels
    tool_labels = get_tool_labels(questions, answers)
    workflow_labels = get_workflow_labels(questions, answers, tool_labels)
    
    # Budget labels: 0=Low, 1=Mid, 2=High
    # Heuristic: based on question complexity
    complexity_scores = []
    for q in questions:
        word_count = len(q.split())
        if word_count < 30:
            complexity_scores.append(0)  # Low
        elif word_count < 80:
            complexity_scores.append(1)  # Mid
        else:
            complexity_scores.append(2)  # High
    
    budget_labels = np.array(complexity_scores)
    
    # Encode tool labels as integers
    from sklearn.preprocessing import LabelEncoder
    le_tool = LabelEncoder()
    tool_labels_encoded = le_tool.fit_transform(tool_labels)
    
    # Prepare data
    X = embeddings.astype(np.float32)
    y_workflow = np.array(workflow_labels)
    y_tool = tool_labels_encoded
    y_budget = budget_labels
    
    # Split
    X_train, X_test, y_wf_train, y_wf_test, y_tool_train, y_tool_test, y_budg_train, y_budg_test = train_test_split(
        X, y_workflow, y_tool, y_budget,
        test_size=EXPERIMENT_SETTINGS["decision_prediction"]["test_size"],
        random_state=EXPERIMENT_SETTINGS["decision_prediction"]["random_state"]
    )
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecisionPredictor(
        input_dim=X.shape[1],
        hidden_dim=EXPERIMENT_SETTINGS["decision_prediction"]["hidden_dim"],
        workflow_dim=len(np.unique(y_workflow)),
        tool_dim=len(np.unique(y_tool)),
        budget_dim=len(np.unique(y_budget))
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_wf_train_t = torch.LongTensor(y_wf_train).to(device)
    y_wf_test_t = torch.LongTensor(y_wf_test).to(device)
    y_tool_train_t = torch.LongTensor(y_tool_train).to(device)
    y_tool_test_t = torch.LongTensor(y_tool_test).to(device)
    y_budg_train_t = torch.LongTensor(y_budg_train).to(device)
    y_budg_test_t = torch.LongTensor(y_budg_test).to(device)
    
    # Training loop
    epochs = EXPERIMENT_SETTINGS["decision_prediction"]["epochs"]
    print(f"\nTraining decision predictor for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        wf_logits, tool_logits, budg_logits = model(X_train_t)
        
        loss_wf = criterion(wf_logits, y_wf_train_t)
        loss_tool = criterion(tool_logits, y_tool_train_t)
        loss_budg = criterion(budg_logits, y_budg_train_t)
        
        loss = loss_wf + loss_tool + loss_budg
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        wf_logits, tool_logits, budg_logits = model(X_test_t)
        
        wf_pred = wf_logits.argmax(dim=1).cpu().numpy()
        tool_pred = tool_logits.argmax(dim=1).cpu().numpy()
        budg_pred = budg_logits.argmax(dim=1).cpu().numpy()
    
    # Metrics
    workflow_acc = accuracy_score(y_wf_test, wf_pred)
    workflow_f1 = f1_score(y_wf_test, wf_pred, average='weighted')
    
    tool_acc = accuracy_score(y_tool_test, tool_pred)
    tool_f1 = f1_score(y_tool_test, tool_pred, average='weighted')
    
    budget_acc = accuracy_score(y_budg_test, budg_pred)
    budget_f1 = f1_score(y_budg_test, budg_pred, average='weighted')
    
    # Combined accuracy (all decisions correct)
    combined_correct = ((y_wf_test == wf_pred) & 
                        (y_tool_test == tool_pred) & 
                        (y_budg_test == budg_pred))
    combined_acc = combined_correct.mean()
    
    results = {
        "embedder": embedder_name,
        "n_samples": len(questions),
        "workflow_prediction": {
            "accuracy": float(workflow_acc),
            "f1": float(workflow_f1),
        },
        "tool_prediction": {
            "accuracy": float(tool_acc),
            "f1": float(tool_f1),
        },
        "budget_prediction": {
            "accuracy": float(budget_acc),
            "f1": float(budget_f1),
        },
        "combined_accuracy": float(combined_acc),
        "average_accuracy": float((workflow_acc + tool_acc + budget_acc) / 3),
        "decision_score": float((workflow_acc + tool_acc + budget_acc + combined_acc) / 4),
    }
    
    print(f"\nDecision Prediction Results:")
    print(f"  Workflow Accuracy: {workflow_acc:.4f}")
    print(f"  Tool Accuracy: {tool_acc:.4f}")
    print(f"  Budget Accuracy: {budget_acc:.4f}")
    print(f"  Combined Accuracy (all correct): {combined_acc:.4f}")
    print(f"  Average Accuracy: {results['average_accuracy']:.4f}")
    print(f"  Decision Score: {results['decision_score']:.4f}")
    
    return results


if __name__ == "__main__":
    from embedders import create_embedder
    
    embedder = create_embedder("sentence-all-MiniLM-L6-v2", target_dim=TARGET_EMBEDDING_DIM)
    results = run_decision_prediction_experiment(embedder, "all-MiniLM-L6-v2")
    print("\nResults:", json.dumps(results, indent=2))

