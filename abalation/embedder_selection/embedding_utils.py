"""
Utility functions for embedding experiments.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.get_dataset import get_dataset_loader
from typing import List, Tuple, Dict


def load_questions_from_datasets(datasets: List[str], max_samples: int = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Load questions from multiple datasets.
    
    Returns:
        questions: List of question strings
        answers: List of answer strings
        labels: List of dataset names (for classification)
    """
    all_questions = []
    all_answers = []
    all_labels = []
    
    for dataset_name in datasets:
        try:
            loader = get_dataset_loader(dataset_name, is_eval=False)
            
            # Get dataset (handle both wrapped and direct datasets)
            if hasattr(loader, 'data'):
                dataset = loader.data
            else:
                dataset = loader
            
            count = 0
            for item in dataset:
                if max_samples and count >= max_samples:
                    break
                
                # Handle MedQA's nested structure
                if dataset_name == "medqa" and 'data' in item:
                    # MedQA structure: {'id': ..., 'data': {'Question': ..., 'Options': ..., 'Correct Answer': ...}, ...}
                    data = item['data']
                    question = data.get('Question', '')
                    options = data.get('Options', {})
                    # Format question with options
                    if options:
                        options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
                        question = f"{question}\n\nOptions:\n{options_text}"
                    answer = data.get('Correct Answer', '')
                    q, a = question, answer
                else:
                    # Standard structure - extract question
                    q = item.get('question') or item.get('Question') or item.get('input') or str(item)
                    
                    # Extract answer
                    a = item.get('answer') or item.get('Final answer') or item.get('output') or item.get('ground_truth') or ""
                
                if q and a:  # Only add if both exist
                    all_questions.append(str(q))
                    all_answers.append(str(a))
                    all_labels.append(dataset_name)
                    count += 1
            
            print(f"  Loaded {count} samples from {dataset_name}")
        except Exception as e:
            print(f"  Warning: Could not load {dataset_name}: {e}")
    
    print(f"Total samples loaded: {len(all_questions)}")
    return all_questions, all_answers, all_labels


def get_complexity_labels(questions: List[str]) -> np.ndarray:
    """
    Heuristic complexity labels based on question characteristics.
    Returns array of complexity scores (0-1, higher = more complex).
    """
    complexity_scores = []
    
    for q in questions:
        score = 0.0
        
        # Length heuristic
        word_count = len(q.split())
        score += min(word_count / 200.0, 0.4)  # Max 0.4 from length
        
        # Multi-step indicators
        if any(keyword in q.lower() for keyword in ['first', 'then', 'next', 'step', 'calculate', 'solve']):
            score += 0.2
        
        # Mathematical complexity
        if any(op in q for op in ['+', '-', '*', '/', '=', 'percent', '%']):
            score += 0.2
        
        # Question words (multiple questions = complex)
        question_words = ['what', 'when', 'where', 'why', 'how', 'which', 'who']
        q_lower = q.lower()
        score += min(sum(1 for word in question_words if word in q_lower) * 0.1, 0.2)
        
        # Cap at 1.0
        complexity_scores.append(min(score, 1.0))
    
    return np.array(complexity_scores)


def get_tool_labels(questions: List[str], answers: List[str]) -> List[str]:
    """
    Heuristic tool requirement labels.
    Returns: ['calculator', 'web_search', 'python', 'none']
    """
    tool_labels = []
    
    for q, a in zip(questions, answers):
        q_lower = q.lower()
        a_lower = str(a).lower()
        
        # Check for calculator needs
        if any(keyword in q_lower for keyword in ['calculate', 'compute', 'sum', 'total', '+', '-', '*', '/', 'percent']):
            tool_labels.append('calculator')
        # Check for web search needs
        elif any(keyword in q_lower for keyword in ['who', 'when', 'where', 'current', 'recent', 'latest']):
            tool_labels.append('web_search')
        # Check for python needs (data processing, file reading)
        elif any(keyword in q_lower for keyword in ['file', 'csv', 'data', 'dataset', 'read', 'filter']) or 'file_path' in q:
            tool_labels.append('python')
        else:
            tool_labels.append('none')
    
    return tool_labels


def get_workflow_labels(questions: List[str], answers: List[str], tool_labels: List[str]) -> List[int]:
    """
    Heuristic workflow labels based on question characteristics.
    Returns: workflow indices (0-8)
    
    Workflows:
    0=Direct, 1=Reason+Ans, 2=Reason+Verify+Ans,
    3=Routing, 4=Parallel-Sectioning, 5=Parallel-Voting,
    6=Orchestrator-Workers, 7=Evaluator-Optimizer, 8=Autonomous-Agent
    """
    workflow_labels = []
    
    for q, a, tool in zip(questions, answers, tool_labels):
        q_lower = q.lower()
        word_count = len(q.split())
        
        # Very simple questions -> Direct (0)
        if word_count < 20 and tool == 'none':
            workflow_labels.append(0)
        # Simple reasoning -> Reason+Ans (1)
        elif word_count < 50 and tool != 'python':
            workflow_labels.append(1)
        # Complex reasoning with verification -> Reason+Verify+Ans (2)
        elif word_count > 100 or tool == 'python':
            workflow_labels.append(2)
        # Multi-part questions -> Routing (3)
        elif ' or ' in q_lower or ' and ' in q_lower:
            workflow_labels.append(3)
        # Default to simple workflow
        else:
            workflow_labels.append(1)
    
    return workflow_labels


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def batch_cosine_similarity(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and batch of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query)
    return np.dot(embeddings, query) / (norms * query_norm + 1e-8)

