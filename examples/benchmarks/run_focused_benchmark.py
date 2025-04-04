"""Focused benchmark for UNIFIED Consciousness Engine pattern recognition.

This script runs a benchmark on a single dataset to validate the implementation.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UNIFIED components
from unified.mtu.learning import PatternMemory
from unified.trits.tryte import Tryte

# Import dataset loaders
from datasets.medical_datasets import load_dataset


def run_focused_benchmark(dataset_name='breast_cancer', output_dir="./benchmark_results"):
    """Run a focused benchmark on a single dataset.
    
    Args:
        dataset_name: Name of dataset to benchmark
        output_dir: Directory to save benchmark results
    
    Returns:
        Dictionary of benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    X, y, metadata = load_dataset(dataset_name)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Dataset loaded: {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    
    # Initialize results dictionary
    results = {
        'dataset': dataset_name,
        'samples': len(X),
        'device': str(device),
        'models': {}
    }
    
    # Benchmark SVM as baseline
    print("\nTraining SVM (baseline)...")
    svm = SVC(probability=True, random_state=42)
    
    start_time = time.time()
    svm.fit(X_train.reshape(len(X_train), -1), y_train)  # Flatten for SVM
    train_time = time.time() - start_time
    
    print("Evaluating SVM...")
    start_time = time.time()
    y_pred = svm.predict(X_test.reshape(len(X_test), -1))  # Flatten for SVM
    y_prob = svm.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
    eval_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results['models']['SVM'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'eval_time': eval_time
    }
    
    print(f"SVM Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Benchmark Basic Pattern Memory
    print("\nTraining Basic Pattern Memory...")
    pattern_memory = PatternMemory(max_patterns=10000)
    
    start_time = time.time()
    # Train pattern memory
    pattern_indices = {}
    for i, (x, y_val) in enumerate(zip(X_train, y_train)):
        idx = pattern_memory.learn_pattern(x)
        if y_val not in pattern_indices:
            pattern_indices[y_val] = []
        pattern_indices[y_val].append(idx)
    train_time = time.time() - start_time
    
    # Evaluate
    print("Evaluating Basic Pattern Memory...")
    start_time = time.time()
    y_pred = []
    y_prob = []
    
    for x in X_test:
        # Get most similar patterns
        matches = pattern_memory.recognize_pattern(x, top_k=5)
        if not matches:
            y_pred.append(0)  # Default to negative class
            y_prob.append(0.0)
            continue
        
        # Count matches by class
        class_scores = {0: 0.0, 1: 0.0}
        for idx, sim in matches:
            # Find which class this pattern belongs to
            for cls, indices in pattern_indices.items():
                if idx in indices:
                    class_scores[cls] += sim
        
        # Predict class with highest score
        if class_scores[1] > class_scores[0]:
            y_pred.append(1)
        else:
            y_pred.append(0)
        
        # Calculate probability
        total_score = class_scores[0] + class_scores[1]
        if total_score > 0:
            y_prob.append(class_scores[1] / total_score)
        else:
            y_prob.append(0.5)
    
    eval_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results['models']['PatternMemory'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'eval_time': eval_time,
        'patterns_stored': len(pattern_memory._patterns)
    }
    
    print(f"Pattern Memory Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame()
    for model_name, model_results in results['models'].items():
        model_df = pd.DataFrame(model_results, index=[model_name])
        results_df = pd.concat([results_df, model_df])
    
    results_df.to_csv(os.path.join(output_dir, f"{dataset_name}_focused_results.csv"))
    
    # Generate plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    model_names = list(results['models'].keys())
    
    # Create dataframe for plotting
    metrics_data = {}
    for metric in metrics:
        metrics_data[metric] = [results['models'][model][metric] for model in model_names]
    
    # Plot metrics
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    for i, metric in enumerate(metrics):
        ax.bar(index + i * bar_width, metrics_data[metric], bar_width, label=metric.upper())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Metrics Comparison - {dataset_name}')
    ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_focused_metrics.png"))
    plt.close()
    
    print(f"\nBenchmark complete. Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    # Run focused benchmark on breast cancer dataset
    run_focused_benchmark('breast_cancer')
