"""Comprehensive benchmark for UNIFIED Consciousness Engine pattern recognition.

This script performs a thorough evaluation of the pattern recognition system
against established methods using real-world medical datasets.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Import UNIFIED components
from unified.mtu.learning import PatternMemory
from unified.mtu.accelerated import AcceleratedPatternMemory
from unified.mtu.hierarchical import HierarchicalPatternNetwork
from unified.mtu.temporal import TemporalPatternMemory, TimeSeriesPredictor
from unified.mtu.clustering import PatternClusteringEngine
from unified.mtu.anomaly import AnomalyScorer

# Import dataset loaders
from datasets.medical_datasets import load_dataset


def run_benchmark(dataset_name, use_gpu=True, output_dir="./benchmark_results"):
    """Run comprehensive benchmark on a specific dataset.
    
    Args:
        dataset_name: Name of dataset to benchmark
        use_gpu: Whether to use GPU acceleration
        output_dir: Directory to save benchmark results
    
    Returns:
        Dictionary of benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device for GPU acceleration
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else 
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
    
    # Benchmark traditional ML models
    print("\nBenchmarking traditional ML models...")
    ml_models = {
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for name, model in ml_models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Evaluating {name}...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        eval_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results['models'][name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_time': train_time,
            'eval_time': eval_time
        }
        
        print(f"{name} Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Benchmark UNIFIED models
    print("\nBenchmarking UNIFIED models...")
    
    # 1. Basic Pattern Memory
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
    results['models']['BasicPatternMemory'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'eval_time': eval_time,
        'patterns_stored': len(pattern_memory._patterns)
    }
    
    print(f"Basic Pattern Memory Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # 2. GPU-Accelerated Pattern Memory
    if device.type != 'cpu':
        print("\nTraining GPU-Accelerated Pattern Memory...")
        accel_memory = AcceleratedPatternMemory(max_patterns=10000, use_gpu=True)
        
        start_time = time.time()
        # Convert data to tensors
        X_train_tensor = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_train]
        
        # Train in batches
        batch_size = 64
        accel_indices = {}
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Learn patterns in batch
            batch_indices = accel_memory.learn_patterns_batch(batch_X)
            
            # Track indices by class
            for j, (idx, y_val) in enumerate(zip(batch_indices, batch_y)):
                if y_val not in accel_indices:
                    accel_indices[y_val] = []
                accel_indices[y_val].append(idx)
        
        train_time = time.time() - start_time
        
        # Evaluate
        print("Evaluating GPU-Accelerated Pattern Memory...")
        start_time = time.time()
        
        # Convert test data to tensors
        X_test_tensor = [torch.tensor(x, dtype=torch.float32).to(device) for x in X_test]
        
        # Recognize in batches
        y_pred = []
        y_prob = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            batch_results = accel_memory.recognize_patterns_batch(batch_X, top_k=5)
            
            for matches in batch_results:
                if not matches:
                    y_pred.append(0)  # Default to negative class
                    y_prob.append(0.0)
                    continue
                
                # Count matches by class
                class_scores = {0: 0.0, 1: 0.0}
                for idx, sim in matches:
                    # Find which class this pattern belongs to
                    for cls, indices in accel_indices.items():
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
        results['models']['AcceleratedPatternMemory'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_time': train_time,
            'eval_time': eval_time,
            'patterns_stored': len(accel_memory._patterns)
        }
        
        print(f"Accelerated Pattern Memory Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # 3. Hierarchical Pattern Network
    print("\nTraining Hierarchical Pattern Network...")
    hierarchical_network = HierarchicalPatternNetwork(
        input_dimensions=X_train[0].shape,
        max_levels=2,
        patterns_per_level=5000
    )
    
    start_time = time.time()
    # Train hierarchical network
    hier_indices = {0: {}, 1: {}}
    for i, (x, y_val) in enumerate(zip(X_train, y_train)):
        level_indices = hierarchical_network.learn_pattern(x)
        
        # Track indices by class and level
        for level, idx in level_indices.items():
            if y_val not in hier_indices[level]:
                hier_indices[level][y_val] = []
            hier_indices[level][y_val].append(idx)
    
    train_time = time.time() - start_time
    
    # Evaluate
    print("Evaluating Hierarchical Pattern Network...")
    start_time = time.time()
    y_pred = []
    y_prob = []
    
    for x in X_test:
        # Recognize pattern hierarchically
        hierarchy_results = hierarchical_network.recognize_pattern(x)
        
        # Combine results from all levels
        class_scores = {0: 0.0, 1: 0.0}
        
        for level, matches in hierarchy_results.items():
            if not matches:
                continue
            
            # Weight higher levels more
            level_weight = level + 1
            
            for idx, sim in matches:
                # Find which class this pattern belongs to
                for cls, indices in hier_indices[level].items():
                    if idx in indices:
                        class_scores[cls] += sim * level_weight
        
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
    results['models']['HierarchicalNetwork'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'eval_time': eval_time,
        'patterns_stored': sum(len(level._patterns) for level in hierarchical_network.levels)
    }
    
    print(f"Hierarchical Network Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # 4. Anomaly Detection
    print("\nTraining Anomaly Detection System...")
    anomaly_detector = AnomalyScorer(use_gpu=(device.type != 'cpu'))
    
    # Separate data by class
    X_normal = [x for x, y_val in zip(X_train, y_train) if y_val == 0]
    X_abnormal = [x for x, y_val in zip(X_train, y_train) if y_val == 1]
    
    start_time = time.time()
    # Train anomaly detector
    anomaly_detector.train(X_normal, X_abnormal)
    train_time = time.time() - start_time
    
    # Evaluate
    print("Evaluating Anomaly Detection System...")
    start_time = time.time()
    y_pred = []
    y_prob = []
    
    for x in X_test:
        # Detect anomaly
        is_anomaly, score, _ = anomaly_detector.is_anomaly(x, threshold=0.5)
        
        # Convert to binary prediction (anomaly = positive class)
        y_pred.append(1 if is_anomaly else 0)
        y_prob.append(score)
    
    eval_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results['models']['AnomalyDetection'] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'train_time': train_time,
        'eval_time': eval_time
    }
    
    print(f"Anomaly Detection Results: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame()
    for model_name, model_results in results['models'].items():
        model_df = pd.DataFrame(model_results, index=[model_name])
        results_df = pd.concat([results_df, model_df])
    
    results_df.to_csv(os.path.join(output_dir, f"{dataset_name}_results.csv"))
    
    # Generate plots
    generate_benchmark_plots(results, output_dir, dataset_name)
    
    return results


def generate_benchmark_plots(results, output_dir, dataset_name):
    """Generate benchmark plots.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plots
        dataset_name: Name of dataset
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Performance metrics comparison
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
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_metrics.png"))
    plt.close()
    
    # 2. Training and evaluation time comparison
    times_data = {
        'train_time': [results['models'][model]['train_time'] for model in model_names],
        'eval_time': [results['models'][model]['eval_time'] for model in model_names]
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    ax.bar(index, times_data['train_time'], bar_width, label='Training Time')
    ax.bar(index + bar_width, times_data['eval_time'], bar_width, label='Evaluation Time')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'Training and Evaluation Time - {dataset_name}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_times.png"))
    plt.close()
    
    # 3. Memory usage comparison (if available)
    memory_data = []
    memory_labels = []
    
    for model in model_names:
        if 'patterns_stored' in results['models'][model]:
            memory_data.append(results['models'][model]['patterns_stored'])
            memory_labels.append(model)
    
    if memory_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(memory_labels, memory_data)
        ax.set_xlabel('Model')
        ax.set_ylabel('Patterns Stored')
        ax.set_title(f'Memory Usage - {dataset_name}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_memory.png"))
        plt.close()


if __name__ == "__main__":
    # List of datasets to benchmark
    datasets = [
        'ecg',
        'breast_cancer',
        'diabetes',
        'skin_lesion'
    ]
    
    # Run benchmarks
    all_results = {}
    for dataset in datasets:
        print(f"\n{'='*50}\nBenchmarking {dataset}\n{'='*50}")
        results = run_benchmark(dataset, use_gpu=True)
        all_results[dataset] = results
    
    print("\nBenchmark complete. Results saved to ./benchmark_results")
