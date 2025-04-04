"""Advanced Pattern Recognition Demo for UNIFIED Consciousness Engine.

This script demonstrates all the advanced features of the enhanced pattern recognition system:
1. GPU Acceleration for neural operations
2. Unsupervised clustering for pattern discovery
3. Advanced anomaly detection with multiple scoring methods
4. Integration with real-world medical datasets
"""

import os
import time
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UNIFIED components
from unified.mtu.accelerated import AcceleratedPatternMemory
from unified.mtu.clustering import PatternClusteringEngine
from unified.mtu.anomaly import AnomalyScorer
from unified.mtu.hierarchical import HierarchicalPatternNetwork

# Import dataset loader
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmarks'))
from datasets.medical_datasets import load_dataset


def run_advanced_demo(dataset_name='breast_cancer', use_gpu=True, output_dir="./demo_results"):
    """Run advanced pattern recognition demo.
    
    Args:
        dataset_name: Name of dataset to use
        use_gpu: Whether to use GPU acceleration
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device for GPU acceleration
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    X, y, metadata = load_dataset(dataset_name)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Dataset loaded: {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
    print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    print(f"Dataset metadata: {metadata['name']} - {metadata['description']}")
    
    # Keep data as numpy arrays for compatibility
    X_train_np = X_train
    X_test_np = X_test
    
    # 1. GPU-Accelerated Pattern Memory Demo
    print("\n" + "=" * 50)
    print("1. GPU-Accelerated Pattern Memory Demo")
    print("=" * 50)
    
    # Initialize accelerated pattern memory
    print("Initializing GPU-accelerated pattern memory...")
    accel_memory = AcceleratedPatternMemory(
        max_patterns=10000,
        sparse_threshold=0.1,
        parallel_threshold=10,
        max_workers=4,
        use_gpu=(device.type != 'cpu'),
        batch_size=64
    )
    
    # Train in batches
    print("Training accelerated pattern memory...")
    start_time = time.time()
    
    batch_size = 64
    accel_indices = {}
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train_np[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Learn patterns in batch
        batch_indices = accel_memory.learn_patterns_batch(batch_X)
        
        # Track indices by class
        for j, (idx, y_val) in enumerate(zip(batch_indices, batch_y)):
            if y_val not in accel_indices:
                accel_indices[y_val] = []
            accel_indices[y_val].append(idx)
    
    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.4f} seconds")
    print(f"Stored {len(accel_memory._patterns)} patterns")
    
    # Evaluate
    print("Evaluating accelerated pattern memory...")
    start_time = time.time()
    
    # Recognize in batches
    y_pred = []
    y_prob = []
    
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test_np[i:i+batch_size]
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
    
    print(f"Evaluation complete in {eval_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 2. Unsupervised Clustering Demo
    print("\n" + "=" * 50)
    print("2. Unsupervised Clustering Demo")
    print("=" * 50)
    
    # Initialize clustering engine
    print("Initializing pattern clustering engine...")
    clustering_engine = PatternClusteringEngine(
        pattern_memory=accel_memory,
        clustering_method='kmeans',
        n_clusters=3,  # Try to find natural clusters
        use_gpu=(device.type != 'cpu'),
        dimensionality_reduction='pca'
    )
    
    # Discover patterns
    print("Discovering patterns through unsupervised clustering...")
    start_time = time.time()
    clusters = clustering_engine.discover_patterns(X_train_np)
    cluster_time = time.time() - start_time
    
    print(f"Clustering complete in {cluster_time:.4f} seconds")
    print(f"Discovered {len(clusters)} clusters")
    
    # Print cluster statistics
    for cluster_id, pattern_indices in clusters.items():
        # Count class distribution in cluster
        class_counts = {0: 0, 1: 0}
        for idx in pattern_indices:
            if idx < len(y_train):
                class_counts[y_train[idx]] += 1
        
        total = sum(class_counts.values())
        if total > 0:
            majority_class = max(class_counts, key=class_counts.get)
            majority_pct = class_counts[majority_class] / total * 100
            print(f"Cluster {cluster_id}: {len(pattern_indices)} patterns, "
                  f"{majority_pct:.1f}% class {majority_class}")
    
    # Visualize clusters
    print("Generating cluster visualizations...")
    clustering_engine.visualize_clusters(os.path.join(output_dir, "pattern_clusters.png"))
    clustering_engine.visualize_cluster_representatives(os.path.join(output_dir, "cluster_representatives.png"))
    
    # 3. Advanced Anomaly Detection Demo
    print("\n" + "=" * 50)
    print("3. Advanced Anomaly Detection Demo")
    print("=" * 50)
    
    # Initialize anomaly detector
    print("Initializing anomaly detection system...")
    anomaly_detector = AnomalyScorer(
        methods=['reconstruction', 'clustering', 'statistical'],
        pattern_memory=accel_memory,
        clustering_engine=clustering_engine,
        use_gpu=(device.type != 'cpu')
    )
    
    # Separate data by class
    X_normal = [x for x, y_val in zip(X_train_np, y_train) if y_val == 0]
    X_abnormal = [x for x, y_val in zip(X_train_np, y_train) if y_val == 1]
    
    # Train anomaly detector
    print("Training anomaly detection system...")
    start_time = time.time()
    anomaly_detector.train(X_normal, X_abnormal)
    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.4f} seconds")
    
    # Evaluate anomaly detection
    print("Evaluating anomaly detection system...")
    start_time = time.time()
    y_pred = []
    y_prob = []
    method_scores = []
    
    for x in X_test_np:
        # Detect anomaly
        is_anomaly, score, scores = anomaly_detector.is_anomaly(x, threshold=0.5)
        
        # Convert to binary prediction (anomaly = positive class)
        y_pred.append(1 if is_anomaly else 0)
        y_prob.append(score)
        method_scores.append(scores)
    
    eval_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Evaluation complete in {eval_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Visualize anomaly scores
    print("Generating anomaly score visualization...")
    anomaly_detector.visualize_scores(X_normal, X_abnormal, 
                                    os.path.join(output_dir, "anomaly_scores.png"))
    
    # 4. Hierarchical Pattern Recognition Demo
    print("\n" + "=" * 50)
    print("4. Hierarchical Pattern Recognition Demo")
    print("=" * 50)
    
    # Initialize hierarchical network
    print("Initializing hierarchical pattern network...")
    hierarchical_network = HierarchicalPatternNetwork(
        input_dimensions=X_train[0].shape,
        max_levels=2,
        patterns_per_level=5000,
        sparsity_increase=0.1
    )
    
    # Train hierarchical network
    print("Training hierarchical pattern network...")
    start_time = time.time()
    hier_indices = {0: {}, 1: {}}
    
    # Process a subset for faster demo
    subset_size = min(200, len(X_train))
    for i in range(subset_size):
        x = X_train[i]
        y_val = y_train[i]
        
        level_indices = hierarchical_network.learn_pattern(x)
        
        # Track indices by class and level
        for level, idx in level_indices.items():
            if y_val not in hier_indices[level]:
                hier_indices[level][y_val] = []
            hier_indices[level][y_val].append(idx)
    
    train_time = time.time() - start_time
    print(f"Training complete in {train_time:.4f} seconds")
    
    # Count patterns at each level
    for level in range(hierarchical_network.max_levels):
        print(f"Level {level}: {len(hierarchical_network.levels[level]._patterns)} patterns")
    
    # Visualize hierarchy
    print("Visualizing hierarchical patterns...")
    if len(hierarchical_network.levels[0]._patterns) > 0:
        hierarchical_network.visualize_hierarchy(level=0, pattern_idx=0)
        print(f"Hierarchy visualization saved to 'hierarchy_level0_pattern0.png'")
    
    # Evaluate hierarchical network
    print("Evaluating hierarchical pattern network...")
    start_time = time.time()
    y_pred = []
    y_prob = []
    
    # Process a subset for faster demo
    subset_size = min(50, len(X_test))
    for i in range(subset_size):
        x = X_test[i]
        
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
    
    # Calculate metrics on subset
    if len(y_pred) > 0:
        accuracy = accuracy_score(y_test[:len(y_pred)], y_pred)
        precision = precision_score(y_test[:len(y_pred)], y_pred)
        recall = recall_score(y_test[:len(y_pred)], y_pred)
        f1 = f1_score(y_test[:len(y_pred)], y_pred)
        
        print(f"Evaluation complete in {eval_time:.4f} seconds (on subset of {len(y_pred)} samples)")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    print("\n" + "=" * 50)
    print("Advanced Pattern Recognition Demo Complete")
    print("Results saved to:", output_dir)
    print("=" * 50)


if __name__ == "__main__":
    # Run the advanced demo
    run_advanced_demo(dataset_name='breast_cancer', use_gpu=True)
