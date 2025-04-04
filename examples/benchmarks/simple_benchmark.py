"""Simple benchmark for PatternSense pattern recognition system.

This script provides a focused evaluation comparing PatternSense's
PatternMemory against traditional machine learning methods on
standard datasets.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes

# Import PatternSense components
from unified.mtu.learning import PatternMemory

# Constants
RESULTS_DIR = "./benchmark_results"
FIGURE_DPI = 300


class PatternSenseClassifier:
    """Wrapper for PatternSense pattern memory to match scikit-learn classifier interface."""
    
    def __init__(self, max_patterns=1000):
        """Initialize the classifier.
        
        Args:
            max_patterns: Maximum number of patterns to store
        """
        self.max_patterns = max_patterns
        self.memory = PatternMemory(max_patterns=self.max_patterns)
        self.class_indices = {0: [], 1: []}  # Map class labels to pattern indices
        self.fitted = False
    
    def fit(self, X, y):
        """Train the classifier on the given data.
        
        Args:
            X: Training data features
            y: Training data labels
            
        Returns:
            self: The trained classifier
        """
        # Reset memory and class indices
        self.memory = PatternMemory(max_patterns=self.max_patterns)
        self.class_indices = {0: [], 1: []}
        
        # Learn patterns for each class
        for i, (x, label) in enumerate(zip(X, y)):
            pattern_idx = self.memory.learn_pattern(x)
            self.class_indices[label].append(pattern_idx)
        
        self.fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels for the given data.
        
        Args:
            X: Test data features
            
        Returns:
            y_pred: Predicted class labels
        """
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        y_pred = np.zeros(len(X), dtype=int)
        
        for i, x in enumerate(X):
            # Get top matches
            matches = self.memory.recognize_pattern(x, top_k=5)
            
            if not matches:
                # No matches found, predict majority class
                majority_class = max(self.class_indices.keys(), 
                                    key=lambda k: len(self.class_indices[k]))
                y_pred[i] = majority_class
                continue
            
            # Count class votes weighted by similarity
            class_scores = {0: 0.0, 1: 0.0}
            for pattern_idx, similarity in matches:
                # Find which class this pattern belongs to
                for class_label, indices in self.class_indices.items():
                    if pattern_idx in indices:
                        class_scores[class_label] += similarity
                        break
            
            # Predict the class with the highest score
            y_pred[i] = max(class_scores.keys(), key=lambda k: class_scores[k])
        
        return y_pred
    
    def predict_proba(self, X):
        """Predict class probabilities for the given data.
        
        Args:
            X: Test data features
            
        Returns:
            y_proba: Predicted class probabilities
        """
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        y_proba = np.zeros((len(X), 2))
        
        for i, x in enumerate(X):
            # Get top matches
            matches = self.memory.recognize_pattern(x, top_k=5)
            
            if not matches:
                # No matches found, predict majority class
                majority_class = max(self.class_indices.keys(), 
                                    key=lambda k: len(self.class_indices[k]))
                y_proba[i, majority_class] = 1.0
                continue
            
            # Count class votes weighted by similarity
            class_scores = {0: 0.0, 1: 0.0}
            for pattern_idx, similarity in matches:
                # Find which class this pattern belongs to
                for class_label, indices in self.class_indices.items():
                    if pattern_idx in indices:
                        class_scores[class_label] += similarity
                        break
            
            # Normalize scores to probabilities
            total_score = sum(class_scores.values())
            if total_score > 0:
                for class_label, score in class_scores.items():
                    y_proba[i, class_label] = score / total_score
            else:
                # If all scores are zero, predict uniform distribution
                y_proba[i, :] = 0.5
        
        return y_proba


def load_dataset(dataset_name):
    """Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        X: Features
        y: Labels
        metadata: Dictionary with dataset information
    """
    if dataset_name == "diabetes":
        # Load diabetes dataset
        data = load_diabetes()
        X = data.data
        
        # Convert to binary classification (above/below median)
        median_target = np.median(data.target)
        y = (data.target > median_target).astype(int)
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        metadata = {
            "name": "Diabetes",
            "description": "Diabetes regression dataset converted to classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": data.feature_names,
            "task": "binary_classification"
        }
        
    elif dataset_name == "breast_cancer":
        # Load breast cancer dataset
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        metadata = {
            "name": "Breast Cancer",
            "description": "Breast cancer diagnostic dataset",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": data.feature_names,
            "task": "binary_classification"
        }
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y, metadata


def run_benchmark(dataset_name):
    """Run benchmark on a specific dataset.
    
    Args:
        dataset_name: Name of dataset to benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    print(f"\n{'='*80}\nBenchmarking dataset: {dataset_name}\n{'='*80}")
    
    # Create output directory
    output_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y, metadata = load_dataset(dataset_name)
    
    print(f"Dataset loaded: {metadata['name']}")
    print(f"  Samples: {metadata['n_samples']}")
    print(f"  Features: {metadata['n_features']}")
    print(f"  Task: {metadata['task']}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'SVM': SVC(probability=True, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'PatternMemory': PatternSenseClassifier(max_patterns=1000)
    }
    
    # Initialize results dictionary
    results = {
        'dataset': metadata,
        'models': {}
    }
    
    # Benchmark each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except (AttributeError, NotImplementedError):
            y_prob = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5
        
        # Store results
        results['models'][name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_time': train_time,
            'inference_time': inference_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Print results
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Training time: {train_time:.4f} seconds")
        print(f"  Inference time: {inference_time:.4f} seconds")
    
    # Generate visualizations
    generate_visualizations(results, output_dir, dataset_name)
    
    # Save results as CSV
    save_results_csv(results, output_dir)
    
    return results


def generate_visualizations(results, output_dir, dataset_name):
    """Generate visualizations from benchmark results.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save visualizations
        dataset_name: Name of dataset
    """
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract model names and metrics
    model_names = list(results['models'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Prepare data for plotting
    metric_values = {metric: [] for metric in metrics}
    for model_name in model_names:
        for metric in metrics:
            metric_values[metric].append(results['models'][model_name][metric])
    
    # 1. Performance comparison bar chart
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(model_names))
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot bars for each metric
    for i, (metric, values) in enumerate(metric_values.items()):
        plt.bar(index + i * bar_width, values, bar_width, label=metric.replace('_', ' ').title())
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(f'Performance Comparison on {dataset_name} Dataset')
    plt.xticks(index + bar_width * 2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=FIGURE_DPI)
    plt.close()
    
    # 2. Training and inference time comparison
    train_times = [results['models'][model]['train_time'] for model in model_names]
    inference_times = [results['models'][model]['inference_time'] for model in model_names]
    
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.bar(index, train_times, bar_width, label='Training Time')
    plt.bar(index + bar_width, inference_times, bar_width, label='Inference Time')
    
    plt.xlabel('Model')
    plt.ylabel('Time (seconds)')
    plt.title(f'Training and Inference Time Comparison on {dataset_name} Dataset')
    plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(os.path.join(viz_dir, 'time_comparison.png'), dpi=FIGURE_DPI)
    plt.close()
    
    # 3. Confusion matrices
    cm_dir = os.path.join(viz_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    for model_name in model_names:
        cm = np.array(results['models'][model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Save figure
        plt.savefig(os.path.join(cm_dir, f'{model_name}_confusion_matrix.png'), dpi=FIGURE_DPI)
        plt.close()


def save_results_csv(results, output_dir):
    """Save benchmark results as CSV.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save results
    """
    # Prepare data for CSV
    data = []
    for model_name, model_results in results['models'].items():
        row = {'Model': model_name}
        row.update({k: v for k, v in model_results.items() if k != 'confusion_matrix'})
        data.append(row)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run benchmarks
    datasets = ["diabetes", "breast_cancer"]
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_benchmark(dataset)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error benchmarking {dataset}: {e}")
    
    print("\nAll benchmarks completed!")
