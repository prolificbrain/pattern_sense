"""Comprehensive Benchmark Report for UNIFIED Consciousness Engine.

This script runs benchmarks on multiple datasets, comparing the UNIFIED pattern recognition
system against traditional machine learning methods. It generates detailed reports and
visualizations to evaluate performance across different metrics.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UNIFIED components
from unified.mtu.learning import PatternMemory
from unified.mtu.accelerated import AcceleratedPatternMemory
from unified.mtu.hierarchical import HierarchicalPatternNetwork
from unified.mtu.anomaly import AnomalyScorer
from unified.mtu.clustering import PatternClusteringEngine

# Import dataset loaders
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from datasets.medical_datasets import load_dataset


class UnifiedBenchmark:
    """Comprehensive benchmark suite for UNIFIED Consciousness Engine."""
    
    def __init__(self, output_dir="./benchmark_results"):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize results dataframe
        self.results_df = pd.DataFrame()
        
        # Track memory usage
        self.memory_usage = {}
    
    def run_benchmarks(self, datasets=['breast_cancer', 'diabetes', 'ecg'], 
                      models=['SVM', 'RandomForest', 'MLP', 'PatternMemory', 
                              'AcceleratedPatternMemory', 'HierarchicalPatternNetwork']):
        """Run benchmarks on multiple datasets and models.
        
        Args:
            datasets: List of dataset names to benchmark
            models: List of model names to benchmark
        
        Returns:
            DataFrame of benchmark results
        """
        all_results = []
        
        for dataset_name in datasets:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking dataset: {dataset_name}")
            print(f"{'=' * 50}")
            
            # Load dataset
            X, y, metadata = load_dataset(dataset_name)
            
            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y)
            
            print(f"Dataset loaded: {len(X)} samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
            print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
            print(f"Dataset metadata: {metadata['name']} - {metadata['description']}")
            
            # Initialize dataset results
            dataset_results = {
                'dataset': dataset_name,
                'samples': len(X),
                'features': X.shape[1] if len(X.shape) > 1 else 1,
                'device': str(self.device),
                'models': {}
            }
            
            # Run benchmarks for each model
            for model_name in models:
                if hasattr(self, f"benchmark_{model_name.lower()}"):
                    print(f"\nBenchmarking {model_name}...")
                    benchmark_method = getattr(self, f"benchmark_{model_name.lower()}")
                    model_results = benchmark_method(X_train, X_test, y_train, y_test)
                    dataset_results['models'][model_name] = model_results
                else:
                    print(f"No benchmark method found for {model_name}")
            
            # Save dataset results
            all_results.append(dataset_results)
            
            # Generate dataset report
            self.generate_dataset_report(dataset_results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_results)
        
        return all_results
    
    def benchmark_svm(self, X_train, X_test, y_train, y_test):
        """Benchmark SVM classifier."""
        # Initialize model
        model = SVC(probability=True, random_state=42)
        
        # Train model
        start_time = time.time()
        model.fit(X_train.reshape(len(X_train), -1), y_train)  # Flatten for SVM
        train_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(X_test.reshape(len(X_test), -1))  # Flatten for SVM
        y_prob = model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
        eval_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['model_size'] = sys.getsizeof(model) / (1024 * 1024)  # Size in MB
        
        print(f"SVM Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def benchmark_randomforest(self, X_train, X_test, y_train, y_test):
        """Benchmark Random Forest classifier."""
        # Initialize model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train model
        start_time = time.time()
        model.fit(X_train.reshape(len(X_train), -1), y_train)  # Flatten for RF
        train_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(X_test.reshape(len(X_test), -1))  # Flatten for RF
        y_prob = model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
        eval_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['model_size'] = sys.getsizeof(model) / (1024 * 1024)  # Size in MB
        
        print(f"Random Forest Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def benchmark_mlp(self, X_train, X_test, y_train, y_test):
        """Benchmark Multi-layer Perceptron classifier."""
        # Initialize model
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        
        # Train model
        start_time = time.time()
        model.fit(X_train.reshape(len(X_train), -1), y_train)  # Flatten for MLP
        train_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        y_pred = model.predict(X_test.reshape(len(X_test), -1))  # Flatten for MLP
        y_prob = model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
        eval_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['model_size'] = sys.getsizeof(model) / (1024 * 1024)  # Size in MB
        
        print(f"MLP Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def benchmark_patternmemory(self, X_train, X_test, y_train, y_test):
        """Benchmark basic Pattern Memory."""
        # Initialize model
        pattern_memory = PatternMemory(max_patterns=10000)
        
        # Train model
        start_time = time.time()
        pattern_indices = {}
        for i, (x, y_val) in enumerate(zip(X_train, y_train)):
            idx = pattern_memory.learn_pattern(x)
            if y_val not in pattern_indices:
                pattern_indices[y_val] = []
            pattern_indices[y_val].append(idx)
        train_time = time.time() - start_time
        
        # Evaluate model
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
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing and memory information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['patterns_stored'] = len(pattern_memory._patterns)
        metrics['model_size'] = sys.getsizeof(pattern_memory) / (1024 * 1024)  # Size in MB
        
        print(f"Pattern Memory Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def benchmark_acceleratedpatternmemory(self, X_train, X_test, y_train, y_test):
        """Benchmark Accelerated Pattern Memory."""
        # Initialize model
        accel_memory = AcceleratedPatternMemory(
            max_patterns=10000,
            sparse_threshold=0.1,
            parallel_threshold=10,
            max_workers=4,
            use_gpu=False,  # Set to False to avoid tensor conversion issues
            batch_size=64
        )
        
        # Train model
        start_time = time.time()
        batch_size = 64
        accel_indices = {}
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # Learn patterns in batch
            batch_indices = accel_memory.learn_patterns_batch(batch_X)
            
            # Track indices by class
            for j, (idx, y_val) in enumerate(zip(batch_indices, batch_y)):
                if y_val not in accel_indices:
                    accel_indices[y_val] = []
                accel_indices[y_val].append(idx)
        
        train_time = time.time() - start_time
        
        # Evaluate model
        start_time = time.time()
        y_pred = []
        y_prob = []
        
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
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
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing and memory information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['patterns_stored'] = len(accel_memory._patterns)
        metrics['model_size'] = sys.getsizeof(accel_memory) / (1024 * 1024)  # Size in MB
        
        print(f"Accelerated Pattern Memory Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def benchmark_hierarchicalpatternnetwork(self, X_train, X_test, y_train, y_test):
        """Benchmark Hierarchical Pattern Network."""
        # Initialize model
        hierarchical_network = HierarchicalPatternNetwork(
            input_dimensions=X_train[0].shape,
            max_levels=2,
            patterns_per_level=5000,
            sparsity_increase=0.1
        )
        
        # Train model
        start_time = time.time()
        hier_indices = {0: {}, 1: {}}
        
        for i, (x, y_val) in enumerate(zip(X_train, y_train)):
            level_indices = hierarchical_network.learn_pattern(x)
            
            # Track indices by class and level
            for level, idx in level_indices.items():
                if y_val not in hier_indices[level]:
                    hier_indices[level][y_val] = []
                hier_indices[level][y_val].append(idx)
        
        train_time = time.time() - start_time
        
        # Evaluate model
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
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        
        # Add timing and memory information
        metrics['train_time'] = train_time
        metrics['eval_time'] = eval_time
        metrics['patterns_stored'] = sum(len(hierarchical_network.levels[level]._patterns) 
                                        for level in range(hierarchical_network.max_levels))
        metrics['model_size'] = sys.getsizeof(hierarchical_network) / (1024 * 1024)  # Size in MB
        
        print(f"Hierarchical Pattern Network Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}")
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        }
        return metrics
    
    def generate_dataset_report(self, dataset_results):
        """Generate report for a single dataset."""
        dataset_name = dataset_results['dataset']
        print(f"\nGenerating report for dataset: {dataset_name}")
        
        # Create dataframe for dataset results
        results_df = pd.DataFrame()
        for model_name, model_results in dataset_results['models'].items():
            model_df = pd.DataFrame(model_results, index=[model_name])
            results_df = pd.concat([results_df, model_df])
        
        # Save results to CSV
        results_df.to_csv(os.path.join(self.output_dir, f"{dataset_name}_results.csv"))
        
        # Generate metrics plot
        self.plot_metrics(results_df, dataset_name)
        
        # Generate time comparison plot
        self.plot_time_comparison(results_df, dataset_name)
        
        # Generate ROC curve plot
        # self.plot_roc_curves(dataset_results, dataset_name)
        
        print(f"Report for {dataset_name} saved to {self.output_dir}")
    
    def generate_comprehensive_report(self, all_results):
        """Generate comprehensive report across all datasets."""
        print("\nGenerating comprehensive benchmark report")
        
        # Combine all results into a single dataframe
        all_df = pd.DataFrame()
        for dataset_results in all_results:
            dataset_name = dataset_results['dataset']
            for model_name, model_results in dataset_results['models'].items():
                model_df = pd.DataFrame(model_results, index=[model_name])
                model_df['dataset'] = dataset_name
                all_df = pd.concat([all_df, model_df])
        
        # Save comprehensive results to CSV
        all_df.to_csv(os.path.join(self.output_dir, "comprehensive_results.csv"))
        
        # Generate comprehensive metrics plot
        self.plot_comprehensive_metrics(all_df)
        
        # Generate comprehensive time comparison plot
        self.plot_comprehensive_time(all_df)
        
        # Generate radar plot for model comparison
        self.plot_radar_comparison(all_df)
        
        print(f"Comprehensive report saved to {self.output_dir}")
    
    def plot_metrics(self, results_df, dataset_name):
        """Plot metrics comparison for a dataset."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        model_names = results_df.index.tolist()
        
        # Create dataframe for plotting
        metrics_data = {}
        for metric in metrics:
            metrics_data[metric] = [results_df.loc[model, metric] for model in model_names]
        
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
        plt.savefig(os.path.join(self.output_dir, f"{dataset_name}_metrics.png"))
        plt.close()
    
    def plot_time_comparison(self, results_df, dataset_name):
        """Plot time comparison for a dataset."""
        model_names = results_df.index.tolist()
        train_times = [results_df.loc[model, 'train_time'] for model in model_names]
        eval_times = [results_df.loc[model, 'eval_time'] for model in model_names]
        
        # Plot time comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.35
        index = np.arange(len(model_names))
        
        ax.bar(index - bar_width/2, train_times, bar_width, label='Training Time')
        ax.bar(index + bar_width/2, eval_times, bar_width, label='Evaluation Time')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Time Comparison - {dataset_name}')
        ax.set_xticks(index)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{dataset_name}_time.png"))
        plt.close()
    
    def plot_comprehensive_metrics(self, all_df):
        """Plot comprehensive metrics comparison across all datasets."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        datasets = all_df['dataset'].unique()
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4 * len(metrics)))
        
        for i, metric in enumerate(metrics):
            # Group by dataset and model, calculate mean
            metric_data = all_df.pivot_table(values=metric, index='dataset', columns=all_df.index)
            
            # Plot heatmap
            sns.heatmap(metric_data, annot=True, cmap='viridis', vmin=0, vmax=1, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Comparison Across Datasets')
            axes[i].set_ylabel('Dataset')
            axes[i].set_xlabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comprehensive_metrics.png"))
        plt.close()
    
    def plot_comprehensive_time(self, all_df):
        """Plot comprehensive time comparison across all datasets."""
        datasets = all_df['dataset'].unique()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Training time
        train_data = all_df.pivot_table(values='train_time', index='dataset', columns=all_df.index)
        sns.heatmap(train_data, annot=True, cmap='coolwarm', ax=axes[0])
        axes[0].set_title('Training Time Comparison Across Datasets (seconds)')
        axes[0].set_ylabel('Dataset')
        axes[0].set_xlabel('Model')
        
        # Evaluation time
        eval_data = all_df.pivot_table(values='eval_time', index='dataset', columns=all_df.index)
        sns.heatmap(eval_data, annot=True, cmap='coolwarm', ax=axes[1])
        axes[1].set_title('Evaluation Time Comparison Across Datasets (seconds)')
        axes[1].set_ylabel('Dataset')
        axes[1].set_xlabel('Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comprehensive_time.png"))
        plt.close()
    
    def plot_radar_comparison(self, all_df):
        """Plot radar chart for model comparison."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        models = all_df.index.unique()
        
        # Average metrics across datasets
        avg_metrics = all_df.groupby(all_df.index)[metrics].mean()
        
        # Create radar plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set the angles for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each model
        for model in models:
            values = avg_metrics.loc[model].tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Model Comparison Across All Metrics')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "radar_comparison.png"))
        plt.close()


def main():
    """Run the comprehensive benchmark suite."""
    # Create benchmark directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize benchmark suite
    benchmark = UnifiedBenchmark(output_dir=output_dir)
    
    # Run benchmarks
    results = benchmark.run_benchmarks(
        datasets=['breast_cancer', 'diabetes'],
        models=['SVM', 'RandomForest', 'PatternMemory', 'HierarchicalPatternNetwork']
    )
    
    print("\nComprehensive benchmarking complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
