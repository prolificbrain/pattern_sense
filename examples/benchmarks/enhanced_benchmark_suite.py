"""Enhanced benchmark suite for PatternSense pattern recognition system.

This script provides a comprehensive evaluation framework comparing
PatternSense against traditional machine learning methods across
multiple datasets and metrics.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import PatternSense components
from unified.mtu.learning import PatternMemory
from unified.mtu.accelerated import AcceleratedPatternMemory
from unified.mtu.hierarchical import HierarchicalPatternNetwork
from unified.mtu.temporal import TemporalPatternMemory
from unified.mtu.clustering import PatternClusteringEngine
from unified.mtu.anomaly import AnomalyScorer

# Constants
RESULTS_DIR = "./benchmark_results"
FIGURE_DPI = 300
CROSS_VAL_FOLDS = 5


class DatasetLoader:
    """Handles loading and preprocessing of various datasets for benchmarking."""
    
    @staticmethod
    def load_dataset(dataset_name, normalize=True):
        """Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            normalize: Whether to normalize features
            
        Returns:
            X: Features
            y: Labels
            metadata: Dictionary with dataset information
        """
        if dataset_name == "diabetes":
            return DatasetLoader._load_diabetes(normalize)
        elif dataset_name == "breast_cancer":
            return DatasetLoader._load_breast_cancer(normalize)
        elif dataset_name == "heart_disease":
            return DatasetLoader._load_heart_disease(normalize)
        elif dataset_name == "ecg":
            return DatasetLoader._load_ecg_data(normalize)
        elif dataset_name == "mnist_subset":
            return DatasetLoader._load_mnist_subset(normalize)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def _load_diabetes(normalize=True):
        """Load diabetes dataset."""
        from sklearn.datasets import load_diabetes as load_diabetes_sklearn
        
        # Load regression dataset and convert to classification
        data = load_diabetes_sklearn()
        X = data.data
        
        # Convert to binary classification (above/below median)
        median_target = np.median(data.target)
        y = (data.target > median_target).astype(int)
        
        if normalize:
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
        
        return X, y, metadata
    
    @staticmethod
    def _load_breast_cancer(normalize=True):
        """Load breast cancer dataset."""
        from sklearn.datasets import load_breast_cancer
        
        data = load_breast_cancer()
        X = data.data
        y = data.target
        
        if normalize:
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
        
        return X, y, metadata
    
    @staticmethod
    def _load_heart_disease(normalize=True):
        """Load heart disease dataset."""
        try:
            # Try to load from sklearn
            from sklearn.datasets import fetch_openml
            data = fetch_openml(name="heart-disease", version=1, as_frame=True)
            X = data.data.values
            y = (data.target.values == "present").astype(int)
        except:
            # Fallback to local file
            heart_data = pd.read_csv("./data/heart_disease/heart.csv")
            X = heart_data.drop("target", axis=1).values
            y = heart_data["target"].values
        
        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        metadata = {
            "name": "Heart Disease",
            "description": "Cleveland heart disease dataset",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "task": "binary_classification"
        }
        
        return X, y, metadata
    
    @staticmethod
    def _load_ecg_data(normalize=True):
        """Load ECG anomaly detection dataset."""
        # This is a placeholder - in a real implementation, you would load actual ECG data
        # For demonstration, we'll create synthetic ECG-like data
        np.random.seed(42)
        
        # Generate 200 samples of length 250 (simulating 10 seconds of ECG at 25Hz)
        n_samples = 200
        sequence_length = 250
        
        # Normal ECG patterns (with some variation)
        normal_patterns = []
        for _ in range(150):
            # Create a synthetic ECG-like pattern with P, QRS, T waves
            pattern = np.zeros(sequence_length)
            for i in range(10):  # 10 heartbeats
                center = i * 25 + np.random.randint(-3, 4)  # Some variation in timing
                # P wave
                pattern[max(0, center-8):min(sequence_length, center-3)] += np.random.normal(0.5, 0.1, size=min(5, min(sequence_length, center-3) - max(0, center-8)))
                # QRS complex
                pattern[max(0, center-2):min(sequence_length, center+3)] += np.random.normal(1.0, 0.2, size=min(5, min(sequence_length, center+3) - max(0, center-2)))
                # T wave
                pattern[max(0, center+4):min(sequence_length, center+10)] += np.random.normal(0.7, 0.15, size=min(6, min(sequence_length, center+10) - max(0, center+4)))
            
            # Add some noise
            pattern += np.random.normal(0, 0.1, size=sequence_length)
            normal_patterns.append(pattern)
        
        # Abnormal ECG patterns (with arrhythmias)
        abnormal_patterns = []
        for _ in range(50):
            # Create a synthetic ECG with abnormalities
            pattern = np.zeros(sequence_length)
            
            # Fewer heartbeats (bradycardia) or more (tachycardia)
            n_beats = np.random.choice([7, 13])  # Either too slow or too fast
            
            for i in range(n_beats):
                center = i * (sequence_length // n_beats) + np.random.randint(-5, 6)
                # P wave might be missing in some arrhythmias
                if np.random.random() > 0.3:  # 70% chance of P wave
                    pattern[max(0, center-8):min(sequence_length, center-3)] += np.random.normal(0.5, 0.1, size=min(5, min(sequence_length, center-3) - max(0, center-8)))
                
                # QRS complex - might be wider in some arrhythmias
                width = np.random.choice([5, 8])  # Normal or wide
                pattern[max(0, center-2):min(sequence_length, center+width-2)] += np.random.normal(1.0, 0.2, size=min(width, min(sequence_length, center+width-2) - max(0, center-2)))
                
                # T wave - might be inverted
                t_amp = np.random.choice([0.7, -0.5])  # Normal or inverted
                pattern[max(0, center+width-1):min(sequence_length, center+width+5)] += np.random.normal(t_amp, 0.15, size=min(6, min(sequence_length, center+width+5) - max(0, center+width-1)))
            
            # Add some noise
            pattern += np.random.normal(0, 0.15, size=sequence_length)  # Slightly more noise
            abnormal_patterns.append(pattern)
        
        # Combine normal and abnormal patterns
        X = np.vstack([np.array(normal_patterns), np.array(abnormal_patterns)])
        y = np.hstack([np.zeros(len(normal_patterns)), np.ones(len(abnormal_patterns))])
        
        # Reshape for sklearn compatibility
        X_reshaped = X.reshape(X.shape[0], -1)
        
        if normalize:
            # Normalize each sequence individually
            for i in range(X_reshaped.shape[0]):
                X_reshaped[i] = (X_reshaped[i] - np.mean(X_reshaped[i])) / (np.std(X_reshaped[i]) + 1e-10)
        
        metadata = {
            "name": "ECG Anomaly",
            "description": "Synthetic ECG data with normal and abnormal patterns",
            "n_samples": X.shape[0],
            "sequence_length": sequence_length,
            "task": "time_series_classification",
            "original_shape": X.shape
        }
        
        return X_reshaped, y, metadata
    
    @staticmethod
    def _load_mnist_subset(normalize=True):
        """Load a subset of MNIST for faster benchmarking."""
        try:
            # Try to load from sklearn
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1, parser='auto')
            X = mnist.data.values[:2000]  # Take only 2000 samples for speed
            y = (mnist.target.values[:2000].astype(int) >= 5).astype(int)  # Binary: 0-4 vs 5-9
        except:
            # Fallback to local file or synthetic data
            # Generate synthetic data that resembles MNIST structure
            np.random.seed(42)
            X = np.random.rand(2000, 784) * 255
            y = np.random.randint(0, 2, size=2000)
        
        if normalize:
            X = X / 255.0  # Scale to [0, 1]
        
        metadata = {
            "name": "MNIST Subset",
            "description": "Subset of MNIST digits converted to binary classification",
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "task": "image_classification",
            "image_shape": (28, 28)
        }
        
        return X, y, metadata


class PatternSenseClassifier:
    """Wrapper for PatternSense pattern memory to match scikit-learn classifier interface."""
    
    def __init__(self, max_patterns=1000, use_gpu=False, batch_size=32):
        """Initialize the classifier.
        
        Args:
            max_patterns: Maximum number of patterns to store
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for parallel processing
        """
        self.max_patterns = max_patterns
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        # Check for GPU availability
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if torch.backends.mps.is_available() else "cpu")
            if self.device.type == "cpu":
                print("Warning: GPU requested but not available. Using CPU instead.")
        else:
            self.device = torch.device("cpu")
        
        # Initialize pattern memory
        if self.use_gpu and self.device.type != "cpu":
            self.memory = AcceleratedPatternMemory(
                max_patterns=self.max_patterns,
                use_gpu=True,
                batch_size=self.batch_size
            )
        else:
            self.memory = PatternMemory(
                max_patterns=self.max_patterns,
                parallel_threshold=self.batch_size
            )
        
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
        if self.use_gpu and self.device.type != "cpu":
            self.memory = AcceleratedPatternMemory(
                max_patterns=self.max_patterns,
                use_gpu=True,
                batch_size=self.batch_size
            )
        else:
            self.memory = PatternMemory(
                max_patterns=self.max_patterns,
                parallel_threshold=self.batch_size
            )
        
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


class HierarchicalPatternClassifier:
    """Wrapper for PatternSense hierarchical pattern network to match scikit-learn classifier interface."""
    
    def __init__(self, max_levels=3, patterns_per_level=500, use_gpu=False):
        """Initialize the classifier.
        
        Args:
            max_levels: Maximum number of levels in the hierarchy
            patterns_per_level: Maximum number of patterns per level
            use_gpu: Whether to use GPU acceleration
        """
        self.max_levels = max_levels
        self.patterns_per_level = patterns_per_level
        self.use_gpu = use_gpu
        
        # Check for GPU availability
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if torch.backends.mps.is_available() else "cpu")
            if self.device.type == "cpu":
                print("Warning: GPU requested but not available. Using CPU instead.")
        else:
            self.device = torch.device("cpu")
        
        # Initialize hierarchical network
        self.network = HierarchicalPatternNetwork(
            input_dimensions=(1, -1),  # Will be set during fit
            max_levels=self.max_levels,
            patterns_per_level=self.patterns_per_level
        )
        
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
        # Reset network with proper input dimensions
        input_dim = X.shape[1]
        self.network = HierarchicalPatternNetwork(
            input_dimensions=(1, input_dim),
            max_levels=self.max_levels,
            patterns_per_level=self.patterns_per_level
        )
        
        self.class_indices = {0: [], 1: []}
        
        # Learn patterns for each class
        for i, (x, label) in enumerate(zip(X, y)):
            # Reshape to 2D if needed
            if len(x.shape) == 1:
                x_reshaped = x.reshape(1, -1)
            else:
                x_reshaped = x
                
            # Learn pattern hierarchically
            level_indices = self.network.learn_pattern(x_reshaped)
            
            # Store top-level pattern index for classification
            if level_indices and len(level_indices) > 0:
                top_level_idx = level_indices[-1][0]  # Get first pattern from top level
                self.class_indices[label].append((len(level_indices)-1, top_level_idx))
        
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
            # Reshape to 2D if needed
            if len(x.shape) == 1:
                x_reshaped = x.reshape(1, -1)
            else:
                x_reshaped = x
                
            # Recognize pattern hierarchically
            level_matches = self.network.recognize_pattern(x_reshaped)
            
            if not level_matches or not any(level_matches):
                # No matches found, predict majority class
                majority_class = max(self.class_indices.keys(), 
                                    key=lambda k: len(self.class_indices[k]))
                y_pred[i] = majority_class
                continue
            
            # Get top-level matches
            top_level = len(level_matches) - 1
            top_matches = level_matches[top_level] if top_level < len(level_matches) else []
            
            # Count class votes weighted by similarity
            class_scores = {0: 0.0, 1: 0.0}
            
            for pattern_idx, similarity in top_matches:
                # Find which class this pattern belongs to
                for class_label, indices in self.class_indices.items():
                    if (top_level, pattern_idx) in indices:
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
            # Reshape to 2D if needed
            if len(x.shape) == 1:
                x_reshaped = x.reshape(1, -1)
            else:
                x_reshaped = x
                
            # Recognize pattern hierarchically
            level_matches = self.network.recognize_pattern(x_reshaped)
            
            if not level_matches or not any(level_matches):
                # No matches found, predict majority class
                majority_class = max(self.class_indices.keys(), 
                                    key=lambda k: len(self.class_indices[k]))
                y_proba[i, majority_class] = 1.0
                continue
            
            # Get top-level matches
            top_level = len(level_matches) - 1
            top_matches = level_matches[top_level] if top_level < len(level_matches) else []
            
            # Count class votes weighted by similarity
            class_scores = {0: 0.0, 1: 0.0}
            
            for pattern_idx, similarity in top_matches:
                # Find which class this pattern belongs to
                for class_label, indices in self.class_indices.items():
                    if (top_level, pattern_idx) in indices:
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


class TemporalPatternClassifier:
    """Wrapper for PatternSense temporal pattern memory to match scikit-learn classifier interface.
    
    Particularly suited for time series and sequence data.
    """
    
    def __init__(self, max_patterns=1000, temporal_window=5):
        """Initialize the classifier.
        
        Args:
            max_patterns: Maximum number of patterns to store
            temporal_window: Size of temporal window for sequence analysis
        """
        self.max_patterns = max_patterns
        self.temporal_window = temporal_window
        
        # Initialize temporal pattern memory
        self.memory = TemporalPatternMemory(
            max_patterns=self.max_patterns,
            temporal_window=self.temporal_window
        )
        
        self.class_patterns = {0: [], 1: []}  # Store representative patterns for each class
        self.fitted = False
    
    def fit(self, X, y):
        """Train the classifier on the given data.
        
        Args:
            X: Training data features (for time series, each row is a full sequence)
            y: Training data labels
            
        Returns:
            self: The trained classifier
        """
        # Reset memory
        self.memory = TemporalPatternMemory(
            max_patterns=self.max_patterns,
            temporal_window=self.temporal_window
        )
        
        self.class_patterns = {0: [], 1: []}
        
        # Process each sequence
        for i, (x, label) in enumerate(zip(X, y)):
            # For time series data, we need to reshape it into a sequence
            # Assuming each row in X is a full sequence
            if len(x.shape) == 1:
                # Reshape 1D array into a sequence of temporal_window-length segments
                seq_length = len(x) // self.temporal_window
                if seq_length > 0:
                    x_seq = x[:seq_length * self.temporal_window].reshape(seq_length, self.temporal_window)
                else:
                    # If sequence is too short, pad it
                    x_seq = np.pad(x, (0, self.temporal_window - len(x) % self.temporal_window))
                    x_seq = x_seq.reshape(-1, self.temporal_window)
            else:
                # Already in sequence format
                x_seq = x
            
            # Learn the sequence
            pattern_indices = []
            for segment in x_seq:
                idx, _ = self.memory.observe_pattern(segment)
                pattern_indices.append(idx)
            
            # Store the sequence of pattern indices for this class
            self.class_patterns[label].append(pattern_indices)
        
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
            # Process the sequence as in fit
            if len(x.shape) == 1:
                seq_length = len(x) // self.temporal_window
                if seq_length > 0:
                    x_seq = x[:seq_length * self.temporal_window].reshape(seq_length, self.temporal_window)
                else:
                    x_seq = np.pad(x, (0, self.temporal_window - len(x) % self.temporal_window))
                    x_seq = x_seq.reshape(-1, self.temporal_window)
            else:
                x_seq = x
            
            # Recognize each segment and build a sequence of matches
            test_sequence = []
            for segment in x_seq:
                _, matches = self.memory.observe_pattern(segment, learn=False)
                test_sequence.append(matches)
            
            # Compare with stored class patterns
            class_scores = {0: 0.0, 1: 0.0}
            
            for class_label, class_sequences in self.class_patterns.items():
                for pattern_sequence in class_sequences:
                    # Calculate sequence similarity
                    similarity = self._calculate_sequence_similarity(test_sequence, pattern_sequence)
                    class_scores[class_label] += similarity
            
            # Normalize by number of sequences in each class
            for class_label in class_scores:
                if len(self.class_patterns[class_label]) > 0:
                    class_scores[class_label] /= len(self.class_patterns[class_label])
            
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
            # Process the sequence as in predict
            if len(x.shape) == 1:
                seq_length = len(x) // self.temporal_window
                if seq_length > 0:
                    x_seq = x[:seq_length * self.temporal_window].reshape(seq_length, self.temporal_window)
                else:
                    x_seq = np.pad(x, (0, self.temporal_window - len(x) % self.temporal_window))
                    x_seq = x_seq.reshape(-1, self.temporal_window)
            else:
                x_seq = x
            
            # Recognize each segment and build a sequence of matches
            test_sequence = []
            for segment in x_seq:
                _, matches = self.memory.observe_pattern(segment, learn=False)
                test_sequence.append(matches)
            
            # Compare with stored class patterns
            class_scores = {0: 0.0, 1: 0.0}
            
            for class_label, class_sequences in self.class_patterns.items():
                for pattern_sequence in class_sequences:
                    # Calculate sequence similarity
                    similarity = self._calculate_sequence_similarity(test_sequence, pattern_sequence)
                    class_scores[class_label] += similarity
            
            # Normalize by number of sequences in each class
            for class_label in class_scores:
                if len(self.class_patterns[class_label]) > 0:
                    class_scores[class_label] /= len(self.class_patterns[class_label])
            
            # Normalize scores to probabilities
            total_score = sum(class_scores.values())
            if total_score > 0:
                for class_label, score in class_scores.items():
                    y_proba[i, class_label] = score / total_score
            else:
                # If all scores are zero, predict uniform distribution
                y_proba[i, :] = 0.5
        
        return y_proba
    
    def _calculate_sequence_similarity(self, test_sequence, pattern_sequence):
        """Calculate similarity between two sequences of pattern matches.
        
        Args:
            test_sequence: Sequence of pattern matches from test data
            pattern_sequence: Sequence of pattern indices from training data
            
        Returns:
            similarity: Similarity score between sequences
        """
        # Ensure sequences are comparable
        min_length = min(len(test_sequence), len(pattern_sequence))
        if min_length == 0:
            return 0.0
        
        # Calculate overlap between sequences
        similarity_sum = 0.0
        for i in range(min_length):
            test_matches = test_sequence[i]
            pattern_idx = pattern_sequence[i]
            
            # Check if the pattern index is in the matches
            match_found = False
            match_similarity = 0.0
            
            for idx, sim in test_matches:
                if idx == pattern_idx:
                    match_found = True
                    match_similarity = sim
                    break
            
            if match_found:
                similarity_sum += match_similarity
        
        # Normalize by sequence length
        return similarity_sum / min_length

class BenchmarkRunner:
    """Runs comprehensive benchmarks comparing PatternSense with traditional ML methods."""
    
    def __init__(self, output_dir=RESULTS_DIR, use_gpu=True):
        """Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            use_gpu: Whether to use GPU acceleration for PatternSense models
        """
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device for GPU acceleration
        if self.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                      "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.traditional_models = {
            'SVM': SVC(probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        
        self.patternsense_models = {
            'PatternMemory': PatternSenseClassifier(max_patterns=1000, use_gpu=self.use_gpu),
            'HierarchicalPatternNetwork': HierarchicalPatternClassifier(max_levels=3, patterns_per_level=500, use_gpu=self.use_gpu),
            'TemporalPatternMemory': TemporalPatternClassifier(max_patterns=1000, temporal_window=5)
        }
    
    def run_benchmark(self, dataset_name, cross_validation=True, n_splits=CROSS_VAL_FOLDS):
        """Run comprehensive benchmark on a specific dataset.
        
        Args:
            dataset_name: Name of dataset to benchmark
            cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary of benchmark results
        """
        print(f"\n{'='*80}\nBenchmarking dataset: {dataset_name}\n{'='*80}")
        
        # Load dataset
        X, y, metadata = DatasetLoader.load_dataset(dataset_name)
        
        print(f"Dataset loaded: {metadata['name']}")
        print(f"  Samples: {metadata['n_samples']}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Task: {metadata['task']}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        # Initialize results dictionary
        results = {
            'dataset': metadata,
            'device': str(self.device),
            'models': {}
        }
        
        # Split dataset for non-cross-validation evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Benchmark traditional ML models
        print("\nBenchmarking traditional ML models...")
        for name, model in self.traditional_models.items():
            print(f"  Training {name}...")
            results['models'][name] = self._evaluate_model(
                model, X, y, X_train, X_test, y_train, y_test,
                cross_validation, n_splits
            )
        
        # Benchmark PatternSense models
        print("\nBenchmarking PatternSense models...")
        for name, model in self.patternsense_models.items():
            # Skip temporal model for non-time-series data if it's not suitable
            if name == 'TemporalPatternMemory' and metadata['task'] not in ['time_series_classification']:
                if X.shape[1] < 10:  # Skip if features are too few for temporal processing
                    print(f"  Skipping {name} (not suitable for this dataset)")
                    continue
            
            print(f"  Training {name}...")
            results['models'][name] = self._evaluate_model(
                model, X, y, X_train, X_test, y_train, y_test,
                cross_validation, n_splits
            )
        
        # Save results
        self._save_results(results, dataset_name)
        
        # Generate visualizations
        self._generate_visualizations(results, dataset_name)
        
        return results
    
    def _evaluate_model(self, model, X, y, X_train, X_test, y_train, y_test, cross_validation, n_splits):
        """Evaluate a model using both train/test split and optional cross-validation.
        
        Args:
            model: Model to evaluate
            X, y: Full dataset
            X_train, X_test, y_train, y_test: Train/test split
            cross_validation: Whether to use cross-validation
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary of evaluation results
        """
        model_results = {}
        
        # Train/test evaluation
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities if available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except (AttributeError, NotImplementedError):
            y_prob = y_pred  # Fall back to binary predictions
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5  # Default for failed AUC calculation
        
        # Store train/test results
        model_results['train_test'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'train_time': train_time,
            'inference_time': inference_time,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob
        }
        
        # Cross-validation if requested
        if cross_validation:
            cv_results = {}
            
            # Set up cross-validation
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Collect metrics across folds
            cv_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'auc': [],
                'train_time': [],
                'inference_time': []
            }
            
            # Perform cross-validation
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_fold_train, X_fold_test = X[train_idx], X[test_idx]
                y_fold_train, y_fold_test = y[train_idx], y[test_idx]
                
                # Train and evaluate
                start_time = time.time()
                model.fit(X_fold_train, y_fold_train)
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_fold_pred = model.predict(X_fold_test)
                inference_time = time.time() - start_time
                
                # Get probabilities if available
                try:
                    y_fold_prob = model.predict_proba(X_fold_test)[:, 1]
                except (AttributeError, NotImplementedError):
                    y_fold_prob = y_fold_pred
                
                # Calculate metrics
                cv_metrics['accuracy'].append(accuracy_score(y_fold_test, y_fold_pred))
                cv_metrics['precision'].append(precision_score(y_fold_test, y_fold_pred))
                cv_metrics['recall'].append(recall_score(y_fold_test, y_fold_pred))
                cv_metrics['f1_score'].append(f1_score(y_fold_test, y_fold_pred))
                
                try:
                    cv_metrics['auc'].append(roc_auc_score(y_fold_test, y_fold_prob))
                except ValueError:
                    cv_metrics['auc'].append(0.5)
                
                cv_metrics['train_time'].append(train_time)
                cv_metrics['inference_time'].append(inference_time)
            
            # Calculate mean and std for each metric
            for metric, values in cv_metrics.items():
                cv_results[f'{metric}_mean'] = np.mean(values)
                cv_results[f'{metric}_std'] = np.std(values)
            
            # Store cross-validation results
            model_results['cross_validation'] = cv_results
        
        return model_results
    
    def _save_results(self, results, dataset_name):
        """Save benchmark results to file.
        
        Args:
            results: Benchmark results dictionary
            dataset_name: Name of dataset
        """
        # Create results directory
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save full results as JSON
        import json
        with open(os.path.join(dataset_dir, 'full_results.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for model_name, model_results in results['models'].items():
            if 'train_test' in model_results:
                row = {
                    'Model': model_name,
                    'Accuracy': model_results['train_test']['accuracy'],
                    'Precision': model_results['train_test']['precision'],
                    'Recall': model_results['train_test']['recall'],
                    'F1 Score': model_results['train_test']['f1_score'],
                    'AUC': model_results['train_test']['auc'],
                    'Training Time (s)': model_results['train_test']['train_time'],
                    'Inference Time (s)': model_results['train_test']['inference_time']
                }
                summary_data.append(row)
        
        # Create summary DataFrame and save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(dataset_dir, 'summary_results.csv'), index=False)
        
        print(f"\nResults saved to {dataset_dir}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _generate_visualizations(self, results, dataset_name):
        """Generate visualizations from benchmark results.
        
        Args:
            results: Benchmark results dictionary
            dataset_name: Name of dataset
        """
        # Create visualizations directory
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        viz_dir = os.path.join(dataset_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Extract model names and metrics
        model_names = list(results['models'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        # Prepare data for plotting
        metric_values = {metric: [] for metric in metrics}
        for model_name in model_names:
            if 'train_test' in results['models'][model_name]:
                for metric in metrics:
                    metric_values[metric].append(results['models'][model_name]['train_test'][metric])
        
        # 1. Performance comparison bar chart
        self._plot_performance_comparison(model_names, metric_values, viz_dir, dataset_name)
        
        # 2. ROC curves
        self._plot_roc_curves(results, viz_dir, dataset_name)
        
        # 3. Confusion matrices
        self._plot_confusion_matrices(results, viz_dir, dataset_name)
        
        # 4. Training and inference time comparison
        self._plot_time_comparison(results, viz_dir, dataset_name)
        
        print(f"Visualizations saved to {viz_dir}")
    
    def _plot_performance_comparison(self, model_names, metric_values, viz_dir, dataset_name):
        """Plot performance comparison bar chart.
        
        Args:
            model_names: List of model names
            metric_values: Dictionary of metric values for each model
            viz_dir: Directory to save visualization
            dataset_name: Name of dataset
        """
        # Create figure
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
    
    def _plot_roc_curves(self, results, viz_dir, dataset_name):
        """Plot ROC curves for each model.
        
        Args:
            results: Benchmark results dictionary
            viz_dir: Directory to save visualization
            dataset_name: Name of dataset
        """
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each model
        for model_name, model_results in results['models'].items():
            if 'train_test' in model_results and isinstance(model_results['train_test']['y_prob'], list):
                y_true = np.array(model_results['train_test']['y_true'])
                y_prob = np.array(model_results['train_test']['y_prob'])
                
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = model_results['train_test']['auc']
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
                except Exception as e:
                    print(f"Could not plot ROC curve for {model_name}: {e}")
        
        # Plot random guessing line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Add labels and title
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves on {dataset_name} Dataset')
        plt.legend(loc="lower right")
        plt.grid(linestyle='--', alpha=0.7)
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, 'roc_curves.png'), dpi=FIGURE_DPI)
        plt.close()
    
    def _plot_confusion_matrices(self, results, viz_dir, dataset_name):
        """Plot confusion matrices for each model.
        
        Args:
            results: Benchmark results dictionary
            viz_dir: Directory to save visualization
            dataset_name: Name of dataset
        """
        # Create directory for confusion matrices
        cm_dir = os.path.join(viz_dir, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        
        # Plot confusion matrix for each model
        for model_name, model_results in results['models'].items():
            if 'train_test' in model_results and 'confusion_matrix' in model_results['train_test']:
                cm = np.array(model_results['train_test']['confusion_matrix'])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {model_name}')
                
                # Save figure
                plt.savefig(os.path.join(cm_dir, f'{model_name}_confusion_matrix.png'), dpi=FIGURE_DPI)
                plt.close()
    
    def _plot_time_comparison(self, results, viz_dir, dataset_name):
        """Plot training and inference time comparison.
        
        Args:
            results: Benchmark results dictionary
            viz_dir: Directory to save visualization
            dataset_name: Name of dataset
        """
        # Extract model names and time metrics
        model_names = []
        train_times = []
        inference_times = []
        
        for model_name, model_results in results['models'].items():
            if 'train_test' in model_results:
                model_names.append(model_name)
                train_times.append(model_results['train_test']['train_time'])
                inference_times.append(model_results['train_test']['inference_time'])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set width of bars
        bar_width = 0.35
        index = np.arange(len(model_names))
        
        # Plot bars
        plt.bar(index, train_times, bar_width, label='Training Time')
        plt.bar(index + bar_width, inference_times, bar_width, label='Inference Time')
        
        # Add labels and title
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
        
        # Also create a log-scale version for better visibility of differences
        plt.figure(figsize=(12, 6))
        plt.bar(index, train_times, bar_width, label='Training Time')
        plt.bar(index + bar_width, inference_times, bar_width, label='Inference Time')
        plt.xlabel('Model')
        plt.ylabel('Time (seconds, log scale)')
        plt.title(f'Training and Inference Time Comparison (Log Scale) on {dataset_name} Dataset')
        plt.xticks(index + bar_width / 2, model_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Save figure
        plt.savefig(os.path.join(viz_dir, 'time_comparison_log.png'), dpi=FIGURE_DPI)
        plt.close()


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced benchmarks for PatternSense")
    parser.add_argument("--datasets", nargs="+", default=["diabetes", "breast_cancer", "heart_disease", "ecg"],
                        help="Datasets to benchmark")
    parser.add_argument("--output", default=RESULTS_DIR, help="Output directory for results")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--no-cv", action="store_true", help="Disable cross-validation")
    parser.add_argument("--cv-folds", type=int, default=CROSS_VAL_FOLDS, help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(output_dir=args.output, use_gpu=not args.no_gpu)
    
    # Run benchmarks for each dataset
    all_results = {}
    for dataset in args.datasets:
        try:
            results = runner.run_benchmark(dataset, cross_validation=not args.no_cv, n_splits=args.cv_folds)
            all_results[dataset] = results
        except Exception as e:
            print(f"Error benchmarking {dataset}: {e}")
    
    print("\nAll benchmarks completed!")
