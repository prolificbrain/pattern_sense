"""Advanced anomaly detection for UNIFIED Consciousness Engine.

This module implements sophisticated anomaly detection algorithms
that combine multiple methods for robust outlier identification.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

from .learning import PatternMemory
from .temporal import TemporalPatternMemory, TimeSeriesPredictor
from .clustering import PatternClusteringEngine


class AnomalyScorer:
    """Advanced anomaly scoring system.
    
    This class implements a multi-method approach to anomaly detection,
    combining reconstruction error, prediction error, cluster distance,
    and statistical methods for robust outlier identification.
    
    Attributes:
        methods: List of anomaly detection methods to use
        pattern_memory: Pattern memory for reconstruction-based detection
        temporal_memory: Temporal memory for sequence-based detection
        clustering_engine: Clustering engine for density-based detection
        isolation_forest: Isolation Forest for statistical detection
        lof: Local Outlier Factor for proximity-based detection
        thresholds: Dictionary of thresholds for each method
        weights: Dictionary of weights for each method
    """
    
    def __init__(self, methods: List[str] = None, 
                 pattern_memory: Optional[PatternMemory] = None,
                 temporal_memory: Optional[TemporalPatternMemory] = None,
                 clustering_engine: Optional[PatternClusteringEngine] = None,
                 use_gpu: bool = True):
        """Initialize anomaly scorer.
        
        Args:
            methods: List of anomaly detection methods to use
            pattern_memory: Pattern memory for reconstruction-based detection
            temporal_memory: Temporal memory for sequence-based detection
            clustering_engine: Clustering engine for density-based detection
            use_gpu: Whether to use GPU acceleration
        """
        # Default methods if none specified
        if methods is None:
            self.methods = ['reconstruction', 'prediction', 'clustering', 'statistical']
        else:
            self.methods = methods
        
        # Initialize components based on selected methods
        self.pattern_memory = pattern_memory
        self.temporal_memory = temporal_memory
        self.clustering_engine = clustering_engine
        
        # Initialize statistical models
        self.isolation_forest = None
        self.lof = None
        
        # Initialize device for GPU acceleration
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Default thresholds and weights for each method
        self.thresholds = {
            'reconstruction': 0.7,  # Similarity threshold
            'prediction': 0.3,      # Error threshold
            'clustering': 0.8,      # Distance threshold
            'statistical': -0.5     # Anomaly score threshold
        }
        
        self.weights = {
            'reconstruction': 0.3,
            'prediction': 0.3,
            'clustering': 0.2,
            'statistical': 0.2
        }
        
        # Calibration data
        self.calibration_scores = {}
        self.is_calibrated = False
    
    def train(self, normal_patterns: List[Union[np.ndarray, torch.Tensor]],
              abnormal_patterns: Optional[List[Union[np.ndarray, torch.Tensor]]] = None):
        """Train the anomaly detection system.
        
        Args:
            normal_patterns: List of normal patterns for training
            abnormal_patterns: Optional list of abnormal patterns for calibration
        """
        # Initialize components if not provided
        if 'reconstruction' in self.methods and self.pattern_memory is None:
            from .accelerated import AcceleratedPatternMemory
            self.pattern_memory = AcceleratedPatternMemory(max_patterns=10000, use_gpu=self.device.type != 'cpu')
        
        if 'prediction' in self.methods and self.temporal_memory is None:
            self.temporal_memory = TemporalPatternMemory(max_patterns=10000)
        
        if 'clustering' in self.methods and self.clustering_engine is None:
            self.clustering_engine = PatternClusteringEngine(pattern_memory=self.pattern_memory, 
                                                           use_gpu=self.device.type != 'cpu')
        
        # Train each component
        print("Training anomaly detection components...")
        
        # Convert patterns to numpy arrays
        normal_np = []
        for pattern in normal_patterns:
            if isinstance(pattern, torch.Tensor):
                normal_np.append(pattern.cpu().numpy())
            else:
                normal_np.append(pattern)
        
        # Train reconstruction-based detector
        if 'reconstruction' in self.methods and self.pattern_memory is not None:
            print("Training reconstruction-based detector...")
            for pattern in normal_patterns:
                self.pattern_memory.learn_pattern(pattern)
        
        # Train prediction-based detector
        if 'prediction' in self.methods and self.temporal_memory is not None:
            print("Training prediction-based detector...")
            for pattern in normal_patterns:
                self.temporal_memory.observe_pattern(pattern)
        
        # Train clustering-based detector
        if 'clustering' in self.methods and self.clustering_engine is not None:
            print("Training clustering-based detector...")
            self.clustering_engine.discover_patterns(normal_patterns)
        
        # Train statistical detectors
        if 'statistical' in self.methods:
            print("Training statistical detectors...")
            # Flatten patterns for statistical models
            flat_patterns = []
            for pattern in normal_np:
                flat_patterns.append(pattern.flatten())
            
            # Check if all patterns have the same shape
            shapes = [p.shape for p in flat_patterns]
            if len(set(shapes)) > 1:
                # Pad or truncate to make all patterns the same length
                max_length = max(len(p) for p in flat_patterns)
                normalized_patterns = []
                for p in flat_patterns:
                    if len(p) < max_length:
                        # Pad with zeros
                        padded = np.zeros(max_length)
                        padded[:len(p)] = p
                        normalized_patterns.append(padded)
                    else:
                        # Truncate
                        normalized_patterns.append(p[:max_length])
                X = np.vstack(normalized_patterns)
            else:
                X = np.vstack(flat_patterns)
            
            # Train Isolation Forest
            self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, 
                                                  random_state=42, n_jobs=-1)
            self.isolation_forest.fit(X)
            
            # Train Local Outlier Factor
            self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True, n_jobs=-1)
            self.lof.fit(X)
        
        # Calibrate if abnormal patterns are provided
        if abnormal_patterns is not None:
            self.calibrate(normal_patterns, abnormal_patterns)
    
    def calibrate(self, normal_patterns: List[Union[np.ndarray, torch.Tensor]],
                 abnormal_patterns: List[Union[np.ndarray, torch.Tensor]]):
        """Calibrate anomaly detection thresholds using labeled data.
        
        Args:
            normal_patterns: List of normal patterns
            abnormal_patterns: List of abnormal patterns
        """
        print("Calibrating anomaly detection thresholds...")
        
        # Calculate scores for normal patterns
        normal_scores = []
        for pattern in normal_patterns:
            scores = self.calculate_anomaly_scores(pattern)
            normal_scores.append(scores)
        
        # Calculate scores for abnormal patterns
        abnormal_scores = []
        for pattern in abnormal_patterns:
            scores = self.calculate_anomaly_scores(pattern)
            abnormal_scores.append(scores)
        
        # Calculate average scores for each method
        for method in self.methods:
            if method not in self.calibration_scores:
                self.calibration_scores[method] = {}
            
            # Calculate average scores for normal patterns
            normal_method_scores = [scores[method] for scores in normal_scores if method in scores]
            if normal_method_scores:
                self.calibration_scores[method]['normal_mean'] = np.mean(normal_method_scores)
                self.calibration_scores[method]['normal_std'] = np.std(normal_method_scores)
            
            # Calculate average scores for abnormal patterns
            abnormal_method_scores = [scores[method] for scores in abnormal_scores if method in scores]
            if abnormal_method_scores:
                self.calibration_scores[method]['abnormal_mean'] = np.mean(abnormal_method_scores)
                self.calibration_scores[method]['abnormal_std'] = np.std(abnormal_method_scores)
            
            # Calculate optimal threshold
            if normal_method_scores and abnormal_method_scores:
                # Find threshold that maximizes separation
                if method in ['reconstruction', 'clustering']:
                    # Higher values are normal, lower values are abnormal
                    threshold = self._find_optimal_threshold(normal_method_scores, abnormal_method_scores, reverse=True)
                else:
                    # Lower values are normal, higher values are abnormal
                    threshold = self._find_optimal_threshold(normal_method_scores, abnormal_method_scores)
                
                self.thresholds[method] = threshold
        
        # Calculate optimal weights based on separation power
        total_separation = 0
        for method in self.methods:
            if method in self.calibration_scores:
                if 'normal_mean' in self.calibration_scores[method] and 'abnormal_mean' in self.calibration_scores[method]:
                    normal_mean = self.calibration_scores[method]['normal_mean']
                    normal_std = self.calibration_scores[method]['normal_std']
                    abnormal_mean = self.calibration_scores[method]['abnormal_mean']
                    abnormal_std = self.calibration_scores[method]['abnormal_std']
                    
                    # Calculate separation power (higher is better)
                    if method in ['reconstruction', 'clustering']:
                        separation = (normal_mean - abnormal_mean) / (normal_std + abnormal_std + 1e-10)
                    else:
                        separation = (abnormal_mean - normal_mean) / (normal_std + abnormal_std + 1e-10)
                    
                    # Store separation power
                    self.calibration_scores[method]['separation'] = separation
                    total_separation += abs(separation)
        
        # Normalize weights
        if total_separation > 0:
            for method in self.methods:
                if method in self.calibration_scores and 'separation' in self.calibration_scores[method]:
                    self.weights[method] = abs(self.calibration_scores[method]['separation']) / total_separation
        
        self.is_calibrated = True
        print("Calibration complete.")
        print(f"Thresholds: {self.thresholds}")
        print(f"Weights: {self.weights}")
    
    def _find_optimal_threshold(self, normal_scores: List[float], abnormal_scores: List[float], 
                              reverse: bool = False) -> float:
        """Find optimal threshold that maximizes separation between normal and abnormal.
        
        Args:
            normal_scores: Scores for normal patterns
            abnormal_scores: Scores for abnormal patterns
            reverse: If True, higher values are normal (e.g., for similarity scores)
            
        Returns:
            Optimal threshold
        """
        # Combine scores and labels
        all_scores = np.concatenate([normal_scores, abnormal_scores])
        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(abnormal_scores))])
        
        # Sort scores and labels
        sorted_indices = np.argsort(all_scores)
        sorted_scores = all_scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        # Calculate metrics for each possible threshold
        best_f1 = 0
        best_threshold = 0
        
        for i in range(len(sorted_scores)):
            threshold = sorted_scores[i]
            
            if reverse:
                # Higher values are normal
                predictions = (all_scores < threshold).astype(int)
            else:
                # Lower values are normal
                predictions = (all_scores > threshold).astype(int)
            
            # Calculate F1 score
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def calculate_anomaly_scores(self, pattern: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Calculate anomaly scores using multiple methods.
        
        Args:
            pattern: Pattern to score
            
        Returns:
            Dictionary of method-specific anomaly scores
        """
        scores = {}
        
        # Reconstruction-based score
        if 'reconstruction' in self.methods and self.pattern_memory is not None:
            # Higher similarity means more normal (less anomalous)
            matches = self.pattern_memory.recognize_pattern(pattern, top_k=1)
            if matches:
                scores['reconstruction'] = matches[0][1]  # Similarity score
            else:
                scores['reconstruction'] = 0.0
        
        # Prediction-based score
        if 'prediction' in self.methods and self.temporal_memory is not None:
            # Observe pattern and get prediction error
            self.temporal_memory.observe_pattern(pattern)
            predictions = self.temporal_memory.predict_next(top_k=1)
            
            if predictions:
                # Get predicted pattern
                pred_idx = predictions[0][0]
                pred_pattern = self.temporal_memory.get_pattern(pred_idx)
                
                if pred_pattern is not None:
                    # Calculate prediction error (MSE)
                    if isinstance(pattern, torch.Tensor):
                        pattern_np = pattern.cpu().numpy()
                    else:
                        pattern_np = pattern
                    
                    # Ensure shapes match
                    if pattern_np.shape == pred_pattern.shape:
                        error = np.mean((pattern_np - pred_pattern) ** 2)
                        scores['prediction'] = error  # Higher error means more anomalous
                    else:
                        scores['prediction'] = 1.0  # Different shape is highly anomalous
                else:
                    scores['prediction'] = 1.0
            else:
                scores['prediction'] = 1.0
        
        # Clustering-based score
        if 'clustering' in self.methods and self.clustering_engine is not None:
            # Get cluster for pattern
            cluster = self.clustering_engine.get_cluster_for_pattern(pattern)
            
            # Calculate distance to cluster centroid
            if cluster in self.clustering_engine.cluster_centroids:
                centroid = self.clustering_engine.cluster_centroids[cluster]
                
                if isinstance(pattern, torch.Tensor):
                    pattern_np = pattern.cpu().numpy()
                else:
                    pattern_np = pattern
                
                # Flatten both for comparison
                flat_pattern = pattern_np.flatten()
                flat_centroid = centroid.flatten()
                
                # Ensure same length
                min_len = min(len(flat_pattern), len(flat_centroid))
                flat_pattern = flat_pattern[:min_len]
                flat_centroid = flat_centroid[:min_len]
                
                # Calculate cosine similarity
                norm_pattern = flat_pattern / (np.linalg.norm(flat_pattern) + 1e-10)
                norm_centroid = flat_centroid / (np.linalg.norm(flat_centroid) + 1e-10)
                similarity = np.dot(norm_pattern, norm_centroid)
                
                scores['clustering'] = similarity  # Higher similarity means more normal
            else:
                scores['clustering'] = 0.0  # No cluster found is highly anomalous
        
        # Statistical-based score
        if 'statistical' in self.methods and self.isolation_forest is not None and self.lof is not None:
            # Flatten pattern
            if isinstance(pattern, torch.Tensor):
                pattern_np = pattern.cpu().numpy()
            else:
                pattern_np = pattern
            
            flat_pattern = pattern_np.flatten()
            
            # Ensure correct shape for statistical models
            if flat_pattern.shape[0] != self.isolation_forest.n_features_in_:
                # Pad or truncate
                if flat_pattern.shape[0] < self.isolation_forest.n_features_in_:
                    padded = np.zeros(self.isolation_forest.n_features_in_)
                    padded[:flat_pattern.shape[0]] = flat_pattern
                    flat_pattern = padded
                else:
                    flat_pattern = flat_pattern[:self.isolation_forest.n_features_in_]
            
            # Calculate Isolation Forest score
            if_score = self.isolation_forest.score_samples([flat_pattern])[0]
            
            # Calculate LOF score
            lof_score = -self.lof.score_samples([flat_pattern])[0]  # Negative because higher means more anomalous
            
            # Combine scores (average)
            scores['statistical'] = (if_score + lof_score) / 2.0
        
        return scores
    
    def score_pattern(self, pattern: Union[np.ndarray, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Calculate combined anomaly score for a pattern.
        
        Args:
            pattern: Pattern to score
            
        Returns:
            Tuple of (combined_score, method_scores)
        """
        # Calculate individual scores
        scores = self.calculate_anomaly_scores(pattern)
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, score in scores.items():
            # Apply thresholds to normalize scores to 0-1 range
            if method in ['reconstruction', 'clustering']:
                # Higher values are normal, lower values are abnormal
                normalized_score = 1.0 - min(1.0, max(0.0, score / self.thresholds[method]))
            else:
                # Lower values are normal, higher values are abnormal
                normalized_score = min(1.0, max(0.0, score / self.thresholds[method]))
            
            # Apply weight
            weighted_sum += normalized_score * self.weights[method]
            total_weight += self.weights[method]
        
        # Calculate final score
        if total_weight > 0:
            combined_score = weighted_sum / total_weight
        else:
            combined_score = 0.5  # Default if no methods available
        
        return combined_score, scores
    
    def is_anomaly(self, pattern: Union[np.ndarray, torch.Tensor], threshold: float = 0.7) -> Tuple[bool, float, Dict[str, float]]:
        """Determine if a pattern is anomalous.
        
        Args:
            pattern: Pattern to evaluate
            threshold: Anomaly threshold (0-1 range)
            
        Returns:
            Tuple of (is_anomalous, anomaly_score, method_scores)
        """
        # Calculate anomaly score
        anomaly_score, method_scores = self.score_pattern(pattern)
        
        # Determine if anomalous
        is_anomalous = anomaly_score > threshold
        
        return is_anomalous, anomaly_score, method_scores
    
    def visualize_scores(self, normal_patterns: List[Union[np.ndarray, torch.Tensor]],
                        abnormal_patterns: List[Union[np.ndarray, torch.Tensor]],
                        output_file: str = 'anomaly_scores.png'):
        """Visualize anomaly scores for normal and abnormal patterns.
        
        Args:
            normal_patterns: List of normal patterns
            abnormal_patterns: List of abnormal patterns
            output_file: Path to save visualization
        """
        # Calculate scores
        normal_scores = [self.score_pattern(p)[0] for p in normal_patterns]
        abnormal_scores = [self.score_pattern(p)[0] for p in abnormal_patterns]
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        plt.hist(normal_scores, bins=20, alpha=0.5, label='Normal', color='green')
        plt.hist(abnormal_scores, bins=20, alpha=0.5, label='Abnormal', color='red')
        
        plt.axvline(x=0.7, color='black', linestyle='--', label='Threshold')
        
        plt.title('Anomaly Score Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Anomaly score visualization saved to {output_file}")
