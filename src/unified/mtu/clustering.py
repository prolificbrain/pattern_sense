"""Unsupervised clustering for pattern discovery in UNIFIED Consciousness Engine.

This module implements unsupervised clustering algorithms for automatic
pattern discovery and organization without labeled training data.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
import concurrent.futures
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .learning import PatternMemory
from .accelerated import AcceleratedPatternMemory
from unified.trits.tryte import Tryte


class PatternClusteringEngine:
    """Unsupervised clustering engine for pattern discovery.
    
    This class implements various clustering algorithms to automatically
    discover and organize patterns without labeled training data.
    
    Attributes:
        pattern_memory: Pattern memory for storing discovered patterns
        clustering_method: Method used for clustering
        n_clusters: Number of clusters for k-means
        min_samples: Minimum samples for density-based clustering
        use_gpu: Whether to use GPU acceleration
        dimensionality_reduction: Method for dimensionality reduction
    """
    
    def __init__(self, pattern_memory: Optional[PatternMemory] = None,
                 clustering_method: str = 'kmeans', n_clusters: int = 10,
                 min_samples: int = 5, use_gpu: bool = True,
                 dimensionality_reduction: str = 'pca'):
        """Initialize pattern clustering engine.
        
        Args:
            pattern_memory: Pattern memory for storing discovered patterns
            clustering_method: Clustering method ('kmeans' or 'dbscan')
            n_clusters: Number of clusters for k-means
            min_samples: Minimum samples for density-based clustering
            use_gpu: Whether to use GPU acceleration
            dimensionality_reduction: Method for dimensionality reduction ('pca' or 'tsne')
        """
        # Initialize pattern memory
        if pattern_memory is None:
            if use_gpu:
                self.pattern_memory = AcceleratedPatternMemory(max_patterns=10000, use_gpu=use_gpu)
            else:
                self.pattern_memory = PatternMemory(max_patterns=10000)
        else:
            self.pattern_memory = pattern_memory
        
        # Clustering parameters
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.use_gpu = use_gpu
        self.dimensionality_reduction = dimensionality_reduction
        
        # Initialize clustering model
        if clustering_method == 'kmeans':
            self.model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=42)
        elif clustering_method == 'dbscan':
            self.model = DBSCAN(min_samples=min_samples, eps=0.5, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        # Initialize device for GPU acceleration
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Store cluster assignments and centroids
        self.cluster_assignments = {}
        self.cluster_centroids = {}
        self.reduced_data = None
    
    def discover_patterns(self, patterns: List[Union[Tryte, np.ndarray, torch.Tensor]]) -> Dict[int, List[int]]:
        """Discover patterns using unsupervised clustering.
        
        Args:
            patterns: List of patterns to cluster
            
        Returns:
            Dictionary mapping cluster IDs to pattern indices
        """
        # Convert patterns to consistent format
        processed_patterns = []
        for pattern in patterns:
            if isinstance(pattern, torch.Tensor):
                if pattern.is_cuda or pattern.device.type == 'mps':
                    processed_patterns.append(pattern.cpu().numpy())
                else:
                    processed_patterns.append(pattern.numpy())
            elif isinstance(pattern, Tryte):
                processed_patterns.append(self.pattern_memory._tryte_to_state_field(pattern))
            elif isinstance(pattern, np.ndarray):
                processed_patterns.append(pattern)
            else:
                raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
        
        # Flatten patterns for clustering
        flattened_patterns = [p.flatten() for p in processed_patterns]
        
        # Check if all patterns have the same shape
        shapes = [p.shape for p in flattened_patterns]
        if len(set(shapes)) > 1:
            # Pad or truncate to make all patterns the same length
            max_length = max(len(p) for p in flattened_patterns)
            normalized_patterns = []
            for p in flattened_patterns:
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
            X = np.vstack(flattened_patterns)
        
        # Apply dimensionality reduction if needed
        if X.shape[1] > 100:  # Only reduce if dimension is high
            if self.dimensionality_reduction == 'pca':
                pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
                X_reduced = pca.fit_transform(X)
            elif self.dimensionality_reduction == 'tsne':
                tsne = TSNE(n_components=2, random_state=42)
                X_reduced = tsne.fit_transform(X)
            else:
                X_reduced = X
            
            # Store reduced data for visualization
            self.reduced_data = X_reduced
        else:
            X_reduced = X
            self.reduced_data = X
        
        # Perform clustering
        cluster_labels = self.model.fit_predict(X_reduced)
        
        # Store cluster assignments
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Store assignments and learn representative patterns
        self.cluster_assignments = clusters
        
        # Calculate cluster centroids
        if self.clustering_method == 'kmeans':
            # For k-means, centroids are already calculated
            centroids = self.model.cluster_centers_
            for i, centroid in enumerate(centroids):
                # Reshape centroid to original pattern shape if possible
                if len(processed_patterns) > 0:
                    original_shape = processed_patterns[0].shape
                    if len(centroid) >= np.prod(original_shape):
                        reshaped_centroid = centroid[:np.prod(original_shape)].reshape(original_shape)
                        self.cluster_centroids[i] = reshaped_centroid
                    else:
                        self.cluster_centroids[i] = centroid
                else:
                    self.cluster_centroids[i] = centroid
        else:
            # For DBSCAN, calculate mean of patterns in each cluster
            for label, indices in clusters.items():
                if label == -1:  # Noise points in DBSCAN
                    continue
                    
                # Calculate mean pattern
                cluster_patterns = [processed_patterns[i] for i in indices]
                if cluster_patterns:
                    mean_pattern = np.mean(cluster_patterns, axis=0)
                    self.cluster_centroids[label] = mean_pattern
        
        # Learn representative patterns
        for label, centroid in self.cluster_centroids.items():
            if label == -1:  # Skip noise cluster
                continue
                
            # Learn centroid as representative pattern
            self.pattern_memory.learn_pattern(centroid)
        
        return clusters
    
    def visualize_clusters(self, output_file: str = 'pattern_clusters.png'):
        """Visualize discovered pattern clusters.
        
        Args:
            output_file: Path to save visualization
        """
        if self.reduced_data is None or not self.cluster_assignments:
            print("No clusters to visualize. Run discover_patterns first.")
            return
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Plot points colored by cluster
        for label, indices in self.cluster_assignments.items():
            if label == -1:  # Noise points in DBSCAN
                color = 'black'
                marker = 'x'
            else:
                color = None  # Use default color cycle
                marker = 'o'
                
            cluster_points = self.reduced_data[indices]
            if cluster_points.shape[1] >= 2:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                           label=f'Cluster {label}', marker=marker, color=color)
        
        # Plot centroids for k-means
        if self.clustering_method == 'kmeans' and hasattr(self.model, 'cluster_centers_'):
            centroids = self.model.cluster_centers_
            if centroids.shape[1] >= 2:
                plt.scatter(centroids[:, 0], centroids[:, 1], s=200, marker='*', 
                           c='red', label='Centroids', edgecolors='black')
        
        plt.title(f'Pattern Clusters ({self.clustering_method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Cluster visualization saved to {output_file}")
    
    def visualize_cluster_representatives(self, output_file: str = 'cluster_representatives.png'):
        """Visualize representative patterns for each cluster.
        
        Args:
            output_file: Path to save visualization
        """
        if not self.cluster_centroids:
            print("No cluster representatives to visualize. Run discover_patterns first.")
            return
        
        # Determine grid size
        n_clusters = len(self.cluster_centroids)
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        
        # Create plot
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot each representative pattern
        for i, (label, centroid) in enumerate(sorted(self.cluster_centroids.items())):
            if i >= len(axes):
                break
                
            if label == -1:  # Skip noise cluster
                continue
                
            # Plot pattern
            if centroid.ndim == 1:
                # Try to reshape to square for visualization
                size = int(np.sqrt(len(centroid)))
                if size * size == len(centroid):
                    centroid = centroid.reshape(size, size)
                else:
                    # Plot as 1D signal
                    axes[i].plot(centroid)
                    axes[i].set_title(f'Cluster {label}')
                    continue
            
            # Plot 2D pattern
            axes[i].imshow(centroid, cmap='viridis')
            axes[i].set_title(f'Cluster {label}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Cluster representatives visualization saved to {output_file}")
    
    def get_cluster_for_pattern(self, pattern: Union[Tryte, np.ndarray, torch.Tensor]) -> int:
        """Get the cluster assignment for a new pattern.
        
        Args:
            pattern: Pattern to classify
            
        Returns:
            Cluster ID
        """
        # Convert pattern to numpy array
        if isinstance(pattern, torch.Tensor):
            if pattern.is_cuda or pattern.device.type == 'mps':
                pattern_array = pattern.cpu().numpy()
            else:
                pattern_array = pattern.numpy()
        elif isinstance(pattern, Tryte):
            pattern_array = self.pattern_memory._tryte_to_state_field(pattern)
        elif isinstance(pattern, np.ndarray):
            pattern_array = pattern
        else:
            raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
        
        # Flatten pattern
        flat_pattern = pattern_array.flatten()
        
        # Apply same dimensionality reduction
        if self.dimensionality_reduction == 'pca' and hasattr(self.model, 'components_'):
            # For PCA, project onto principal components
            reduced_pattern = np.dot(flat_pattern - self.model.mean_, self.model.components_.T)
        else:
            # For other methods, use the pattern as is
            reduced_pattern = flat_pattern
        
        # Predict cluster
        cluster = self.model.predict([reduced_pattern])[0]
        
        return cluster
    
    def find_similar_patterns(self, pattern: Union[Tryte, np.ndarray, torch.Tensor], 
                             top_k: int = 5) -> List[Tuple[int, float]]:
        """Find patterns similar to the input pattern using clustering.
        
        Args:
            pattern: Input pattern
            top_k: Number of similar patterns to return
            
        Returns:
            List of (pattern_idx, similarity) tuples
        """
        # Get cluster for pattern
        cluster = self.get_cluster_for_pattern(pattern)
        
        # Get patterns in the same cluster
        if cluster in self.cluster_assignments:
            cluster_patterns = self.cluster_assignments[cluster]
        else:
            # If cluster not found, return empty list
            return []
        
        # Find similar patterns using pattern memory
        return self.pattern_memory.recognize_pattern(pattern, top_k)
