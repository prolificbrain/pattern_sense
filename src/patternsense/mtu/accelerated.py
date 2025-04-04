"""GPU-accelerated pattern recognition for UNIFIED Consciousness Engine.

This module implements GPU-accelerated versions of the pattern recognition
algorithms, providing significant speedups for large-scale pattern processing.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
import concurrent.futures
from functools import partial

from .learning import PatternMemory
from unified.trits.tryte import Tryte


class AcceleratedPatternMemory(PatternMemory):
    """GPU-accelerated pattern memory implementation.
    
    This class extends PatternMemory with GPU acceleration for faster
    pattern learning and recognition on large datasets.
    
    Attributes:
        device: PyTorch device (CPU or GPU)
        use_gpu: Whether to use GPU acceleration
        batch_size: Batch size for GPU operations
    """
    
    def __init__(self, max_patterns: int = 1000, sparse_threshold: float = 0.1,
                 parallel_threshold: int = 10, max_workers: int = 4,
                 use_gpu: bool = True, batch_size: int = 64):
        """Initialize accelerated pattern memory.
        
        Args:
            max_patterns: Maximum number of patterns to store
            sparse_threshold: Threshold for sparse representation
            parallel_threshold: Minimum batch size for parallel processing
            max_workers: Maximum number of worker threads
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for GPU operations
        """
        super().__init__(max_patterns, sparse_threshold, parallel_threshold, max_workers)
        
        # GPU settings
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Convert patterns to PyTorch tensors if using GPU
        if self.use_gpu:
            self._tensor_patterns = []
        
        print(f"Using device: {self.device}")
    
    def learn_pattern(self, pattern: Union[Tryte, np.ndarray, torch.Tensor]) -> int:
        """Learn a new pattern with GPU acceleration.
        
        Args:
            pattern: Input pattern (Tryte, numpy array, or PyTorch tensor)
            
        Returns:
            Index of stored pattern
        """
        try:
            # Handle different input types
            if isinstance(pattern, torch.Tensor):
                # Use tensor directly
                tensor_pattern = pattern.to(self.device)
                # Convert to numpy for storage in parent class
                if tensor_pattern.is_cuda or tensor_pattern.device.type == 'mps':
                    state_field = tensor_pattern.cpu().numpy()
                else:
                    state_field = tensor_pattern.numpy()
            elif isinstance(pattern, Tryte):
                # Convert Tryte to state field
                state_field = self._tryte_to_state_field(pattern)
                # Convert to tensor for GPU operations
                tensor_pattern = torch.from_numpy(state_field).float().to(self.device)
            elif isinstance(pattern, np.ndarray):
                # Use numpy array directly
                state_field = pattern
                # Convert to tensor for GPU operations
                tensor_pattern = torch.from_numpy(state_field).float().to(self.device)
            else:
                raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
            
            # Check memory capacity
            if len(self._patterns) >= self._max_patterns:
                raise ValueError(f"Pattern memory full (max {self._max_patterns} patterns)")
            
            # Normalize on GPU for efficiency
            norm = torch.norm(tensor_pattern)
            if norm > 1e-10:
                norm_tensor = tensor_pattern / norm
            else:
                norm_tensor = tensor_pattern
            
            # Store normalized pattern
            if self.use_gpu:
                self._tensor_patterns.append(norm_tensor)
            
            # Convert back to numpy for storage in parent class
            if norm_tensor.is_cuda or norm_tensor.device.type == 'mps':
                norm_state = norm_tensor.cpu().numpy()
            else:
                norm_state = norm_tensor.numpy()
            
            # Use sparse representation if pattern is sparse enough
            sparsity = np.count_nonzero(norm_state) / norm_state.size
            if sparsity < self._sparse_threshold:
                from scipy import sparse
                norm_state = sparse.csr_matrix(norm_state)
            
            # Store the pattern
            self._patterns.append(norm_state)
            self._pattern_strengths.append(1.0)
            self._pattern_activations.append(1)
            
            return len(self._patterns) - 1
            
        except Exception as e:
            raise ValueError(f"Failed to learn pattern: {str(e)}")
    
    def learn_patterns_batch(self, patterns: List[Union[Tryte, np.ndarray, torch.Tensor]]) -> List[int]:
        """Learn multiple patterns in a GPU-accelerated batch.
        
        Args:
            patterns: List of patterns to learn
            
        Returns:
            List of indices for stored patterns
        """
        if not self.use_gpu or len(patterns) < self.batch_size:
            # Fall back to CPU implementation for small batches
            return super().learn_patterns_batch(patterns)
        
        # Process in batches for GPU efficiency
        indices = []
        for i in range(0, len(patterns), self.batch_size):
            batch = patterns[i:i+self.batch_size]
            
            # Convert batch to tensors
            tensor_batch = []
            for pattern in batch:
                if isinstance(pattern, torch.Tensor):
                    tensor_batch.append(pattern.to(self.device))
                elif isinstance(pattern, Tryte):
                    state_field = self._tryte_to_state_field(pattern)
                    tensor_batch.append(torch.from_numpy(state_field).float().to(self.device))
                elif isinstance(pattern, np.ndarray):
                    tensor_batch.append(torch.from_numpy(pattern).float().to(self.device))
                else:
                    raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
            
            # Stack tensors into a single batch tensor
            if tensor_batch:
                # Ensure all tensors have the same shape
                shapes = [t.shape for t in tensor_batch]
                if len(set(shapes)) > 1:
                    # Handle different shapes individually
                    batch_indices = [self.learn_pattern(p) for p in batch]
                else:
                    # Process as a single batch
                    stacked_batch = torch.stack(tensor_batch)
                    
                    # Normalize batch
                    norms = torch.norm(stacked_batch, dim=(1, 2), keepdim=True)
                    # Avoid division by zero
                    norms = torch.where(norms > 1e-10, norms, torch.ones_like(norms))
                    norm_batch = stacked_batch / norms
                    
                    # Store each normalized pattern
                    batch_indices = []
                    for j in range(norm_batch.shape[0]):
                        norm_tensor = norm_batch[j]
                        if self.use_gpu:
                            self._tensor_patterns.append(norm_tensor)
                        
                        # Convert to numpy for storage
                        norm_state = norm_tensor.cpu().numpy()
                        
                        # Use sparse representation if needed
                        sparsity = np.count_nonzero(norm_state) / norm_state.size
                        if sparsity < self._sparse_threshold:
                            from scipy import sparse
                            norm_state = sparse.csr_matrix(norm_state)
                        
                        # Store pattern
                        self._patterns.append(norm_state)
                        self._pattern_strengths.append(1.0)
                        self._pattern_activations.append(1)
                        
                        batch_indices.append(len(self._patterns) - 1)
                
                indices.extend(batch_indices)
        
        return indices
    
    def recognize_pattern(self, pattern: Union[Tryte, np.ndarray, torch.Tensor], top_k: int = 1) -> List[Tuple[int, float]]:
        """Recognize a pattern using GPU acceleration.
        
        Args:
            pattern: Pattern to recognize
            top_k: Number of top matches to return
            
        Returns:
            List of (pattern_idx, similarity) tuples
        """
        if not self.use_gpu or not hasattr(self, '_tensor_patterns') or not self._tensor_patterns:
            # Fall back to CPU implementation
            return super().recognize_pattern(pattern, top_k)
        
        # Convert input to tensor
        if isinstance(pattern, torch.Tensor):
            tensor_pattern = pattern.to(self.device)
        elif isinstance(pattern, Tryte):
            state_field = self._tryte_to_state_field(pattern)
            tensor_pattern = torch.from_numpy(state_field).float().to(self.device)
        elif isinstance(pattern, np.ndarray):
            tensor_pattern = torch.from_numpy(pattern).float().to(self.device)
        else:
            raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
        
        # Normalize input
        norm = torch.norm(tensor_pattern)
        if norm > 1e-10:
            norm_pattern = tensor_pattern / norm
        else:
            norm_pattern = tensor_pattern
        
        # Calculate similarity to each pattern in batches
        similarities = []
        
        # Process in batches for GPU efficiency
        batch_size = 100  # Larger batch for similarity calculation
        for i in range(0, len(self._tensor_patterns), batch_size):
            batch = self._tensor_patterns[i:i+batch_size]
            
            # Stack patterns into a batch
            try:
                pattern_batch = torch.stack(batch)
                
                # Reshape input pattern for batch processing
                input_pattern = norm_pattern.view(1, *norm_pattern.shape)
                
                # Calculate cosine similarity
                flat_input = input_pattern.reshape(1, -1)
                flat_batch = pattern_batch.reshape(len(batch), -1)
                
                # Normalize batch patterns
                batch_norms = torch.norm(flat_batch, dim=1, keepdim=True)
                batch_norms = torch.where(batch_norms > 1e-10, batch_norms, torch.ones_like(batch_norms))
                norm_batch = flat_batch / batch_norms
                
                # Calculate dot product (cosine similarity)
                sims = torch.matmul(flat_input, norm_batch.t()).squeeze()
                
                # Calculate MSE similarity
                mse = torch.mean((input_pattern - pattern_batch) ** 2, dim=(1, 2))
                mse_sims = 1.0 / (1.0 + 10.0 * mse)
                
                # Combine similarities
                combined_sims = 0.7 * sims + 0.3 * mse_sims
                
                # Convert to numpy and add to results
                batch_sims = combined_sims.cpu().numpy()
                for j, sim in enumerate(batch_sims):
                    similarities.append((i + j, float(sim)))
            except Exception as e:
                # Handle patterns with different shapes individually
                for j, tensor in enumerate(batch):
                    try:
                        # Calculate similarity directly
                        if tensor.shape != norm_pattern.shape:
                            # Skip patterns with different shapes
                            continue
                        
                        # Calculate cosine similarity
                        flat_input = norm_pattern.reshape(-1)
                        flat_pattern = tensor.reshape(-1)
                        cosine_sim = torch.dot(flat_input, flat_pattern)
                        
                        # Calculate MSE similarity
                        mse = torch.mean((norm_pattern - tensor) ** 2)
                        mse_sim = 1.0 / (1.0 + 10.0 * mse)
                        
                        # Combine similarities
                        combined_sim = 0.7 * cosine_sim + 0.3 * mse_sim
                        
                        similarities.append((i + j, float(combined_sim.cpu().numpy())))
                    except Exception:
                        # Skip problematic patterns
                        continue
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply reinforcement to top matches
        for idx, sim in similarities[:top_k]:
            self._reinforce_pattern(idx, sim)
        
        return similarities[:top_k]
    
    def recognize_patterns_batch(self, patterns: List[Union[Tryte, np.ndarray, torch.Tensor]], 
                               top_k: int = 1) -> List[List[Tuple[int, float]]]:
        """Recognize multiple patterns in a GPU-accelerated batch.
        
        Args:
            patterns: List of patterns to recognize
            top_k: Number of top matches to return for each pattern
            
        Returns:
            List of recognition results for each pattern
        """
        if not self.use_gpu or not hasattr(self, '_tensor_patterns') or not self._tensor_patterns:
            # Fall back to CPU implementation
            return super().recognize_patterns_batch(patterns, top_k)
        
        # Process in batches for GPU efficiency
        results = []
        for i in range(0, len(patterns), self.batch_size):
            batch = patterns[i:i+self.batch_size]
            
            # Convert batch to tensors
            tensor_batch = []
            for pattern in batch:
                if isinstance(pattern, torch.Tensor):
                    tensor_batch.append(pattern.to(self.device))
                elif isinstance(pattern, Tryte):
                    state_field = self._tryte_to_state_field(pattern)
                    tensor_batch.append(torch.from_numpy(state_field).float().to(self.device))
                elif isinstance(pattern, np.ndarray):
                    tensor_batch.append(torch.from_numpy(pattern).float().to(self.device))
                else:
                    raise ValueError(f"Expected Tryte, numpy array, or PyTorch tensor, got {type(pattern).__name__}")
            
            # Process each pattern individually (for now)
            # Future optimization: implement true batch similarity calculation
            batch_results = [self.recognize_pattern(p, top_k) for p in tensor_batch]
            results.extend(batch_results)
        
        return results
