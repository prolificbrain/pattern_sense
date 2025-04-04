"""Learning mechanisms for the UNIFIED Consciousness Engine.

This module implements various learning mechanisms for MTU networks, including
Hebbian learning, pattern recognition, and attractor formation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import concurrent.futures
from functools import partial

from .mtu import MinimalThinkingUnit
from unified.trits.tryte import Tryte


@dataclass
class HebbianLearning:
    """Implements Hebbian learning for MTU connections.
    
    Hebbian learning is based on the principle that "neurons that fire together,
    wire together". This class implements this mechanism for MTU networks.
    
    Attributes:
        learning_rate: How quickly connections strengthen
        decay_rate: How quickly unused connections weaken
        min_weight: Minimum connection weight
        max_weight: Maximum connection weight
    """
    
    learning_rate: float = 0.05
    decay_rate: float = 0.001
    min_weight: float = 0.0
    max_weight: float = 1.0
    _coactivation_history: Dict[Tuple[int, int], List[float]] = field(default_factory=dict, init=False, repr=False)
    _connection_strengths: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False, repr=False)
    
    def initialize_connections(self, connections: Dict[int, List[int]], initial_weight: float = 0.5) -> Dict[Tuple[int, int], float]:
        """Initialize connection strengths.
        
        Args:
            connections: Dictionary mapping MTU indices to lists of connected MTU indices
            initial_weight: Initial weight for all connections
            
        Returns:
            Dictionary mapping (source, target) tuples to connection strengths
        """
        # Initialize connection strengths
        for source, targets in connections.items():
            for target in targets:
                self._connection_strengths[(source, target)] = initial_weight
                self._coactivation_history[(source, target)] = [0.0] * 10  # Sliding window of recent coactivations
        
        return self._connection_strengths
    
    def update_connections(self, 
                          active_mtus: List[int], 
                          mtu_outputs: Dict[int, float], 
                          connections: Dict[int, List[int]], 
                          learning_scale: float = 1.0) -> Dict[Tuple[int, int], float]:
        """Update connection strengths based on Hebbian learning rule.
        
        Args:
            active_mtus: List of indices of currently active MTUs
            mtu_outputs: Dictionary mapping MTU indices to their output strengths
            connections: Dictionary mapping MTU indices to lists of connected MTU indices
            learning_scale: Scaling factor for learning rate (for adaptive learning)
            
        Returns:
            Updated connection strengths
        """
        # Apply decay to all connections
        for key in self._connection_strengths:
            # Decay unused connections
            self._connection_strengths[key] = max(
                self.min_weight,
                self._connection_strengths[key] - self.decay_rate * learning_scale
            )
        
        # Update connections for active MTUs
        for source in active_mtus:
            if source not in connections:
                continue
                
            source_output = mtu_outputs.get(source, 0.0)
            
            for target in connections[source]:
                target_output = mtu_outputs.get(target, 0.0)
                
                # Calculate coactivation (product of outputs)
                coactivation = source_output * target_output
                
                # Update coactivation history (sliding window)
                self._coactivation_history[(source, target)].append(coactivation)
                if len(self._coactivation_history[(source, target)]) > 10:
                    self._coactivation_history[(source, target)].pop(0)
                
                # Calculate recent average coactivation
                avg_coactivation = sum(self._coactivation_history[(source, target)]) / len(self._coactivation_history[(source, target)])
                
                # Apply Hebbian update rule (strengthen connections between coactive MTUs)
                if avg_coactivation > 0:
                    self._connection_strengths[(source, target)] = min(
                        self.max_weight,
                        self._connection_strengths[(source, target)] + (self.learning_rate * learning_scale) * avg_coactivation
                    )
        
        return self._connection_strengths
    
    def update_connection(self, current_strength: float, coactivation: bool, learning_scale: float = 1.0) -> float:
        """Update connection strength based on Hebbian learning rule.
        
        Args:
            current_strength: Current connection strength
            coactivation: Whether the connected neurons were active together
            learning_scale: Scaling factor for learning rate (for adaptive learning)
            
        Returns:
            Updated connection strength
        """
        if coactivation:
            # Strengthen connection when neurons fire together
            new_strength = current_strength + (self.learning_rate * learning_scale) * (1 - current_strength)
        else:
            # Weaken connection otherwise
            new_strength = current_strength - (self.decay_rate * learning_scale) * current_strength
            
        # Keep strength within bounds
        return min(1.0, max(0.0, new_strength))


class PatternMemory:
    """Memory for storing and retrieving patterns.
    
    This class implements a pattern memory that can store and retrieve patterns.
    It uses sparse representation for efficiency with high-dimensional patterns.
    
    Attributes:
        _patterns: List of stored patterns (sparse matrices)
        _pattern_strengths: Strength of each pattern (0.0-1.0)
        _pattern_activations: Number of times each pattern was activated
        _max_patterns: Maximum number of patterns to store
        _sparse_threshold: Sparsity threshold for conversion to sparse format
        _parallel_threshold: Minimum batch size for parallel processing
        _max_workers: Maximum number of worker threads for parallel processing
    """
    
    def __init__(self, max_patterns: int = 1000, sparse_threshold: float = 0.1,
                 parallel_threshold: int = 10, max_workers: int = 4):
        """Initialize pattern memory.
        
        Args:
            max_patterns: Maximum number of patterns to store
            sparse_threshold: Threshold for sparse representation (fraction of non-zeros)
            parallel_threshold: Minimum batch size for parallel processing
            max_workers: Maximum number of worker threads for parallel processing
        """
        self._patterns = []
        self._pattern_strengths = []
        self._pattern_activations = []
        self._max_patterns = max_patterns
        self._sparse_threshold = sparse_threshold
        self._parallel_threshold = parallel_threshold
        self._max_workers = max_workers
        
    def learn_pattern(self, pattern: Union[Tryte, np.ndarray]) -> int:
        """Learn a new pattern by storing it in memory.
        
        Args:
            pattern: Input pattern (Tryte object or numpy array)
            
        Returns:
            Index of stored pattern
            
        Raises:
            ValueError: If pattern is invalid or memory is full
        """
        # Vectorized conversion from Tryte to state field
        try:
            # Handle different input types
            if isinstance(pattern, Tryte):
                # Convert Tryte to state field
                state_field = self._tryte_to_state_field(pattern)
            elif isinstance(pattern, np.ndarray):
                # Use numpy array directly
                state_field = pattern
            else:
                raise ValueError(f"Expected Tryte or numpy array, got {type(pattern).__name__}")
            
            # Check memory capacity
            if len(self._patterns) >= self._max_patterns:
                raise ValueError(f"Pattern memory full (max {self._max_patterns} patterns)")
            
            # Use sparse representation if pattern is sparse enough
            sparsity = np.count_nonzero(state_field) / (state_field.size)
            
            # Normalize the state field
            norm = np.linalg.norm(state_field)
            if norm > 1e-10:  # Avoid division by zero
                norm_state = state_field / norm
            else:
                norm_state = state_field
                
            # Store as sparse matrix if sparse enough
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
            
    def learn_patterns_batch(self, patterns: List[Union[Tryte, np.ndarray]]) -> List[int]:
        """Learn multiple patterns in parallel.
        
        Args:
            patterns: List of Tryte patterns to learn
            
        Returns:
            List of indices for stored patterns
        """
        # Use parallel processing if batch size exceeds threshold
        if len(patterns) >= self._parallel_threshold:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Process patterns in parallel
                future_to_pattern = {executor.submit(self._process_pattern, pattern): i 
                                    for i, pattern in enumerate(patterns)}
                
                # Collect results as they complete
                processed_patterns = [None] * len(patterns)
                for future in concurrent.futures.as_completed(future_to_pattern):
                    pattern_idx = future_to_pattern[future]
                    try:
                        processed_patterns[pattern_idx] = future.result()
                    except Exception as e:
                        print(f"Error processing pattern {pattern_idx}: {e}")
                        processed_patterns[pattern_idx] = None
            
            # Store all valid processed patterns
            indices = []
            for state_field in processed_patterns:
                if state_field is not None:
                    idx = self._store_pattern(state_field)
                    indices.append(idx)
                else:
                    indices.append(None)
            
            return indices
        else:
            # Process sequentially for small batches
            return [self.learn_pattern(pattern) for pattern in patterns]
    
    def _process_pattern(self, pattern: Union[Tryte, np.ndarray]) -> np.ndarray:
        """Process a pattern for storage (without storing it).
        
        Args:
            pattern: Input pattern (Tryte object or numpy array)
            
        Returns:
            Processed state field ready for storage
        """
        if not isinstance(pattern, (Tryte, np.ndarray)):
            raise ValueError(f"Expected Tryte or numpy array, got {type(pattern).__name__}")
        
        # Convert Tryte to state field
        if isinstance(pattern, Tryte):
            state_field = self._tryte_to_state_field(pattern)
        else:
            state_field = pattern
            
        # Normalize the state field
        norm = np.linalg.norm(state_field)
        if norm > 1e-10:  # Avoid division by zero
            norm_state = state_field / norm
        else:
            norm_state = state_field
            
        # Use sparse representation if pattern is sparse enough
        sparsity = np.count_nonzero(state_field) / (pattern.height * pattern.width)
        if sparsity < self._sparse_threshold:
            from scipy import sparse
            norm_state = sparse.csr_matrix(norm_state)
            
        return norm_state
    
    def _store_pattern(self, norm_state) -> int:
        """Store a processed pattern in memory.
        
        Args:
            norm_state: Normalized state field to store
            
        Returns:
            Index of stored pattern
        """
        # Check memory capacity
        if len(self._patterns) >= self._max_patterns:
            raise ValueError(f"Pattern memory full (max {self._max_patterns} patterns)")
            
        # Store the pattern
        self._patterns.append(norm_state)
        self._pattern_strengths.append(1.0)
        self._pattern_activations.append(1)
        
        return len(self._patterns) - 1
        
    def get_attractor_pattern(self, pattern_idx: int) -> Optional[np.ndarray]:
        """Get a pattern from memory to serve as an attractor.
        
        Args:
            pattern_idx: Index of pattern to retrieve
            
        Returns:
            Pattern as numpy array, or None if not found
        """
        if pattern_idx < 0 or pattern_idx >= len(self._patterns):
            return None
            
        # Increment activation count
        self._pattern_activations[pattern_idx] += 1
        
        # Convert sparse matrix to dense if needed
        pattern = self._patterns[pattern_idx]
        if hasattr(pattern, 'toarray'):  # Check if it's a sparse matrix
            return pattern.toarray()
        return pattern
    
    def recognize_pattern(self, input_pattern: Union[Tryte, np.ndarray], top_k: int = 1) -> List[Tuple[int, float]]:
        """Recognize a pattern by finding the most similar patterns in memory.
        
        Args:
            input_pattern: Input pattern (Tryte or numpy array)
            top_k: Number of top matches to return
            
        Returns:
            List of (pattern_idx, similarity) tuples, sorted by similarity
        """
        # Convert input to normalized state field if it's a Tryte
        if isinstance(input_pattern, Tryte):
            state_field = self._tryte_to_state_field(input_pattern)
        else:
            state_field = input_pattern
            
        # Normalize input
        norm = np.linalg.norm(state_field)
        if norm > 1e-10:
            norm_state = state_field / norm
        else:
            norm_state = state_field
            
        # Calculate similarity to each pattern
        similarities = []
        
        for i, pattern in enumerate(self._patterns):
            # Convert sparse matrix to dense if needed
            if hasattr(pattern, 'toarray'):
                dense_pattern = pattern.toarray()
            else:
                dense_pattern = pattern
                
            # Calculate similarity
            sim = self._calculate_similarity(norm_state, dense_pattern)
            similarities.append((i, sim))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply reinforcement to top matches
        for idx, sim in similarities[:top_k]:
            self._reinforce_pattern(idx, sim)
            
        return similarities[:top_k]
    
    def recognize_patterns_batch(self, patterns: List[Union[Tryte, np.ndarray]], 
                               top_k: int = 1) -> List[List[Tuple[int, float]]]:
        """Recognize multiple patterns in parallel.
        
        Args:
            patterns: List of patterns to recognize
            top_k: Number of top matches to return for each pattern
            
        Returns:
            List of recognition results for each pattern
        """
        # Convert all patterns to state fields
        state_fields = []
        for pattern in patterns:
            if isinstance(pattern, Tryte):
                state_fields.append(self._tryte_to_state_field(pattern))
            else:
                state_fields.append(pattern)
        
        # Use parallel processing if batch size exceeds threshold
        if len(patterns) >= self._parallel_threshold:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                # Create a partial function with fixed parameters
                recognize_func = partial(self._calculate_similarities, top_k=top_k)
                
                # Submit all tasks
                future_to_pattern = {executor.submit(recognize_func, state_field): i 
                                   for i, state_field in enumerate(state_fields)}
                
                # Collect results
                results = [None] * len(patterns)
                for future in concurrent.futures.as_completed(future_to_pattern):
                    pattern_idx = future_to_pattern[future]
                    try:
                        results[pattern_idx] = future.result()
                    except Exception as e:
                        print(f"Error recognizing pattern {pattern_idx}: {e}")
                        results[pattern_idx] = []
            
            return results
        else:
            # Process sequentially for small batches
            return [self.recognize_pattern(pattern, top_k) for pattern in patterns]
    
    def _calculate_similarities(self, state_field: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Calculate similarities between a state field and all stored patterns.
        
        Args:
            state_field: Input state field
            top_k: Number of top matches to return
            
        Returns:
            List of (pattern_idx, similarity) tuples
        """
        # Normalize input
        norm = np.linalg.norm(state_field)
        if norm > 1e-10:
            norm_state = state_field / norm
        else:
            norm_state = state_field
            
        # Calculate similarity to each pattern
        similarities = []
        
        for i, pattern in enumerate(self._patterns):
            # Convert sparse matrix to dense if needed
            if hasattr(pattern, 'toarray'):
                dense_pattern = pattern.toarray()
            else:
                dense_pattern = pattern
                
            # Calculate similarity
            sim = self._calculate_similarity(norm_state, dense_pattern)
            similarities.append((i, sim))
            
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply reinforcement to top matches
        for idx, sim in similarities[:top_k]:
            self._reinforce_pattern(idx, sim)
            
        return similarities[:top_k]
    
    def _calculate_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns using multiple metrics.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Ensure patterns have same shape
        if pattern1.shape != pattern2.shape:
            raise ValueError(f"Pattern shapes don't match: {pattern1.shape} vs {pattern2.shape}")
        
        # Cosine similarity (dot product of normalized vectors)
        # Flatten patterns for dot product
        p1_flat = pattern1.flatten()
        p2_flat = pattern2.flatten()
        
        # Normalize if needed
        p1_norm = np.linalg.norm(p1_flat)
        if p1_norm > 1e-10:
            p1_flat = p1_flat / p1_norm
        
        p2_norm = np.linalg.norm(p2_flat)
        if p2_norm > 1e-10:
            p2_flat = p2_flat / p2_norm
        
        # Calculate cosine similarity
        cosine_sim = np.dot(p1_flat, p2_flat)
        
        # For simple patterns, use mean squared error as additional metric
        mse = np.mean((pattern1 - pattern2) ** 2)
        mse_sim = 1.0 / (1.0 + 10.0 * mse)  # Convert to similarity (0-1)
        
        # Weighted combination
        return 0.7 * cosine_sim + 0.3 * mse_sim
    
    def _reinforce_pattern(self, pattern_idx: int, similarity: float) -> None:
        """Reinforce a pattern based on recognition strength.
        
        Args:
            pattern_idx: Index of pattern to reinforce
            similarity: Similarity score that triggered reinforcement
        """
        if 0 <= pattern_idx < len(self._patterns):
            # Increment activation count
            self._pattern_activations[pattern_idx] += 1
            
            # Strengthen pattern based on similarity and activation frequency
            activation_factor = np.log1p(self._pattern_activations[pattern_idx]) / 10
            reinforcement = similarity * activation_factor
            
            # Cap maximum strength
            self._pattern_strengths[pattern_idx] = min(2.0, 
                                                     self._pattern_strengths[pattern_idx] + reinforcement)
    
    def _tryte_to_state_field(self, tryte: Tryte) -> np.ndarray:
        """Convert a Tryte to a state field representation.
        
        Args:
            tryte: Input Tryte
            
        Returns:
            State field as numpy array
        """
        # Reuse code from learn_pattern but return the state field
        state_field = np.zeros((tryte.height, tryte.width))
        
        # Extract coordinates and values
        coords = []
        values = []
        
        for trit in tryte.trits:
            if hasattr(trit.orientation, 'x'):  # Object-style
                coords.append((trit.orientation.x, trit.orientation.y))
            elif isinstance(trit.orientation, dict):  # Dict-style
                coords.append((trit.orientation['x'], trit.orientation['y']))
            elif isinstance(trit.orientation, (tuple, list)):  # Tuple-style
                coords.append((float(trit.orientation[0]), float(trit.orientation[1])))
            values.append(trit.value)
        
        # Convert to numpy arrays
        coords = np.array(coords)
        values = np.array(values)
        
        # Filter valid coordinates
        if len(coords) > 0:  # Only if we have coordinates
            valid = (coords[:,0] >= 0) & (coords[:,0] < tryte.width) & \
                    (coords[:,1] >= 0) & (coords[:,1] < tryte.height)
            
            # Apply valid coordinates
            valid_coords = coords[valid]
            valid_values = values[valid]
            
            rows = valid_coords[:,1].astype(int)
            cols = valid_coords[:,0].astype(int)
            state_field[rows, cols] = valid_values
            
        return state_field


@dataclass
class AttractorDynamics:
    """Implements attractor dynamics for the MTU state fields.
    
    This class enhances the state field dynamics to form attractors that
    pull the system toward learned patterns.
    
    Attributes:
        attractor_strength: How strongly attractors pull the state
        basin_width: Width of attractor basins in state space
    """
    
    attractor_strength: float = 0.2
    basin_width: float = 2.0
    
    def apply_attractors(self, 
                        state_field: np.ndarray, 
                        attractors: List[np.ndarray], 
                        attractor_strengths: List[float]) -> np.ndarray:
        """Apply attractor dynamics to state field.
        
        Args:
            state_field: Current state field
            attractors: List of attractor patterns
            attractor_strengths: Strength of each attractor
            
        Returns:
            Updated state field with attractor influence
        """
        if not attractors:
            return state_field
        
        # Normalize state field
        norm_state = state_field / (np.linalg.norm(state_field) + 1e-10)
        
        # Calculate influence from each attractor
        attractor_field = np.zeros_like(state_field)
        
        for i, attractor in enumerate(attractors):
            # Calculate similarity
            similarity = np.sum(norm_state * attractor)
            
            # Apply attractor influence based on similarity and strength
            # The closer the state is to the attractor, the stronger the pull
            # This creates basin-like dynamics in state space
            attraction = attractor_strengths[i] * np.exp(similarity / self.basin_width)
            attractor_field += attraction * attractor
        
        # Update field with attractor influence
        updated_field = state_field + self.attractor_strength * attractor_field
        
        # Apply nonlinear dynamics (helps form stable attractors)
        updated_field = np.tanh(updated_field)
        
        return updated_field


@dataclass
class AdaptiveLearning:
    """Implements adaptive learning rate adjustments based on network performance.
    
    This class modifies learning rates during training to optimize convergence
    and performance. It increases rates when progress is slow and decreases them
    when performance is unstable.
    
    Attributes:
        initial_rate: Starting learning rate
        min_rate: Minimum allowed learning rate
        max_rate: Maximum allowed learning rate
        adaptation_speed: How quickly to adjust the learning rate
    """
    
    initial_rate: float = 0.01
    min_rate: float = 0.001
    max_rate: float = 0.1
    adaptation_speed: float = 0.1
    
    _current_rate: float = field(init=False)
    _performance_history: List[float] = field(default_factory=list, init=False)
    
    def __post_init__(self):
        """Initialize adaptive learning."""
        self._current_rate = self.initial_rate
    
    def update_rate(self, current_performance: float) -> float:
        """Update the learning rate based on performance metrics.
        
        Args:
            current_performance: Current performance metric (e.g., accuracy or loss)
            
        Returns:
            New learning rate
        """
        # Add current performance to history
        self._performance_history.append(current_performance)
        
        # Need at least 3 data points to make meaningful adjustments
        if len(self._performance_history) < 3:
            return self._current_rate
        
        # Calculate performance change rate
        recent_perf = self._performance_history[-3:]
        perf_change = (recent_perf[2] - recent_perf[0]) / 2
        
        # If performance is improving rapidly, keep current rate
        if perf_change > 0.05:
            # No change needed
            pass
        # If performance is improving slowly, increase learning rate
        elif 0 < perf_change < 0.05:
            self._current_rate = min(self._current_rate * (1 + self.adaptation_speed), self.max_rate)
        # If performance is declining, decrease learning rate
        elif perf_change < 0:
            self._current_rate = max(self._current_rate * (1 - self.adaptation_speed), self.min_rate)
        
        return self._current_rate
    
    def get_current_rate(self) -> float:
        """Get the current learning rate.
        
        Returns:
            Current learning rate
        """
        return self._current_rate
