"""Hierarchical pattern recognition for UNIFIED Consciousness Engine.

This module implements hierarchical pattern recognition capabilities,
allowing the system to recognize patterns of patterns and form
higher-level abstractions from lower-level features.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field

from .learning import PatternMemory
from unified.trits.tryte import Tryte


class HierarchicalPatternNetwork:
    """A hierarchical network of pattern memories.
    
    This class implements a multi-level pattern recognition system,
    where each level recognizes patterns in the level below it.
    
    Attributes:
        levels: List of pattern memories at each level
        level_connections: Connections between levels
        max_levels: Maximum number of levels in the hierarchy
    """
    
    def __init__(self, input_dimensions: Tuple[int, int], max_levels: int = 3,
                 patterns_per_level: int = 1000, sparsity_increase: float = 0.1):
        """Initialize hierarchical pattern network.
        
        Args:
            input_dimensions: Dimensions of input patterns (height, width)
            max_levels: Maximum number of levels in the hierarchy
            patterns_per_level: Maximum patterns stored at each level
            sparsity_increase: How much sparsity increases at each level
        """
        self.input_dimensions = input_dimensions
        self.max_levels = max_levels
        
        # Create pattern memories for each level
        self.levels = []
        for level in range(max_levels):
            # Increase sparsity threshold at higher levels
            sparse_threshold = 0.1 + level * sparsity_increase
            
            # Create pattern memory for this level
            self.levels.append(PatternMemory(
                max_patterns=patterns_per_level,
                sparse_threshold=sparse_threshold,
                parallel_threshold=10,
                max_workers=4
            ))
        
        # Track connections between levels
        self.level_connections = [[] for _ in range(max_levels - 1)]
        
        # Track patterns at each level
        self.level_patterns = [[] for _ in range(max_levels)]
        
        # Activation thresholds for each level
        self.activation_thresholds = [0.7 - 0.1 * level for level in range(max_levels)]
    
    def learn_pattern(self, pattern: Union[Tryte, np.ndarray], max_level: Optional[int] = None):
        """Learn a pattern hierarchically.
        
        Args:
            pattern: Input pattern (Tryte or numpy array)
            max_level: Maximum level to learn up to (None for all levels)
        
        Returns:
            Dictionary of pattern indices at each level
        """
        if max_level is None:
            max_level = self.max_levels - 1
        
        # Convert to numpy array if needed
        if isinstance(pattern, Tryte):
            # Level 0: Base pattern
            state_field = self.levels[0]._tryte_to_state_field(pattern)
        else:
            state_field = pattern
        
        # Learn at each level
        level_indices = {}
        current_pattern = state_field
        
        for level in range(min(max_level + 1, self.max_levels)):
            # Learn pattern at this level
            pattern_idx = self.levels[level].learn_pattern(current_pattern)
            level_indices[level] = pattern_idx
            
            # Store pattern for this level
            self.level_patterns[level].append(pattern_idx)
            
            # If not at max level, prepare pattern for next level
            if level < max_level and level < self.max_levels - 1:
                # Extract features from this level to form higher-level pattern
                next_pattern = self._extract_higher_level_features(current_pattern, level)
                
                # Connect patterns between levels
                self.level_connections[level].append((pattern_idx, len(self.level_patterns[level+1])))
                
                # Update current pattern for next level
                current_pattern = next_pattern
        
        return level_indices
    
    def recognize_pattern(self, pattern: Union[Tryte, np.ndarray], 
                         start_level: int = 0,
                         propagate_up: bool = True) -> Dict[int, List[Tuple[int, float]]]:
        """Recognize a pattern hierarchically.
        
        Args:
            pattern: Input pattern to recognize
            start_level: Level to start recognition from
            propagate_up: Whether to propagate recognition up the hierarchy
        
        Returns:
            Dictionary mapping level to list of (pattern_idx, similarity) tuples
        """
        # Convert to numpy array if needed
        if isinstance(pattern, Tryte):
            state_field = self.levels[0]._tryte_to_state_field(pattern)
        else:
            state_field = pattern
        
        # Recognize at each level
        results = {}
        current_pattern = state_field
        
        for level in range(start_level, self.max_levels):
            # Recognize pattern at this level
            matches = self.levels[level].recognize_pattern(current_pattern, top_k=3)
            results[level] = matches
            
            # Stop if no good matches or not propagating up
            if not propagate_up or not matches or matches[0][1] < self.activation_thresholds[level]:
                break
            
            # If not at max level, prepare pattern for next level
            if level < self.max_levels - 1:
                # Get the best matching pattern at this level
                best_match_idx = matches[0][0]
                
                # Find connected patterns at next level
                next_level_patterns = []
                for idx, next_idx in self.level_connections[level]:
                    if idx == best_match_idx:
                        next_level_patterns.append(next_idx)
                
                if not next_level_patterns:
                    # No connections to next level, extract features from current pattern
                    next_pattern = self._extract_higher_level_features(current_pattern, level)
                else:
                    # Use the connected pattern at the next level
                    next_pattern = self.levels[level+1].get_attractor_pattern(next_level_patterns[0])
                
                # Update current pattern for next level
                current_pattern = next_pattern
        
        return results
    
    def _extract_higher_level_features(self, pattern: np.ndarray, level: int) -> np.ndarray:
        """Extract higher-level features from a pattern.
        
        Args:
            pattern: Input pattern
            level: Current level
        
        Returns:
            Higher-level pattern for the next level
        """
        # Get pattern dimensions
        height, width = pattern.shape
        
        # For higher levels, use larger receptive fields
        receptive_field_size = 2 * (level + 1)
        stride = max(1, receptive_field_size // 2)
        
        # Calculate output dimensions
        out_height = max(1, (height - receptive_field_size) // stride + 1)
        out_width = max(1, (width - receptive_field_size) // stride + 1)
        
        # Initialize higher-level pattern
        higher_pattern = np.zeros((out_height, out_width))
        
        # Extract features using sliding window
        for i in range(0, height - receptive_field_size + 1, stride):
            for j in range(0, width - receptive_field_size + 1, stride):
                # Extract receptive field
                receptive_field = pattern[i:i+receptive_field_size, j:j+receptive_field_size]
                
                # Calculate feature value (e.g., average activation, max activation)
                feature_value = np.mean(receptive_field)
                
                # Set feature in higher-level pattern
                out_i = i // stride
                out_j = j // stride
                if out_i < out_height and out_j < out_width:
                    higher_pattern[out_i, out_j] = feature_value
        
        return higher_pattern
    
    def visualize_hierarchy(self, level: int = 0, pattern_idx: int = 0):
        """Visualize patterns in the hierarchy.
        
        Args:
            level: Starting level
            pattern_idx: Pattern index at starting level
        
        Returns:
            List of patterns at each level connected to the starting pattern
        """
        import matplotlib.pyplot as plt
        
        # Get pattern at starting level
        pattern = self.levels[level].get_attractor_pattern(pattern_idx)
        if pattern is None:
            return None
        
        # Initialize list of patterns
        patterns = [pattern]
        current_level = level
        current_idx = pattern_idx
        
        # Traverse up the hierarchy
        while current_level < self.max_levels - 1:
            # Find connected patterns at next level
            next_indices = []
            for idx, next_idx in self.level_connections[current_level]:
                if idx == current_idx:
                    next_indices.append(next_idx)
            
            if not next_indices:
                break
            
            # Get pattern at next level
            next_pattern = self.levels[current_level+1].get_attractor_pattern(next_indices[0])
            if next_pattern is None:
                break
            
            # Add pattern to list
            patterns.append(next_pattern)
            
            # Update current level and index
            current_level += 1
            current_idx = next_indices[0]
        
        # Visualize patterns
        fig, axes = plt.subplots(1, len(patterns), figsize=(4*len(patterns), 4))
        if len(patterns) == 1:
            axes = [axes]
        
        for i, p in enumerate(patterns):
            axes[i].imshow(p, cmap='viridis')
            axes[i].set_title(f"Level {level+i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"hierarchy_level{level}_pattern{pattern_idx}.png")
        plt.close()
        
        return patterns
