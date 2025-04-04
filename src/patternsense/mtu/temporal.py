"""Temporal pattern recognition for UNIFIED Consciousness Engine.

This module implements time-series pattern recognition capabilities,
allowing the system to recognize patterns that unfold over time.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Deque
from collections import deque
from dataclasses import dataclass, field

from .learning import PatternMemory
from unified.trits.tryte import Tryte


class TemporalPatternMemory:
    """Memory for storing and recognizing temporal patterns.
    
    This class extends the pattern memory concept to sequences of patterns
    that unfold over time, enabling recognition of dynamic patterns.
    
    Attributes:
        _sequence_memory: Memory for storing sequences of pattern indices
        _pattern_memory: Memory for storing individual patterns
        _temporal_window: Size of temporal window (number of time steps)
        _history: Recent history of observed patterns
    """
    
    def __init__(self, max_patterns: int = 1000, max_sequences: int = 100,
                 temporal_window: int = 5):
        """Initialize temporal pattern memory.
        
        Args:
            max_patterns: Maximum number of individual patterns to store
            max_sequences: Maximum number of sequences to store
            temporal_window: Size of temporal window (number of time steps)
        """
        # Memory for individual patterns
        self._pattern_memory = PatternMemory(max_patterns=max_patterns)
        
        # Memory for sequences
        self._sequence_memory = []
        self._sequence_strengths = []
        self._max_sequences = max_sequences
        
        # Temporal parameters
        self._temporal_window = temporal_window
        self._history = deque(maxlen=temporal_window)
        
        # Initialize history with None values
        for _ in range(temporal_window):
            self._history.append(None)
    
    def observe_pattern(self, pattern: Union[Tryte, np.ndarray]) -> Tuple[int, List[Tuple[int, float]]]:
        """Observe a new pattern and update temporal memory.
        
        Args:
            pattern: New pattern observation
            
        Returns:
            Tuple of (pattern_idx, sequence_matches)
        """
        # Learn or recognize the individual pattern
        pattern_matches = self._pattern_memory.recognize_pattern(pattern, top_k=3)
        
        if pattern_matches and pattern_matches[0][1] > 0.8:
            # Pattern is recognized, use existing index
            pattern_idx = pattern_matches[0][0]
        else:
            # New pattern, learn it
            pattern_idx = self._pattern_memory.learn_pattern(pattern)
        
        # Update history
        self._history.append(pattern_idx)
        
        # Recognize current sequence
        sequence_matches = self._recognize_sequence()
        
        # Learn sequence if it's novel
        if not sequence_matches or sequence_matches[0][1] < 0.7:
            self._learn_sequence()
        
        return pattern_idx, sequence_matches
    
    def _recognize_sequence(self) -> List[Tuple[int, float]]:
        """Recognize the current sequence in memory.
        
        Returns:
            List of (sequence_idx, similarity) tuples
        """
        # Get current sequence from history
        current_sequence = list(self._history)
        
        # Skip if history contains None values
        if None in current_sequence:
            return []
        
        # Calculate similarity to each stored sequence
        similarities = []
        
        for i, sequence in enumerate(self._sequence_memory):
            # Calculate sequence similarity
            sim = self._calculate_sequence_similarity(current_sequence, sequence)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches
        return similarities[:3]
    
    def _learn_sequence(self) -> int:
        """Learn the current sequence.
        
        Returns:
            Index of the stored sequence
        """
        # Get current sequence from history
        current_sequence = list(self._history)
        
        # Skip if history contains None values
        if None in current_sequence:
            return -1
        
        # Check if memory is full
        if len(self._sequence_memory) >= self._max_sequences:
            # Find weakest sequence
            weakest_idx = np.argmin(self._sequence_strengths)
            
            # Replace weakest sequence
            self._sequence_memory[weakest_idx] = current_sequence
            self._sequence_strengths[weakest_idx] = 1.0
            
            return weakest_idx
        else:
            # Add new sequence
            self._sequence_memory.append(current_sequence)
            self._sequence_strengths.append(1.0)
            
            return len(self._sequence_memory) - 1
    
    def _calculate_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate similarity between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Ensure sequences have same length
        if len(seq1) != len(seq2):
            return 0.0
        
        # Count matching elements
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        
        # Calculate similarity
        return matches / len(seq1)
    
    def predict_next(self, top_k: int = 3) -> List[Tuple[int, float]]:
        """Predict the next pattern in the sequence.
        
        Args:
            top_k: Number of top predictions to return
            
        Returns:
            List of (pattern_idx, probability) tuples
        """
        # Recognize current sequence
        sequence_matches = self._recognize_sequence()
        
        if not sequence_matches:
            return []
        
        # Get predictions from matching sequences
        predictions = {}
        total_weight = 0.0
        
        for seq_idx, similarity in sequence_matches:
            # Skip if sequence is too short
            if len(self._sequence_memory[seq_idx]) <= self._temporal_window:
                continue
            
            # Get next pattern in sequence
            next_pattern = self._sequence_memory[seq_idx][self._temporal_window]
            
            # Add to predictions with weight based on similarity
            if next_pattern in predictions:
                predictions[next_pattern] += similarity
            else:
                predictions[next_pattern] = similarity
            
            total_weight += similarity
        
        # Normalize predictions
        if total_weight > 0:
            for pattern in predictions:
                predictions[pattern] /= total_weight
        
        # Sort predictions by probability (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:top_k]
    
    def get_pattern(self, pattern_idx: int) -> Optional[np.ndarray]:
        """Get a pattern from memory.
        
        Args:
            pattern_idx: Index of pattern to retrieve
            
        Returns:
            Pattern as numpy array, or None if not found
        """
        return self._pattern_memory.get_attractor_pattern(pattern_idx)
    
    def visualize_sequence(self, sequence_idx: int):
        """Visualize a sequence from memory.
        
        Args:
            sequence_idx: Index of sequence to visualize
        """
        import matplotlib.pyplot as plt
        
        # Check if sequence exists
        if sequence_idx < 0 or sequence_idx >= len(self._sequence_memory):
            print(f"Sequence {sequence_idx} not found")
            return
        
        # Get sequence
        sequence = self._sequence_memory[sequence_idx]
        
        # Get patterns for each step in sequence
        patterns = []
        for pattern_idx in sequence:
            pattern = self._pattern_memory.get_attractor_pattern(pattern_idx)
            if pattern is not None:
                patterns.append(pattern)
        
        # Skip if no patterns found
        if not patterns:
            print(f"No patterns found for sequence {sequence_idx}")
            return
        
        # Visualize patterns
        fig, axes = plt.subplots(1, len(patterns), figsize=(4*len(patterns), 4))
        if len(patterns) == 1:
            axes = [axes]
        
        for i, pattern in enumerate(patterns):
            axes[i].imshow(pattern, cmap='viridis')
            axes[i].set_title(f"Step {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"sequence_{sequence_idx}.png")
        plt.close()


class TimeSeriesPredictor:
    """Predictor for time-series data.
    
    This class specializes in processing and predicting time-series data
    such as sensor readings, financial data, or other sequential measurements.
    
    Attributes:
        _temporal_memory: Memory for storing temporal patterns
        _window_size: Size of sliding window for pattern extraction
        _step_size: Step size for sliding window
        _normalization: Whether to normalize input data
    """
    
    def __init__(self, window_size: int = 10, prediction_horizon: int = 5,
                 step_size: int = 1, normalization: bool = True):
        """Initialize time-series predictor.
        
        Args:
            window_size: Size of sliding window for pattern extraction
            prediction_horizon: Number of future time steps to predict
            step_size: Step size for sliding window
            normalization: Whether to normalize input data
        """
        self._window_size = window_size
        self._prediction_horizon = prediction_horizon
        self._step_size = step_size
        self._normalization = normalization
        
        # Initialize temporal memory
        self._temporal_memory = TemporalPatternMemory(
            max_patterns=1000,
            max_sequences=200,
            temporal_window=5
        )
        
        # Store statistics for normalization
        self._mean = 0.0
        self._std = 1.0
        self._min = 0.0
        self._max = 1.0
        
        # Buffer for recent observations
        self._buffer = deque(maxlen=window_size + prediction_horizon)
    
    def train(self, time_series: np.ndarray):
        """Train the predictor on a time series.
        
        Args:
            time_series: 1D array of time-series data
        """
        # Calculate statistics for normalization
        if self._normalization:
            self._mean = np.mean(time_series)
            self._std = np.std(time_series) + 1e-10  # Avoid division by zero
            self._min = np.min(time_series)
            self._max = np.max(time_series)
        
        # Process time series with sliding window
        for i in range(0, len(time_series) - self._window_size - self._prediction_horizon + 1, self._step_size):
            # Extract window
            window = time_series[i:i+self._window_size]
            
            # Normalize if needed
            if self._normalization:
                window = (window - self._mean) / self._std
            
            # Convert to 2D pattern (reshape as column vector)
            pattern = window.reshape(-1, 1)
            
            # Observe pattern
            self._temporal_memory.observe_pattern(pattern)
    
    def predict(self, recent_values: np.ndarray, num_steps: int = None) -> np.ndarray:
        """Predict future values based on recent observations.
        
        Args:
            recent_values: Recent time-series values
            num_steps: Number of steps to predict (default: prediction_horizon)
            
        Returns:
            Array of predicted values
        """
        if num_steps is None:
            num_steps = self._prediction_horizon
        
        # Ensure we have enough recent values
        if len(recent_values) < self._window_size:
            raise ValueError(f"Need at least {self._window_size} recent values")
        
        # Use the most recent window
        window = recent_values[-self._window_size:].copy()
        
        # Initialize buffer with recent values
        self._buffer.clear()
        for val in recent_values:
            self._buffer.append(val)
        
        # Normalize if needed
        if self._normalization:
            window = (window - self._mean) / self._std
        
        # Convert to 2D pattern
        pattern = window.reshape(-1, 1)
        
        # Get predictions for each step
        predictions = []
        current_pattern = pattern
        
        for _ in range(num_steps):
            # Observe current pattern
            pattern_idx, _ = self._temporal_memory.observe_pattern(current_pattern)
            
            # Predict next pattern
            next_predictions = self._temporal_memory.predict_next(top_k=1)
            
            if not next_predictions:
                # No prediction available, use last value
                next_value = self._buffer[-1]
            else:
                # Get predicted pattern
                next_pattern_idx = next_predictions[0][0]
                next_pattern = self._temporal_memory.get_pattern(next_pattern_idx)
                
                if next_pattern is None:
                    # Pattern not found, use last value
                    next_value = self._buffer[-1]
                else:
                    # Extract predicted value (last value in pattern)
                    next_value = next_pattern[-1, 0]
                    
                    # Denormalize if needed
                    if self._normalization:
                        next_value = next_value * self._std + self._mean
            
            # Add prediction to results
            predictions.append(next_value)
            
            # Update buffer
            self._buffer.append(next_value)
            
            # Update current pattern for next iteration
            window = np.array([self._buffer[i] for i in range(-self._window_size, 0)])
            
            # Normalize if needed
            if self._normalization:
                window = (window - self._mean) / self._std
            
            # Convert to 2D pattern
            current_pattern = window.reshape(-1, 1)
        
        return np.array(predictions)
