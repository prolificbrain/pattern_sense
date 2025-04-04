"""Minimal Thinking Unit implementation for UNIFIED Consciousness Engine."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from ..trits.trit import Trit, TritState
from ..trits.tryte import Tryte


@dataclass
class MinimalThinkingUnit:
    """Represents a Minimal Thinking Unit (MTU) in the UNIFIED Consciousness Engine.
    
    The MTU is a field-local structure that processes incoming trits/trytes,
    maintains an internal state, and produces output based on field interactions.
    It is the fundamental computational element in the system.
    
    Attributes:
        position: Spatial position of the MTU in the field
        state_field: Internal memory field (resonance basin)
        input_buffer: Buffer for incoming trytes
        output_buffer: Buffer for outgoing trytes
        learning_rate: How quickly the MTU adapts to new inputs
        pattern_memory_enabled: Whether to enable pattern memory
        attractor_dynamics_enabled: Whether to enable attractor dynamics
    """
    
    position: Tuple[int, ...]
    dimensions: int = 2
    state_field_size: int = 5
    learning_rate: float = 0.2
    pattern_memory_enabled: bool = True
    attractor_dynamics_enabled: bool = True
    _state_field: np.ndarray = field(init=False, repr=False)
    _input_buffer: List[Tryte] = field(default_factory=list, init=False, repr=False)
    _output_buffer: List[Tryte] = field(default=None, init=False, repr=False)
    _local_field_state: np.ndarray = field(init=False, repr=False)
    _history: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _pattern_memory: Optional[Any] = field(default=None, init=False, repr=False)
    _attractor_dynamics: Optional[Any] = field(default=None, init=False, repr=False)
    _known_patterns: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _pattern_strengths: List[float] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the MTU's internal fields."""
        # Create state field with small random values
        self._state_field = np.random.uniform(-0.1, 0.1, (self.state_field_size,) * self.dimensions)
        
        # Initialize local field state (represents local curvature of substrate)
        self._local_field_state = np.zeros((self.state_field_size,) * self.dimensions)
        
        # Initialize learning components if enabled
        if self.pattern_memory_enabled:
            # Import here to avoid circular imports
            from .learning import PatternMemory
            self._pattern_memory = PatternMemory()
            
        if self.attractor_dynamics_enabled:
            # Import here to avoid circular imports
            from .learning import AttractorDynamics
            self._attractor_dynamics = AttractorDynamics()
        
        # Record initial state
        self._record_state()
    
    def _record_state(self):
        """Record current state in history."""
        state_snapshot = {
            'time': len(self._history),
            'state_energy': np.sum(self._state_field**2),
            'input_count': len(self._input_buffer),
            'output_count': len(self._output_buffer) if self._output_buffer else 0,
            'state_entropy': self._calculate_state_entropy(),
            'pattern_count': len(self._known_patterns) if self._pattern_memory else 0
        }
        
        # Limit history size
        if len(self._history) >= 100:
            self._history.pop(0)
        
        self._history.append(state_snapshot)
    
    def _calculate_state_entropy(self) -> float:
        """Calculate the entropy of the state field.
        
        Returns:
            Entropy value (higher = more disordered)
        """
        # Normalize the state field
        abs_field = np.abs(self._state_field)
        total = np.sum(abs_field)
        
        if total > 0:
            normalized_field = abs_field / total
            # Remove zeros to avoid log(0)
            nonzero_mask = normalized_field > 0
            entropy = -np.sum(normalized_field[nonzero_mask] * 
                             np.log(normalized_field[nonzero_mask]))
            max_entropy = np.log(self._state_field.size)
            return entropy / max_entropy if max_entropy > 0 else 0
        else:
            return 0
    
    def receive_input(self, tryte: Tryte) -> None:
        """Receive an input tryte into the buffer.
        
        Args:
            tryte: Input tryte to process
        """
        self._input_buffer.append(tryte)
    
    def update_local_field(self, field_state: np.ndarray) -> None:
        """Update the local field state around the MTU.
        
        Args:
            field_state: New local field state
        """
        # Ensure dimensions match
        if field_state.shape != self._local_field_state.shape:
            raise ValueError(f"Expected field state of shape {self._local_field_state.shape}, "
                           f"got {field_state.shape}")
        
        self._local_field_state = field_state
    
    def process(self) -> Optional[Tryte]:
        """Process inputs and generate output based on state.
        
        Returns:
            Output tryte if one was generated, else None
        """
        # Ensure output buffer is initialized
        if self._output_buffer is None:
            self._output_buffer = []
            
        # If no inputs, just update state and return None
        if not self._input_buffer:
            # Apply state decay (memory fades over time)
            self._state_field *= 0.95
            
            # Apply attractor dynamics even without input
            if self.attractor_dynamics_enabled and self._attractor_dynamics and self._known_patterns:
                self._state_field = self._attractor_dynamics.apply_attractors(
                    self._state_field, 
                    self._known_patterns, 
                    self._pattern_strengths
                )
                
            self._record_state()
            return None
        
        # Get the next input tryte
        input_tryte = self._input_buffer.pop(0)
        
        # Convert input tryte to field pattern
        input_pattern = self._tryte_to_field_pattern(input_tryte)
        
        # Modulate input by local field state (simulates curvature effects)
        modulated_input = input_pattern * (1 + 0.1 * self._local_field_state)
        
        # Integrate input with state field
        integrated_field = self._state_field + self.learning_rate * modulated_input
        
        # Apply attractor dynamics if enabled
        if self.attractor_dynamics_enabled and self._attractor_dynamics and self._known_patterns:
            integrated_field = self._attractor_dynamics.apply_attractors(
                integrated_field, 
                self._known_patterns, 
                self._pattern_strengths
            )
        
        # Apply nonlinear dynamics (tanh squashing function)
        # This creates attractor dynamics in the state space
        self._state_field = np.tanh(integrated_field)
        
        # Learn pattern if enabled
        if self.pattern_memory_enabled and self._pattern_memory:
            # Try to recognize pattern
            recognized, pattern_idx, similarity = self._pattern_memory.recognize_pattern(self._state_field)
            
            if recognized:
                # Pattern was recognized, update attractors
                attractor = self._pattern_memory.get_attractor_pattern(pattern_idx)
                if attractor is not None:
                    # Pull state toward attractor
                    self._state_field = 0.7 * self._state_field + 0.3 * attractor
            else:
                # Learn new pattern if state is sufficiently structured
                energy = np.sum(self._state_field**2)
                structure = 1.0 - self._calculate_state_entropy()
                
                if energy > 0.5 and structure > 0.4:
                    pattern_idx = self._pattern_memory.learn_pattern(self._state_field)
                    
                    # Update known patterns and strengths
                    self._known_patterns = []
                    self._pattern_strengths = []
                    
                    for i in range(len(self._pattern_memory._patterns)):
                        pattern = self._pattern_memory.get_attractor_pattern(i)
                        if pattern is not None:
                            self._known_patterns.append(pattern)
                            self._pattern_strengths.append(self._pattern_memory._pattern_strengths[i])
        
        # Check if the state field has formed a strong pattern
        # This is how the MTU decides whether to output
        energy = np.sum(self._state_field**2)
        structure = 1.0 - self._calculate_state_entropy()  # Structure is inverse of entropy
        
        # If strong, structured state formed, generate output
        if energy > 1.0 and structure > 0.6:
            output_tryte = self._field_pattern_to_tryte()
            self._output_buffer.append(output_tryte)
            
            # Reset state field partially (like a neuron firing)
            self._state_field *= 0.5
            
            self._record_state()
            return output_tryte
        
        self._record_state()
        return None
    
    def _tryte_to_field_pattern(self, tryte: Tryte) -> np.ndarray:
        """Convert a tryte to a field pattern.
        
        Args:
            tryte: Input tryte
            
        Returns:
            Field pattern representing the tryte
        """
        # Initialize pattern
        pattern = np.zeros(self._state_field.shape)
        
        # Get positions of trits in the tryte
        positions = tryte.get_trit_positions()
        
        # Map tryte's relative positions to the state field
        field_center = tuple(s // 2 for s in self._state_field.shape)
        scale = min(self._state_field.shape) / 4  # Scale relative to field size
        
        # Add contribution from each trit
        for i, trit in enumerate(tryte.trits):
            # Get relative position of this trit
            rel_pos = positions[i]
            
            # Calculate position in the field
            trit_pos = tuple(int(c + scale * p) for c, p in zip(field_center, rel_pos[:self.dimensions]))
            
            # Ensure position is within bounds
            trit_pos = tuple(max(0, min(s-1, p)) for p, s in zip(trit_pos, self._state_field.shape))
            
            # Add trit influence to pattern
            # Create a small gaussian around the trit position
            indices = np.indices(self._state_field.shape)
            sq_distances = np.zeros(self._state_field.shape)
            
            for dim, pos in enumerate(trit_pos):
                sq_distances += (indices[dim] - pos) ** 2
            
            # Width of gaussian proportional to coherence
            width = 1.0 + (1.0 - tryte.coherence) * 2
            gaussian = np.exp(-sq_distances / (2 * width ** 2))
            
            # Add trit influence to pattern
            pattern += trit.value * trit.energy * gaussian
        
        return pattern
    
    def _field_pattern_to_tryte(self) -> Tryte:
        """Convert the current state field to an output tryte.
        
        Returns:
            Output tryte derived from state field
        """
        # Find local maxima and minima in the field
        from scipy import ndimage
        max_coords = ndimage.maximum_filter(self._state_field, size=3) == self._state_field
        min_coords = ndimage.minimum_filter(self._state_field, size=3) == self._state_field
        
        # Threshold to ignore small maxima/minima
        threshold = 0.3
        max_coords &= self._state_field > threshold
        min_coords &= self._state_field < -threshold
        
        # Get coordinates of maxima and minima
        max_indices = np.where(max_coords)
        min_indices = np.where(min_coords)
        
        # Create trits from these points
        trits = []
        
        # Create positive trits from maxima
        for i in range(min(len(max_indices[0]), 3)):  # Limit to 3 positive trits
            # Extract coordinates for this maximum
            coords = tuple(index[i] for index in max_indices)
            
            # Calculate relative position (centered at middle of field)
            field_center = tuple(s // 2 for s in self._state_field.shape)
            rel_pos = tuple((c - fc) / (self._state_field.shape[d] / 4) 
                          for d, (c, fc) in enumerate(zip(coords, field_center)))
            
            # Get field value and create trit
            value = self._state_field[coords]
            energy = min(1.0, abs(value))
            
            # Create a positive trit with this energy and position
            trit = Trit(state=TritState.POSITIVE, energy=energy, position=rel_pos)
            trits.append(trit)
        
        # Create negative trits from minima
        for i in range(min(len(min_indices[0]), 3)):  # Limit to 3 negative trits
            # Extract coordinates for this minimum
            coords = tuple(index[i] for index in min_indices)
            
            # Calculate relative position
            field_center = tuple(s // 2 for s in self._state_field.shape)
            rel_pos = tuple((c - fc) / (self._state_field.shape[d] / 4) 
                          for d, (c, fc) in enumerate(zip(coords, field_center)))
            
            # Get field value and create trit
            value = self._state_field[coords]
            energy = min(1.0, abs(value))
            
            # Create a negative trit with this energy and position
            trit = Trit(state=TritState.NEGATIVE, energy=energy, position=rel_pos)
            trits.append(trit)
        
        # If we didn't find enough extrema, add some neutral trits
        while len(trits) < 3:
            # Create a neutral trit with random position
            rel_pos = tuple(np.random.uniform(-1, 1) for _ in range(3))
            trit = Trit(state=TritState.NEUTRAL, energy=0.1, position=rel_pos)
            trits.append(trit)
        
        # Calculate coherence based on state field entropy
        coherence = 1.0 - self._calculate_state_entropy()
        
        # Create a tryte from these trits
        return Tryte(trits=trits, coherence=coherence)
    
    def get_output(self) -> Optional[Tryte]:
        """Get the next output tryte from the buffer if available.
        
        Returns:
            Next output tryte or None if buffer is empty
        """
        if self._output_buffer:
            return self._output_buffer.pop(0)
        return None
    
    def clear_buffers(self):
        """Clear all input/output buffers.
        
        This is useful when resetting the MTU between processing different inputs.
        """
        self._input_buffer = []
        self._output_buffer = None
        self._last_output = None
    
    def visualize(self, ax=None):
        """Visualize the MTU's state field.
        
        Args:
            ax: Matplotlib axis for plotting
            
        Returns:
            Matplotlib axis with the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if self.dimensions == 1:
            ax.plot(self._state_field)
            ax.set_title('MTU State Field')
            ax.set_xlabel('Position')
            ax.set_ylabel('Value')
        elif self.dimensions == 2:
            im = ax.imshow(self._state_field, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
            ax.set_title('MTU State Field')
            plt.colorbar(im, ax=ax)
        else:
            # For higher dimensions, just show a 2D slice
            slice_indices = tuple(s//2 for s in self._state_field.shape[2:]) 
            ax.imshow(self._state_field[:, :, *slice_indices], cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
            ax.set_title('MTU State Field (2D slice)')
        
        return ax
    
    def plot_history(self, figsize=(10, 6)):
        """Plot the MTU's history of state variables.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with plots
        """
        import matplotlib.pyplot as plt
        
        if not self._history:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Extract history data
        times = [h['time'] for h in self._history]
        energies = [h['state_energy'] for h in self._history]
        entropies = [h['state_entropy'] for h in self._history]
        inputs = [h['input_count'] for h in self._history]
        outputs = [h['output_count'] for h in self._history]
        
        # Plot state energy
        axes[0].plot(times, energies)
        axes[0].set_title('State Energy')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Energy')
        
        # Plot state entropy
        axes[1].plot(times, entropies)
        axes[1].set_title('State Entropy')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Entropy')
        
        # Plot input/output counts
        axes[2].plot(times, inputs, label='Inputs')
        axes[2].plot(times, outputs, label='Outputs')
        axes[2].set_title('Input/Output Activity')
        axes[2].set_xlabel('Time Steps')
        axes[2].set_ylabel('Count')
        axes[2].legend()
        
        # Plot pattern metrics if pattern memory is enabled
        if self.pattern_memory_enabled and self._pattern_memory:
            pattern_counts = [h.get('pattern_count', 0) for h in self._history]
            axes[3].plot(times, pattern_counts)
            axes[3].set_title('Known Patterns')
            axes[3].set_xlabel('Time Steps')
            axes[3].set_ylabel('Count')
        else:
            axes[3].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_pattern_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern memory.
        
        Returns:
            Dictionary with pattern memory statistics
        """
        if not self.pattern_memory_enabled or self._pattern_memory is None:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'pattern_count': len(self._pattern_memory._patterns),
            'avg_strength': np.mean(self._pattern_memory._pattern_strengths) if self._pattern_memory._pattern_strengths else 0,
            'total_activations': sum(self._pattern_memory._pattern_activations) if self._pattern_memory._pattern_activations else 0
        }
