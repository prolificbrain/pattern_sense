"""Tryte implementation for UNIFIED Consciousness Engine."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from .trit import Trit, TritState


@dataclass
class Tryte:
    """Represents a structured collection of trits with defined geometry.
    
    A Tryte is the minimal semantic unit in the UNIFIED system, composed of
    multiple trits arranged in a specific geometric configuration. Trytes can
    represent more complex information patterns and form the basis of knowledge
    structures.
    
    Attributes:
        trits: List of trits that compose the tryte
        geometry: Geometric configuration of the trits (e.g., 'linear', 'triangle')
        coherence: A measure of how tightly bound the trits are (0.0-1.0)
        height: Vertical dimension of the tryte's field representation
        width: Horizontal dimension of the tryte's field representation
    """
    
    trits: List[Trit]
    geometry: str = 'linear'
    coherence: float = 0.8
    height: int = 28  # Default dimensions for visual patterns
    width: int = 28
    _positions: Dict[int, Tuple[float, ...]] = field(init=False, repr=False, default_factory=dict)
    
    def __post_init__(self):
        """Initialize tryte and calculate positions of trits."""
        if not self.trits:
            raise ValueError("Tryte must contain at least one trit")
        
        # Generate positions based on geometry
        self._calculate_positions()
    
    def _calculate_positions(self) -> None:
        """Calculate the relative positions of trits based on geometry."""
        n_trits = len(self.trits)
        
        if self.geometry == 'linear':
            # Linear arrangement along x-axis
            for i in range(n_trits):
                self._positions[i] = (float(i - (n_trits - 1) / 2), 0.0)
                
        elif self.geometry == 'triangle':
            # Triangular arrangement in 2D
            if n_trits <= 3:
                # For 3 or fewer trits, place at vertices of equilateral triangle
                for i in range(min(n_trits, 3)):
                    angle = 2 * np.pi * i / 3
                    self._positions[i] = (np.cos(angle), np.sin(angle))
            else:
                # For more trits, distribute in rings
                self._positions[0] = (0.0, 0.0)  # Center trit
                remaining = n_trits - 1
                
                # Place remaining trits in concentric rings
                ring = 1
                placed = 1
                while placed < n_trits:
                    # Number of trits in this ring
                    ring_size = min(remaining, 3 * ring)
                    remaining -= ring_size
                    
                    # Place trits evenly around the ring
                    for i in range(ring_size):
                        angle = 2 * np.pi * i / ring_size
                        self._positions[placed] = (ring * np.cos(angle), ring * np.sin(angle))
                        placed += 1
                    ring += 1
                    
        elif self.geometry == 'tetrahedron':
            # Tetrahedral arrangement in 3D
            if n_trits <= 4:
                # For 4 or fewer trits, place at vertices of tetrahedron
                tetra_vertices = [
                    (1, 1, 1),
                    (1, -1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1)
                ]
                for i in range(min(n_trits, 4)):
                    self._positions[i] = tetra_vertices[i]
            else:
                # For more trits, distribute in shells
                # Start with tetrahedron vertices
                tetra_vertices = [
                    (1, 1, 1),
                    (1, -1, -1),
                    (-1, 1, -1),
                    (-1, -1, 1)
                ]
                for i in range(min(n_trits, 4)):
                    self._positions[i] = tetra_vertices[i]
                
                # Place remaining trits on faces and edges
                if n_trits > 4:
                    remaining = n_trits - 4
                    placed = 4
                    
                    # Place on edges first
                    edges = [
                        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
                    ]
                    for edge_start, edge_end in edges:
                        if placed >= n_trits:
                            break
                        # Midpoint of edge
                        start_pos = tetra_vertices[edge_start]
                        end_pos = tetra_vertices[edge_end]
                        mid_pos = tuple((s + e) / 2 for s, e in zip(start_pos, end_pos))
                        self._positions[placed] = mid_pos
                        placed += 1
        else:
            # Default to linear if geometry not recognized
            for i in range(n_trits):
                self._positions[i] = (float(i), 0.0)
    
    def get_trit_positions(self) -> Dict[int, Tuple[float, ...]]:
        """Get the positions of all trits in the tryte.
        
        Returns:
            Dictionary mapping trit indices to their positions
        """
        return self._positions
    
    def to_wave_pattern(self, shape: Tuple[int, ...], 
                       center: Optional[Tuple[int, ...]] = None,
                       scale: float = 5.0) -> np.ndarray:
        """Convert the tryte to a wave pattern that can be injected into a data field.
        
        Args:
            shape: Shape of the target field
            center: Center coordinates for the pattern (default: middle of field)
            scale: Scaling factor for the pattern size
            
        Returns:
            NumPy array containing the field pattern
        """
        if center is None:
            # Default to center of field
            center = tuple(s // 2 for s in shape)
        
        # Initialize pattern
        pattern = np.zeros(shape)
        
        # Add contribution from each trit
        for i, trit in enumerate(self.trits):
            # Get relative position of this trit
            rel_pos = self._positions[i]
            
            # Calculate absolute position in the field
            trit_center = tuple(int(c + scale * p) for c, p in zip(center, rel_pos[:len(center)]))
            
            # Generate trit pattern
            trit_pattern = trit.to_field_pattern(shape, trit_center)
            
            # Add to overall pattern, modulated by coherence
            # Higher coherence = stronger coupling between trits
            pattern += trit_pattern * (self.coherence + (1 - self.coherence) / len(self.trits))
        
        return pattern
    
    def to_wave_pattern(self, shape, center=None):
        """Convert the tryte to a wave pattern centered at a specific position.
        
        This is useful for injecting tryte patterns into data fields.
        
        Args:
            shape: Shape of the target field
            center: Center position for the pattern
            
        Returns:
            pattern: NumPy array with the wave pattern
        """
        import numpy as np
        pattern = np.zeros(shape)
        
        # Use center of field if not specified
        if center is None:
            center = tuple(s // 2 for s in shape)
            
        # Maximum radius for the pattern (1/3 of the smallest dimension)
        max_radius = min(shape) // 3
        
        # Create coordinate grid
        grid_coords = []
        for dim in range(len(shape)):
            coords = np.linspace(-1, 1, shape[dim])
            grid_coords.append(coords)
        mesh_grid = np.meshgrid(*grid_coords, indexing='ij')
        
        # For each trit, add a gaussian wave centered at its position
        for trit in self.trits:
            # Calculate relative position from center
            rel_pos = []
            for dim, pos in enumerate(trit.orientation):
                if dim < len(center):
                    rel_pos.append((pos - center[dim]) / max_radius)
                else:
                    rel_pos.append(0)
                    
            # Create wave centered at the trit's position
            wave = np.zeros(shape)
            for dim in range(len(shape)):
                if dim < len(rel_pos):
                    distance = (mesh_grid[dim] - rel_pos[dim])**2
                    wave += distance
                    
            # Create gaussian with amplitude based on trit value
            amplitude = trit.value
            sigma = 0.3  # Width of gaussian
            wave = amplitude * np.exp(-wave / (2 * sigma**2))
            
            # Add to overall pattern
            pattern += wave
            
        # Normalize pattern
        pattern_max = np.max(np.abs(pattern))
        if pattern_max > 0:
            pattern /= pattern_max
            
        return pattern
    
    def interfere_with(self, other: 'Tryte') -> 'Tryte':
        """Compute the interference pattern of this tryte with another tryte.
        
        Args:
            other: Another tryte to interfere with
            
        Returns:
            A new tryte resulting from the interference
        """
        # Pair each trit in this tryte with the nearest trit in the other tryte
        # and compute their interference
        new_trits = []
        
        # If trytes have the same number of trits, pair them directly
        if len(self.trits) == len(other.trits):
            for self_trit, other_trit in zip(self.trits, other.trits):
                new_trits.append(self_trit.interfere_with(other_trit))
        
        # If trytes have different numbers of trits, find nearest pairs
        else:
            # For each trit in this tryte
            for i, self_trit in enumerate(self.trits):
                self_pos = self._positions[i]
                
                # Find the nearest trit in the other tryte
                min_dist = float('inf')
                nearest_trit = None
                
                for j, other_trit in enumerate(other.trits):
                    other_pos = other._positions[j]
                    
                    # Calculate distance (only for common dimensions)
                    common_dims = min(len(self_pos), len(other_pos))
                    squared_dist = sum((self_pos[d] - other_pos[d])**2 for d in range(common_dims))
                    
                    if squared_dist < min_dist:
                        min_dist = squared_dist
                        nearest_trit = other_trit
                
                # Interfere with the nearest trit
                if nearest_trit is not None:
                    new_trits.append(self_trit.interfere_with(nearest_trit))
                else:
                    # If no trit found (shouldn't happen), keep original
                    new_trits.append(self_trit)
            
            # Add any remaining trits from the larger tryte
            if len(other.trits) > len(self.trits):
                for i in range(len(self.trits), len(other.trits)):
                    new_trits.append(other.trits[i])
        
        # Combine coherence (weighted average)
        total_trits = len(self.trits) + len(other.trits)
        if total_trits > 0:
            new_coherence = (self.coherence * len(self.trits) + 
                           other.coherence * len(other.trits)) / total_trits
        else:
            new_coherence = self.coherence
        
        # Return new tryte with same geometry but new trits
        return Tryte(trits=new_trits, geometry=self.geometry, coherence=new_coherence)
    
    def morph(self, morph_factor: float = 0.1) -> 'Tryte':
        """Apply a random morphing to the tryte, changing its internal structure.
        
        Args:
            morph_factor: Strength of the morphing effect (0.0-1.0)
            
        Returns:
            A new, morphed tryte
        """
        # Create new trits with slight variations
        new_trits = []
        
        for trit in self.trits:
            # Randomly decide what to modify
            morph_type = np.random.choice(['state', 'energy', 'phase', 'orientation'])
            
            if morph_type == 'state' and np.random.random() < morph_factor:
                # Randomly change state
                new_state = np.random.choice(list(TritState))
                new_trit = Trit(state=new_state, energy=trit.energy,
                              phase=trit.phase, orientation=trit.orientation)
            
            elif morph_type == 'energy':
                # Modify energy
                energy_change = 1.0 + morph_factor * (2 * np.random.random() - 1)
                new_trit = trit.modulate(energy_change)
            
            elif morph_type == 'phase':
                # Modify phase
                phase_change = morph_factor * np.pi * (2 * np.random.random() - 1)
                new_trit = trit.rotate(phase_change)
            
            elif morph_type == 'orientation':
                # Modify orientation slightly
                new_orientation = tuple(o + morph_factor * (2 * np.random.random() - 1) 
                                      for o in trit.orientation)
                new_trit = trit.reorient(new_orientation)
            
            else:
                # No change
                new_trit = trit
            
            new_trits.append(new_trit)
        
        # Potentially modify coherence
        new_coherence = max(0.0, min(1.0, self.coherence + 
                                   morph_factor * (2 * np.random.random() - 1)))
        
        # Return new tryte with same geometry but morphed properties
        return Tryte(trits=new_trits, geometry=self.geometry, coherence=new_coherence)
    
    def get_energy(self) -> float:
        """Calculate the total energy of the tryte.
        
        Returns:
            Sum of energies of all trits, modified by coherence
        """
        # Base energy is sum of trit energies
        base_energy = sum(trit.energy for trit in self.trits)
        
        # Coherence factor: high coherence produces emergent energy beyond sum
        coherence_factor = 1.0 + 0.2 * self.coherence  # Up to 20% bonus for high coherence
        
        return base_energy * coherence_factor
    
    def get_structure_metric(self) -> float:
        """Calculate a metric representing the internal structure of the tryte.
        
        Higher values indicate more organized/coherent structure.
        
        Returns:
            Structure metric value
        """
        # Base: coherence value
        structure = self.coherence
        
        # Add factor for trit alignment (similar phases increase structure)
        if len(self.trits) > 1:
            phases = [trit.phase for trit in self.trits]
            # Calculate standard deviation of phases
            phase_std = np.std(phases)
            # Convert to a 0-1 metric (lower std = higher structure)
            phase_alignment = np.exp(-phase_std / np.pi)
            structure += 0.3 * phase_alignment
        
        # Add factor for trit state consistency
        if len(self.trits) > 1:
            # Count occurrences of each state
            state_counts = {state: 0 for state in TritState}
            for trit in self.trits:
                state_counts[trit.state] += 1
            
            # Calculate normalized entropy of state distribution
            total = len(self.trits)
            entropy = 0.0
            for count in state_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log(p)
            
            # Normalize by maximum entropy (log of number of possible states)
            max_entropy = np.log(len(TritState))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
                # Convert to a 0-1 metric (lower entropy = higher structure)
                state_consistency = 1.0 - normalized_entropy
                structure += 0.3 * state_consistency
        
        # Normalize final result to 0-1 range
        return min(1.0, structure)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the tryte to a dictionary representation.
        
        Returns:
            Dictionary containing the tryte's properties
        """
        return {
            'trits': [{
                'state': trit.state.name,
                'value': trit.value,
                'energy': trit.energy,
                'phase': trit.phase,
                'orientation': trit.orientation
            } for trit in self.trits],
            'geometry': self.geometry,
            'coherence': self.coherence,
            'positions': {i: pos for i, pos in self._positions.items()},
            'metrics': {
                'energy': self.get_energy(),
                'structure': self.get_structure_metric()
            }
        }
    
    @classmethod
    def from_pattern(cls, pattern: List[int], energy: float = 1.0, 
                    geometry: str = 'linear', coherence: float = 0.8) -> 'Tryte':
        """Create a tryte from a pattern of integer trit values.
        
        Args:
            pattern: List of integer values (-1, 0, 1)
            energy: Energy for all trits
            geometry: Geometric configuration
            coherence: Coherence factor
            
        Returns:
            Newly created tryte
        """
        trits = []
        
        for value in pattern:
            if value == -1:
                state = TritState.NEGATIVE
            elif value == 0:
                state = TritState.NEUTRAL
            elif value == 1:
                state = TritState.POSITIVE
            else:
                raise ValueError(f"Invalid trit value: {value}. Must be -1, 0, or 1.")
            
            # Create trit with random phase
            phase = 2 * np.pi * np.random.random()
            trits.append(Trit(state=state, energy=energy, phase=phase))
        
        return cls(trits=trits, geometry=geometry, coherence=coherence)
    
    def copy(self) -> 'Tryte':
        """Create a copy of this tryte.
        
        Returns:
            A new Tryte with the same properties
        """
        return Tryte(
            trits=[trit.copy() for trit in self.trits],
            geometry=self.geometry,
            coherence=self.coherence
        )
    
    def scale_energy(self, scale_factor: float) -> None:
        """Scale the energy of all trits in this tryte.
        
        Args:
            scale_factor: Factor to scale energy by (0.0-1.0)
        """
        for trit in self.trits:
            trit.energy *= scale_factor
    
    def get_average_energy(self) -> float:
        """Calculate the average energy of all trits in this tryte.
        
        Returns:
            Average energy value
        """
        if not self.trits:
            return 0.0
        return sum(trit.energy for trit in self.trits) / len(self.trits)
