"""Trit implementation for UNIFIED Consciousness Engine."""

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Optional


class TritState(Enum):
    """Represents the three possible states of a trit."""
    NEGATIVE = -1  # Negative state (-1)
    NEUTRAL = 0    # Neutral state (0)
    POSITIVE = 1   # Positive state (+1)


@dataclass
class Trit:
    """Represents a fundamental trit (trinary digit) as used in the UNIFIED consciousness engine.
    
    A trit is the most fundamental unit of information in the UNIFIED system, 
    representing a 3-state logic primitive (+1, 0, -1). Each trit has not just a
    logical value but also physical properties that influence how it propagates
    and interacts in the holographic field.
    
    Attributes:
        state: The logical state of the trit (positive, neutral, or negative)
        energy: Energy intensity of the trit pulse
        phase: Phase angle in the field (radians)
        orientation: Directional vector in field space
    """
    
    state: TritState
    energy: float = 1.0
    phase: float = 0.0  # in radians
    orientation: Optional[Tuple[float, ...]] = None
    
    def __post_init__(self):
        """Initialize trit with default orientation if none provided."""
        if self.orientation is None:
            # Default to unit vector in positive x direction
            self.orientation = (1.0, 0.0)
    
    @property
    def value(self) -> int:
        """Get the integer value of the trit state."""
        return self.state.value
    
    def flip(self) -> 'Trit':
        """Flip the trit state (positive → negative, negative → positive, neutral stays).
        
        Returns:
            A new trit with flipped state
        """
        if self.state == TritState.POSITIVE:
            new_state = TritState.NEGATIVE
        elif self.state == TritState.NEGATIVE:
            new_state = TritState.POSITIVE
        else:
            new_state = TritState.NEUTRAL
        
        return Trit(state=new_state, energy=self.energy, 
                  phase=self.phase, orientation=self.orientation)
    
    def rotate(self, angle: float) -> 'Trit':
        """Rotate the trit's phase by the specified angle.
        
        Args:
            angle: Rotation angle in radians
            
        Returns:
            A new trit with rotated phase
        """
        new_phase = (self.phase + angle) % (2 * np.pi)
        return Trit(state=self.state, energy=self.energy, 
                  phase=new_phase, orientation=self.orientation)
    
    def modulate(self, energy_factor: float) -> 'Trit':
        """Modulate the trit's energy by the specified factor.
        
        Args:
            energy_factor: Factor to multiply energy by
            
        Returns:
            A new trit with modulated energy
        """
        new_energy = self.energy * energy_factor
        return Trit(state=self.state, energy=new_energy, 
                  phase=self.phase, orientation=self.orientation)
    
    def reorient(self, new_orientation: Tuple[float, ...]) -> 'Trit':
        """Change the trit's orientation vector.
        
        Args:
            new_orientation: New orientation vector
            
        Returns:
            A new trit with new orientation
        """
        return Trit(state=self.state, energy=self.energy, 
                  phase=self.phase, orientation=new_orientation)
    
    def to_wave_component(self, position: Tuple[float, ...], time: float = 0.0) -> float:
        """Convert the trit to a wave amplitude at a specific position and time.
        
        This allows trits to be embedded in a continuous field as wave-like
        perturbations with physical characteristics.
        
        Args:
            position: Spatial coordinates to evaluate the wave at
            time: Time to evaluate the wave at
            
        Returns:
            Wave amplitude value
        """
        # Calculate dot product of position and orientation
        if len(position) != len(self.orientation):
            raise ValueError(f"Position dimensions {len(position)} don't match "
                           f"orientation dimensions {len(self.orientation)}")
        
        # Calculate distance along orientation
        dot_product = sum(p * o for p, o in zip(position, self.orientation))
        
        # Calculate wave phase including spatial position, time, and trit phase
        wave_phase = dot_product - time + self.phase
        
        # Generate sinusoidal wave modulated by energy and state
        amplitude = self.energy * np.sin(wave_phase) * self.value
        
        return amplitude
    
    def interfere_with(self, other: 'Trit') -> 'Trit':
        """Compute the interference pattern of this trit with another trit.
        
        Args:
            other: Another trit to interfere with
            
        Returns:
            A new trit resulting from the interference
        """
        # Phase difference determines constructive vs destructive interference
        phase_diff = abs(self.phase - other.phase) % (2 * np.pi)
        constructive = phase_diff < np.pi / 2 or phase_diff > 3 * np.pi / 2
        
        # Calculate new state based on interference pattern
        if constructive:
            # Constructive interference: states reinforce
            if self.state == other.state:
                new_state = self.state
            elif self.state == TritState.NEUTRAL:
                new_state = other.state
            elif other.state == TritState.NEUTRAL:
                new_state = self.state
            else:
                # Opposing states with constructive phase: strongest wins
                if self.energy > other.energy:
                    new_state = self.state
                else:
                    new_state = other.state
        else:
            # Destructive interference
            if self.state == other.state:
                # Same states with destructive phase: cancel toward neutral
                if abs(self.energy - other.energy) < 0.1:
                    new_state = TritState.NEUTRAL
                elif self.energy > other.energy:
                    new_state = self.state
                else:
                    new_state = other.state
            elif self.state == TritState.NEUTRAL or other.state == TritState.NEUTRAL:
                # Neutral with anything destructively: weaker effect
                new_state = TritState.NEUTRAL
            else:
                # Opposing states with destructive phase: enhancement
                if self.value > other.value:  # Positive dominates
                    new_state = TritState.POSITIVE
                else:
                    new_state = TritState.NEGATIVE
        
        # Calculate new energy
        if constructive and self.state == other.state:
            # Constructive interference with same state: energy adds
            new_energy = self.energy + other.energy
        elif not constructive and self.state != other.state and self.state != TritState.NEUTRAL and other.state != TritState.NEUTRAL:
            # Destructive interference with opposite states: energy difference
            new_energy = abs(self.energy - other.energy)
        else:
            # Other cases: weighted average
            total_energy = self.energy + other.energy
            if total_energy > 0:
                new_energy = (self.energy * self.value + other.energy * other.value) / total_energy
            else:
                new_energy = 0.0
        
        # Calculate new phase (weighted average)
        total_energy = self.energy + other.energy
        if total_energy > 0:
            x_component = (self.energy * np.cos(self.phase) + other.energy * np.cos(other.phase)) / total_energy
            y_component = (self.energy * np.sin(self.phase) + other.energy * np.sin(other.phase)) / total_energy
            new_phase = np.arctan2(y_component, x_component) % (2 * np.pi)
        else:
            new_phase = 0.0
        
        # Calculate new orientation (weighted average)
        if total_energy > 0:
            new_orientation = tuple((self.energy * so + other.energy * oo) / total_energy 
                                  for so, oo in zip(self.orientation, other.orientation))
        else:
            new_orientation = self.orientation
        
        return Trit(state=new_state, energy=new_energy, 
                  phase=new_phase, orientation=new_orientation)
    
    def to_field_pattern(self, shape: Tuple[int, ...], 
                        center: Optional[Tuple[int, ...]] = None, 
                        width: float = 1.0) -> np.ndarray:
        """Convert the trit to a field pattern that can be injected into a data field.
        
        Args:
            shape: Shape of the target field
            center: Center coordinates for the pattern (default: middle of field)
            width: Width of the gaussian pulse
            
        Returns:
            NumPy array containing the field pattern
        """
        if center is None:
            # Default to center of field
            center = tuple(s // 2 for s in shape)
        
        # Create coordinate grid
        indices = np.indices(shape)
        
        # Calculate squared distances from center
        sq_distances = np.zeros(shape)
        for dim, loc in enumerate(center):
            sq_distances += (indices[dim] - loc) ** 2
        
        # Create gaussian pulse modulated by trit properties
        gaussian = np.exp(-sq_distances / (2 * width ** 2))
        
        # Apply trit state and energy
        pattern = self.value * self.energy * gaussian
        
        # Apply phase modulation (creates wave-like structure)
        phase_grid = np.zeros(shape)
        for dim, orientation in enumerate(self.orientation[:len(shape)]):
            phase_grid += orientation * (indices[dim] - center[dim])
        
        pattern *= np.cos(phase_grid + self.phase)
        
        return pattern
    
    def copy(self) -> 'Trit':
        """Create a copy of this trit.
        
        Returns:
            A new Trit with the same properties
        """
        return Trit(
            state=self.state,
            energy=self.energy,
            phase=self.phase,
            orientation=self.orientation
        )
