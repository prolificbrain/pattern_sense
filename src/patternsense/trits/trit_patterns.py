"""Pattern creation utilities for the UNIFIED Consciousness Engine."""

import numpy as np
from typing import Tuple, Optional, List

from .trit import Trit, TritState
from .tryte import Tryte


def create_pulse_pattern(shape: Tuple[int, ...], 
                        center: Optional[Tuple[int, ...]] = None,
                        value: int = 1,
                        energy: float = 1.0,
                        width: float = 1.0) -> np.ndarray:
    """Create a single pulse pattern that can be injected into a data field.
    
    Args:
        shape: Shape of the target field
        center: Center coordinates for the pattern (default: middle of field)
        value: Trit value (-1, 0, or 1)
        energy: Energy of the pulse
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
    
    # Apply value and energy
    pattern = value * energy * gaussian
    
    return pattern


def create_wave_pattern(shape: Tuple[int, ...],
                       center: Optional[Tuple[int, ...]] = None,
                       value: int = 1,
                       energy: float = 1.0,
                       frequency: float = 1.0,
                       orientation: Optional[Tuple[float, ...]] = None) -> np.ndarray:
    """Create a wave pattern that can be injected into a data field.
    
    Args:
        shape: Shape of the target field
        center: Center coordinates for the pattern (default: middle of field)
        value: Trit value (-1, 0, or 1)
        energy: Energy of the wave
        frequency: Spatial frequency of the wave
        orientation: Direction vector of the wave (default: x-direction)
        
    Returns:
        NumPy array containing the field pattern
    """
    if center is None:
        # Default to center of field
        center = tuple(s // 2 for s in shape)
    
    if orientation is None:
        # Default to unit vector in x direction
        orientation = tuple(1.0 if i == 0 else 0.0 for i in range(len(shape)))
    
    # Normalize orientation vector
    norm = np.sqrt(sum(o**2 for o in orientation))
    if norm > 0:
        orientation = tuple(o / norm for o in orientation)
    
    # Create coordinate grid
    indices = np.indices(shape)
    
    # Calculate phase grid (dot product of position and orientation)
    phase_grid = np.zeros(shape)
    for dim, ori in enumerate(orientation[:len(shape)]):
        phase_grid += ori * (indices[dim] - center[dim])
    
    # Create wave pattern
    wave = np.cos(frequency * 2 * np.pi * phase_grid)
    
    # Apply gaussian envelope to localize the wave
    sq_distances = np.zeros(shape)
    for dim, loc in enumerate(center):
        sq_distances += (indices[dim] - loc) ** 2
    
    envelope = np.exp(-sq_distances / (2 * (shape[0] / 4) ** 2))
    
    # Apply value, energy, and envelope
    pattern = value * energy * wave * envelope
    
    return pattern


def create_tryte_pattern(shape: Tuple[int, ...],
                        center: Optional[Tuple[int, ...]] = None,
                        pattern: List[int] = [1, 0, -1],
                        energy: float = 1.0,
                        geometry: str = 'triangle',
                        coherence: float = 0.8,
                        scale: float = 5.0) -> np.ndarray:
    """Create a tryte pattern that can be injected into a data field.
    
    Args:
        shape: Shape of the target field
        center: Center coordinates for the pattern (default: middle of field)
        pattern: List of trit values (-1, 0, 1) defining the tryte
        energy: Energy of the tryte
        geometry: Geometric arrangement ('linear', 'triangle', 'tetrahedron')
        coherence: Coherence factor of the tryte
        scale: Scaling factor for the pattern size
        
    Returns:
        NumPy array containing the field pattern
    """
    # Create a tryte from the pattern
    tryte = Tryte.from_pattern(pattern, energy, geometry, coherence)
    
    # Convert to field pattern
    return tryte.to_wave_pattern(shape, center, scale)


def create_resonance_pattern(shape: Tuple[int, ...],
                           center: Optional[Tuple[int, ...]] = None,
                           wavelength: float = 10.0,
                           amplitude: float = 1.0,
                           decay_rate: float = 0.1) -> np.ndarray:
    """Create a resonant standing wave pattern.
    
    This creates a pattern of concentric rings that can form resonance with
    other patterns, useful for memory and attractor formation.
    
    Args:
        shape: Shape of the target field
        center: Center coordinates for the pattern (default: middle of field)
        wavelength: Wavelength of the resonance pattern
        amplitude: Maximum amplitude of the pattern
        decay_rate: How quickly the amplitude decays with distance
        
    Returns:
        NumPy array containing the resonance pattern
    """
    if center is None:
        # Default to center of field
        center = tuple(s // 2 for s in shape)
    
    # Create coordinate grid
    indices = np.indices(shape)
    
    # Calculate distances from center
    sq_distances = np.zeros(shape)
    for dim, loc in enumerate(center):
        sq_distances += (indices[dim] - loc) ** 2
    distances = np.sqrt(sq_distances)
    
    # Create resonance pattern (concentric rings)
    pattern = amplitude * np.cos(2 * np.pi * distances / wavelength)
    
    # Apply decay with distance
    envelope = np.exp(-decay_rate * distances)
    pattern *= envelope
    
    return pattern


def create_attractor_pattern(shape: Tuple[int, ...],
                           center: Optional[Tuple[int, ...]] = None,
                           strength: float = 1.0,
                           radius: float = 5.0) -> np.ndarray:
    """Create an attractor basin pattern that can influence field dynamics.
    
    Args:
        shape: Shape of the target field
        center: Center coordinates for the pattern (default: middle of field)
        strength: Strength of the attractor (positive or negative)
        radius: Radius of the attractor basin
        
    Returns:
        NumPy array containing the attractor pattern
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
    
    # Create attractor basin (negative gaussian = attractor)
    # The negative sign creates a "basin" that draws energy toward it
    basin = -strength * np.exp(-sq_distances / (2 * radius ** 2))
    
    # Add small positive rim around the basin for stability
    rim = 0.2 * strength * np.exp(-(np.sqrt(sq_distances) - 1.5 * radius)**2 / (0.5 * radius)**2)
    
    pattern = basin + rim
    
    return pattern


def create_knowledge_pattern(shape: Tuple[int, ...],
                           trytes: List[Tryte],
                           centers: Optional[List[Tuple[int, ...]]] = None,
                           connection_strength: float = 0.5) -> np.ndarray:
    """Create a complex knowledge pattern from multiple interconnected trytes.
    
    Args:
        shape: Shape of the target field
        trytes: List of trytes forming the knowledge structure
        centers: List of center positions for each tryte (default: distributed pattern)
        connection_strength: Strength of connections between trytes
        
    Returns:
        NumPy array containing the knowledge pattern
    """
    pattern = np.zeros(shape)
    n_trytes = len(trytes)
    
    # Generate centers if not provided
    if centers is None:
        # Distribute trytes in a circular pattern
        centers = []
        center = tuple(s // 2 for s in shape)  # Center of field
        radius = min(s // 3 for s in shape)    # Radius for distribution
        
        for i in range(n_trytes):
            angle = 2 * np.pi * i / n_trytes
            offset = (int(radius * np.cos(angle)), int(radius * np.sin(angle)))
            tryte_center = tuple(c + o for c, o in zip(center, offset[:len(center)]))
            centers.append(tryte_center)
    
    # Add each tryte to the pattern
    for i, (tryte, center) in enumerate(zip(trytes, centers)):
        tryte_pattern = tryte.to_wave_pattern(shape, center)
        pattern += tryte_pattern
    
    # Add connections between trytes
    if n_trytes > 1 and connection_strength > 0:
        for i in range(n_trytes):
            for j in range(i+1, n_trytes):
                # Create a simple connection as a damped sine wave along the path
                start = centers[i]
                end = centers[j]
                
                # Create coordinate grid
                indices = np.indices(shape)
                
                # Calculate squared distances from the line connecting the two centers
                # This creates a 'path' between the two trytes
                path_length = np.sqrt(sum((s - e)**2 for s, e in zip(start, end)))
                if path_length > 0:
                    # Unit vector from start to end
                    direction = tuple((e - s) / path_length for s, e in zip(start, end))
                    
                    # For each point, find the projection onto the path
                    projections = np.zeros(shape)
                    for dim, (s, d) in enumerate(zip(start, direction)):
                        projections += (indices[dim] - s) * d
                    
                    # Calculate distance from path
                    sq_dist_from_path = np.zeros(shape)
                    for dim in range(len(shape)):
                        # Vector from start to current point
                        from_start = indices[dim] - start[dim]
                        # Component of this vector parallel to path
                        parallel_comp = projections * direction[dim]
                        # Component perpendicular to path
                        perp_comp = from_start - parallel_comp
                        sq_dist_from_path += perp_comp**2
                    
                    # Create connection pattern: oscillating wave along the path, decaying away from path
                    path_phase = 5 * np.sin(2 * np.pi * projections / path_length)
                    path_mask = np.exp(-sq_dist_from_path / (2 * (2.0)**2))
                    
                    # Connection is only along the path between start and end
                    path_mask *= (projections >= 0) & (projections <= path_length)
                    
                    # Add to overall pattern
                    pattern += connection_strength * path_phase * path_mask
    
    return pattern
