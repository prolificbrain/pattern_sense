"""Substrate field implementation for UNIFIED Consciousness Engine."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

from .metrics import CurvatureField


@dataclass
class SubstrateField:
    """Represents the underlying field of the holographic substrate.
    
    The substrate field defines the physical properties of the space where
    computation occurs, integrating curvature, phase, and other properties
    that affect how data and energy propagate.
    
    Attributes:
        shape: Dimensions of the field
        curvature_field: Curvature properties of the field
        phase_function: Function to calculate phase at a point
    """
    
    shape: Tuple[int, ...]
    curvature_field: Optional[CurvatureField] = None
    phase_function: Optional[Callable] = None
    _field: Optional[np.ndarray] = None
    _phase: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize the substrate field."""
        if self.curvature_field is None:
            # Default to flat curvature field
            self.curvature_field = CurvatureField(dimensions=len(self.shape))
        
        if self.phase_function is None:
            # Default to zero phase everywhere
            self.phase_function = lambda coords: 0.0
        
        # Initialize the field
        self.initialize()
    
    def initialize(self) -> None:
        """Initialize the substrate field with curvature and phase values."""
        # Initialize curvature grid
        curvature_grid = self.curvature_field.initialize_grid(self.shape)
        
        # Initialize phase grid
        self._phase = np.zeros(self.shape)
        
        # Create coordinate grid
        indices = np.indices(self.shape)
        coords = np.zeros(self.shape + (len(self.shape),))
        
        for dim in range(len(self.shape)):
            # Normalize coordinates to [-1, 1]
            coords[..., dim] = 2 * indices[dim] / (self.shape[dim] - 1) - 1
        
        # Calculate phase at each point
        it = np.nditer(self._phase, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            point_coords = tuple(coords[idx][dim] for dim in range(len(self.shape)))
            self._phase[idx] = self.phase_function(point_coords)
        
        # Combined field representation (currently just the curvature)
        self._field = curvature_grid
    
    @property
    def field(self) -> np.ndarray:
        """Get the current substrate field state."""
        if self._field is None:
            self.initialize()
        return self._field
    
    @property
    def phase(self) -> np.ndarray:
        """Get the current phase grid."""
        if self._phase is None:
            self.initialize()
        return self._phase
    
    def warp_propagation(self, data_field: np.ndarray) -> np.ndarray:
        """Warp the propagation of data based on substrate properties.
        
        Args:
            data_field: Data field to warp
            
        Returns:
            Warped data field
        """
        # Simple implementation: modulate data by curvature
        return data_field * (1 + 0.1 * self._field)
    
    def phase_modulate(self, data_field: np.ndarray) -> np.ndarray:
        """Modulate data field by the substrate phase.
        
        Args:
            data_field: Data field to modulate
            
        Returns:
            Phase-modulated data field
        """
        # Apply phase modulation (wave-like effect)
        return data_field * np.cos(self._phase)
    
    def get_energy_gradient(self) -> np.ndarray:
        """Calculate the gradient field that affects energy flow.
        
        Returns:
            Gradient vector field with shape (*self.shape, len(self.shape))
        """
        # Calculate gradient for each dimension
        gradient = np.zeros(self._field.shape + (len(self.shape),))
        
        for dim in range(len(self.shape)):
            # Create slices for forward and backward differences
            slices_forward = [slice(None)] * len(self.shape)
            slices_backward = [slice(None)] * len(self.shape)
            
            slices_forward[dim] = slice(1, None)
            slices_backward[dim] = slice(0, -1)
            
            # Forward difference for the first derivatives
            central_grad = np.zeros_like(self._field)
            interior = self._field[tuple(slices_forward)] - self._field[tuple(slices_backward)]
            
            central_slices = [slice(None)] * len(self.shape)
            central_slices[dim] = slice(0, -1)
            central_grad[tuple(central_slices)] = interior
            
            # Handle boundary (periodic boundary condition)
            if self.shape[dim] > 1:
                boundary_slices = [slice(None)] * len(self.shape)
                boundary_slices[dim] = slice(-1, None)
                central_grad[tuple(boundary_slices)] = self._field[tuple([slice(0, 1) if i == dim else slice(None) for i in range(len(self.shape))])] - \
                                                    self._field[tuple(boundary_slices)]
            
            gradient[..., dim] = central_grad
        
        return gradient
    
    def update(self, energy_field: Optional[np.ndarray] = None) -> None:
        """Update the substrate field based on energy distribution.
        
        Args:
            energy_field: Energy field that warps the substrate
        """
        if energy_field is not None:
            # Energy warps the substrate (similar to mass in GR)
            self._field += 0.01 * energy_field
