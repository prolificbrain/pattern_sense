"""Energy field implementation for UNIFIED Consciousness Engine."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable


@dataclass
class EnergyField:
    """Represents the energy distribution across the holographic substrate.
    
    The energy field defines how computation is powered and how information
    can be transformed. Energy flows across the substrate, enabling state
    changes and computational operations.
    
    Attributes:
        shape: Dimensions of the field
        initial_distribution: Function to initialize the energy distribution
        dissipation_rate: Rate at which energy dissipates over time
    """
    
    shape: Tuple[int, ...]
    initial_distribution: Optional[Callable] = None
    dissipation_rate: float = 0.01
    _field: np.ndarray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the energy field."""
        if self.initial_distribution is None:
            # Default to random energy distribution
            self._field = np.random.random(self.shape) * 0.1
        else:
            # Use provided initialization function
            coords = self._create_coordinate_grid()
            self._field = self.initial_distribution(coords)
    
    def _create_coordinate_grid(self) -> np.ndarray:
        """Create a grid of coordinates for the field."""
        grid_points = [np.linspace(-1, 1, dim) for dim in self.shape]
        mesh_grid = np.meshgrid(*grid_points, indexing='ij')
        return np.stack(mesh_grid, axis=-1)
    
    @property
    def field(self) -> np.ndarray:
        """Get the current energy field state."""
        return self._field
    
    def update(self, dt: float = 0.1, external_sources: Optional[np.ndarray] = None) -> None:
        """Update the energy field state over time.
        
        Args:
            dt: Time step size
            external_sources: Additional energy input from outside the system
        """
        # Apply diffusion (energy spreads out)
        laplacian = self._compute_laplacian()
        diffusion = 0.1 * laplacian
        
        # Apply dissipation (energy is consumed by computation)
        dissipation = -self.dissipation_rate * self._field
        
        # Apply external sources
        source_term = np.zeros_like(self._field)
        if external_sources is not None:
            source_term = external_sources
        
        # Update using a simplified field equation
        self._field += dt * (diffusion + dissipation + source_term)
        
        # Ensure energy remains positive
        self._field = np.maximum(0, self._field)
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute the Laplacian of the energy field."""
        laplacian = np.zeros_like(self._field)
        
        # For each dimension, compute second derivative
        for dim in range(len(self.shape)):
            # Create slices for forward, center, and backward differences
            slices_forward = [slice(None)] * len(self.shape)
            slices_center = [slice(None)] * len(self.shape)
            slices_backward = [slice(None)] * len(self.shape)
            
            slices_forward[dim] = slice(2, None)
            slices_center[dim] = slice(1, -1)
            slices_backward[dim] = slice(0, -2)
            
            # Interior points (using central difference)
            if self.shape[dim] > 2:
                interior = self._field[tuple(slices_forward)] + \
                           self._field[tuple(slices_backward)] - \
                           2 * self._field[tuple(slices_center)]
                
                # Assign to corresponding part of laplacian
                laplacian_slices = [slice(None)] * len(self.shape)
                laplacian_slices[dim] = slice(1, -1)
                laplacian[tuple(laplacian_slices)] += interior
        
        return laplacian
    
    def inject_energy(self, location: Tuple[int, ...], amount: float, radius: int = 1) -> None:
        """Inject energy at a specific location in the field.
        
        Args:
            location: Coordinates where energy is injected
            amount: Amount of energy to inject
            radius: Radius of the energy injection (creates a pulse)
        """
        # Create a grid of distances from the location
        grid_shape = self._field.shape
        indices = np.indices(grid_shape)
        
        # Calculate squared distances
        sq_distances = np.zeros(grid_shape)
        for dim, loc in enumerate(location):
            sq_distances += (indices[dim] - loc) ** 2
        
        # Apply Gaussian energy distribution
        distance_mask = np.exp(-sq_distances / (2 * radius ** 2))
        self._field += amount * distance_mask
    
    def get_gradient(self) -> np.ndarray:
        """Calculate the gradient of the energy field.
        
        Returns:
            Gradient vector field with shape (*self.shape, len(self.shape))
        """
        gradient = np.zeros(self._field.shape + (len(self.shape),))
        
        # Calculate gradient for each dimension
        for dim in range(len(self.shape)):
            # Create slices for forward and backward differences
            slices_forward = [slice(None)] * len(self.shape)
            slices_backward = [slice(None)] * len(self.shape)
            
            slices_forward[dim] = slice(1, None)
            slices_backward[dim] = slice(0, -1)
            
            # Forward difference for the first derivatives
            # We'll use padding to maintain shape
            central_grad = np.zeros_like(self._field)
            interior = self._field[tuple(slices_forward)] - self._field[tuple(slices_backward)]
            
            central_slices = [slice(None)] * len(self.shape)
            central_slices[dim] = slice(0, -1)
            central_grad[tuple(central_slices)] = interior
            
            # Handle boundary
            if self.shape[dim] > 1:
                # Periodic boundary condition
                boundary_slices = [slice(None)] * len(self.shape)
                boundary_slices[dim] = slice(-1, None)
                central_grad[tuple(boundary_slices)] = self._field[tuple([slice(0, 1) if i == dim else slice(None) for i in range(len(self.shape))])] - \
                                                       self._field[tuple(boundary_slices)]
            
            gradient[..., dim] = central_grad
        
        return gradient
    
    def get_energy_at(self, location: Tuple[int, ...]) -> float:
        """Get energy value at a specific location.
        
        Args:
            location: Coordinates to query
            
        Returns:
            Energy value at the specified location
        """
        return self._field[location]
    
    def get_total_energy(self) -> float:
        """Calculate the total energy in the field.
        
        Returns:
            Sum of all energy values in the field
        """
        return np.sum(self._field)
    
    def visualize(self, ax=None):
        """Visualize the energy field (for 1D and 2D fields).
        
        Args:
            ax: Matplotlib axis for plotting
            
        Returns:
            Matplotlib axis with plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            _, ax = plt.subplots()
        
        if len(self.shape) == 1:
            ax.plot(self._field)
            ax.set_title("Energy Field")
            ax.set_xlabel("Position")
            ax.set_ylabel("Energy")
        elif len(self.shape) == 2:
            im = ax.imshow(self._field, cmap='inferno', origin='lower')
            plt.colorbar(im, ax=ax, label="Energy")
            ax.set_title("Energy Field")
        else:
            # For higher dimensions, just show a slice
            middle_indices = tuple(s // 2 for s in self.shape[2:])
            slice_indices = (slice(None), slice(None)) + middle_indices
            im = ax.imshow(self._field[slice_indices], cmap='inferno', origin='lower')
            plt.colorbar(im, ax=ax, label="Energy")
            ax.set_title(f"Energy Field (Slice at {middle_indices})")
        
        return ax
