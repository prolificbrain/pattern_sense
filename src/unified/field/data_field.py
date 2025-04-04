"""Data field implementation for UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, List, Dict, Any


@dataclass
class DataField:
    """Represents structured information embedded in the holographic substrate.
    
    In the UNIFIED Consciousness framework, data is not just symbols but
    structured perturbations in the substrate field with physical properties.
    Data fields can morph, interact, and form stable patterns.
    
    Attributes:
        shape: Dimensions of the field
        initial_distribution: Function to initialize the data distribution
        persistence: How strongly the data structure resists entropy/decay
    """
    
    shape: Tuple[int, ...]
    initial_distribution: Optional[Callable] = None
    persistence: float = 0.9  # Higher values mean slower decay
    _field: np.ndarray = field(init=False, repr=False)
    _history: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the data field."""
        if self.initial_distribution is None:
            # Default to zero data field (no data)
            self._field = np.zeros(self.shape)
        else:
            # Use provided initialization function
            coords = self._create_coordinate_grid()
            self._field = self.initial_distribution(coords)
        
        # Add initial state to history
        self._history.append(self._field.copy())
    
    def _create_coordinate_grid(self) -> np.ndarray:
        """Create a grid of coordinates for the field."""
        grid_points = [np.linspace(-1, 1, dim) for dim in self.shape]
        mesh_grid = np.meshgrid(*grid_points, indexing='ij')
        return np.stack(mesh_grid, axis=-1)
    
    @property
    def field(self) -> np.ndarray:
        """Get the current data field state."""
        return self._field
    
    @property
    def history(self) -> List[np.ndarray]:
        """Get the history of data field states."""
        return self._history
    
    def inject_pattern(self, pattern: np.ndarray, location: Tuple[int, ...]) -> None:
        """Inject a data pattern at a specific location.
        
        Args:
            pattern: Data pattern to inject
            location: Coordinates where pattern is centered
        """
        # Handle the case where pattern shape doesn't match field shape
        if pattern.shape != self._field.shape:
            # Create a new pattern with the correct shape
            new_pattern = np.zeros_like(self._field)
            
            # Calculate center of pattern
            pattern_center = tuple(s // 2 for s in pattern.shape)
            
            # Calculate boundaries for both arrays
            # For destination (field)
            dest_start = tuple(max(0, loc - pc) for loc, pc in zip(location, pattern_center))
            dest_end = tuple(min(fs, loc + (ps - pc)) 
                           for fs, loc, ps, pc in zip(self._field.shape, location, 
                                                   pattern.shape, pattern_center))
            
            # For source (pattern)
            src_start = tuple(max(0, pc - loc) if loc < pc else 0 
                            for loc, pc in zip(location, pattern_center))
            src_end = tuple(min(ps, pc + (fs - loc)) if loc >= pc else ps - (pc - loc)
                          for fs, loc, ps, pc in zip(self._field.shape, location, 
                                                  pattern.shape, pattern_center))
            
            # Create slices
            dest_slices = tuple(slice(ds, de) for ds, de in zip(dest_start, dest_end))
            src_slices = tuple(slice(ss, se) for ss, se in zip(src_start, src_end))
            
            # Copy the relevant part of the pattern to the field
            try:
                new_pattern[dest_slices] = pattern[src_slices]
                pattern = new_pattern
            except ValueError as e:
                # If there's still an issue, fallback to a simple centered Gaussian
                from scipy.ndimage import gaussian_filter
                new_pattern = np.zeros_like(self._field)
                new_pattern[location] = np.max(pattern) if np.max(pattern) != 0 else 1.0
                pattern = gaussian_filter(new_pattern, sigma=2.0)
        
        # Now proceed with the injection using the correctly shaped pattern
        self._field += pattern
    
    def update(self, dt: float = 0.1, energy_field: Optional[np.ndarray] = None,
              substrate_curvature: Optional[np.ndarray] = None) -> None:
        """Update the data field state over time.
        
        Args:
            dt: Time step size
            energy_field: Energy field influencing data transformation
            substrate_curvature: Curvature of the substrate affecting propagation
        """
        # Apply diffusion (controlled by substrate curvature)
        laplacian = self._compute_laplacian()
        
        # Modulate diffusion by substrate curvature if provided
        diffusion_rate = 0.05  # Base diffusion rate
        if substrate_curvature is not None:
            # Higher curvature = slower diffusion
            # Ensure shapes match
            if substrate_curvature.shape == self._field.shape:
                diffusion_modifier = 1 / (1 + np.abs(substrate_curvature))
                diffusion = diffusion_rate * diffusion_modifier * laplacian
            else:
                diffusion = diffusion_rate * laplacian
        else:
            diffusion = diffusion_rate * laplacian
        
        # Apply energy field effects if provided
        energy_term = np.zeros_like(self._field)
        if energy_field is not None:
            # Ensure shapes match
            if energy_field.shape == self._field.shape:
                # Energy amplifies or suppresses data based on local values
                # High energy = more change, low energy = less change
                energy_term = 0.1 * energy_field * self._field
        
        # Apply persistence/decay
        decay = (self.persistence - 1) * self._field
        
        # Update using a field equation inspired by wave dynamics
        self._field += dt * (diffusion + energy_term + decay)
        
        # Record history (limited to avoid memory issues)
        if len(self._history) > 100:
            self._history.pop(0)
        self._history.append(self._field.copy())
    
    def _compute_laplacian(self) -> np.ndarray:
        """Compute the Laplacian of the data field."""
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
    
    def interfere(self, other_field: 'DataField', interaction_strength: float = 0.5) -> 'DataField':
        """Create a new field from the interference of this field with another.
        
        Args:
            other_field: Another data field to interfere with
            interaction_strength: Strength of the interference effect
            
        Returns:
            New data field resulting from interference
        """
        if self.shape != other_field.shape:
            raise ValueError("Fields must have the same shape to interfere")
        
        # Create result field
        result = DataField(shape=self.shape)
        
        # Interference formula: constructive and destructive based on phase
        # Simplified model: linear combination
        result._field = self._field + other_field.field \
                       + interaction_strength * self._field * other_field.field
        
        return result
    
    def compute_structure_metric(self) -> float:
        """Calculate a metric of how structured the data field is.
        
        Higher values indicate more organized/structured data.
        
        Returns:
            Structure metric value
        """
        # Calculate gradient magnitude
        gradient = self.get_gradient()
        gradient_magnitude = np.sum(gradient**2, axis=-1)
        
        # Calculate normalized entropy
        abs_field = np.abs(self._field)
        total = np.sum(abs_field)
        if total > 0:
            normalized_field = abs_field / total
            # Remove zeros to avoid log(0)
            nonzero_mask = normalized_field > 0
            entropy = -np.sum(normalized_field[nonzero_mask] * 
                             np.log(normalized_field[nonzero_mask]))
            max_entropy = np.log(self._field.size)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            normalized_entropy = 0
        
        # Combine gradient and entropy for structure metric
        # High structure = high gradient (sharp features) and low entropy (order)
        structure_metric = np.mean(gradient_magnitude) * (1 - normalized_entropy)
        
        return structure_metric
    
    def get_gradient(self) -> np.ndarray:
        """Calculate the gradient of the data field.
        
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
    
    def find_attractors(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find attractor basins in the data field.
        
        Attractors are regions of high data intensity surrounded by lower values,
        forming stable structures that can represent knowledge or memory.
        
        Args:
            threshold: Minimum value to consider as a potential attractor
            
        Returns:
            List of dictionaries containing attractor properties
        """
        from scipy import ndimage
        
        # Find local maxima
        data_max = ndimage.maximum_filter(self._field, size=3)
        maxima = (self._field == data_max) & (self._field > threshold)
        
        # Label connected components
        labeled_maxima, num_maxima = ndimage.label(maxima)
        
        attractors = []
        for i in range(1, num_maxima + 1):
            # Get indices of this attractor
            indices = np.where(labeled_maxima == i)
            
            # Calculate centroid
            centroid = tuple(int(np.mean(idx)) for idx in indices)
            
            # Get strength (maximum value in the attractor)
            strength = np.max(self._field[labeled_maxima == i])
            
            # Calculate size (number of points)
            size = np.sum(labeled_maxima == i)
            
            attractors.append({
                'centroid': centroid,
                'strength': strength,
                'size': size
            })
        
        return attractors
    
    def visualize(self, ax=None, show_attractors: bool = True):
        """Visualize the data field and its attractors.
        
        Args:
            ax: Matplotlib axis for plotting
            show_attractors: Whether to highlight attractors
            
        Returns:
            Matplotlib axis with plot
        """
        if ax is None:
            _, ax = plt.subplots()
        
        if len(self.shape) == 1:
            ax.plot(self._field)
            ax.set_title("Data Field")
            ax.set_xlabel("Position")
            ax.set_ylabel("Data Value")
            
            if show_attractors:
                attractors = self.find_attractors()
                for attractor in attractors:
                    ax.plot(attractor['centroid'][0], self._field[attractor['centroid']], 'ro')
        
        elif len(self.shape) == 2:
            im = ax.imshow(self._field, cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, label="Data Value")
            ax.set_title("Data Field")
            
            if show_attractors:
                attractors = self.find_attractors()
                for attractor in attractors:
                    ax.plot(attractor['centroid'][1], attractor['centroid'][0], 'ro', markersize=5)
        
        else:
            # For higher dimensions, just show a slice
            middle_indices = tuple(s // 2 for s in self.shape[2:])
            slice_indices = (slice(None), slice(None)) + middle_indices
            im = ax.imshow(self._field[slice_indices], cmap='viridis', origin='lower')
            plt.colorbar(im, ax=ax, label="Data Value")
            ax.set_title(f"Data Field (Slice at {middle_indices})")
        
        return ax
    
    def animate_history(self, interval: int = 100, figsize: Tuple[int, int] = (8, 8)):
        """Create an animation of the data field's evolution over time.
        
        Args:
            interval: Time between animation frames in milliseconds
            figsize: Figure size
            
        Returns:
            Matplotlib animation object
        """
        import matplotlib.animation as animation
        
        if len(self._history) < 2:
            raise ValueError("Need at least two history frames for animation")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(self.shape) == 1:
            line, = ax.plot(self._history[0])
            ax.set_ylim(min(np.min(h) for h in self._history) - 0.1,
                        max(np.max(h) for h in self._history) + 0.1)
            
            def update(frame):
                line.set_ydata(self._history[frame])
                ax.set_title(f"Data Field (Frame {frame})")
                return line,
                
        elif len(self.shape) == 2:
            # For 2D, use imshow
            im = ax.imshow(self._history[0], cmap='viridis', origin='lower',
                           vmin=min(np.min(h) for h in self._history),
                           vmax=max(np.max(h) for h in self._history))
            plt.colorbar(im, ax=ax, label="Data Value")
            
            def update(frame):
                im.set_array(self._history[frame])
                ax.set_title(f"Data Field (Frame {frame})")
                return im,
                
        else:
            # For higher dimensions, just animate a slice
            middle_indices = tuple(s // 2 for s in self.shape[2:])
            slice_indices = lambda h: h[(slice(None), slice(None)) + middle_indices]
            
            im = ax.imshow(slice_indices(self._history[0]), cmap='viridis', origin='lower',
                           vmin=min(np.min(slice_indices(h)) for h in self._history),
                           vmax=max(np.max(slice_indices(h)) for h in self._history))
            plt.colorbar(im, ax=ax, label="Data Value")
            
            def update(frame):
                im.set_array(slice_indices(self._history[frame]))
                ax.set_title(f"Data Field (Frame {frame})")
                return im,
        
        ani = animation.FuncAnimation(fig, update, frames=len(self._history),
                                      interval=interval, blit=True)
        
        return ani
