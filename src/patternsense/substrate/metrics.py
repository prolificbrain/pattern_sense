"""Metric tensor and curvature field implementations for holographic substrate."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable


@dataclass
class MetricTensor:
    """Represents the metric tensor of the holographic manifold.
    
    The metric tensor defines how distance is measured in curved space,
    which affects how information and energy propagate through the substrate.
    
    Attributes:
        dimensions: Number of dimensions of the space
        tensor_function: Function that calculates the metric tensor at a point
    """
    
    dimensions: int
    tensor_function: Optional[Callable] = None
    
    def __post_init__(self):
        """Initialize the metric tensor with a default function if none provided."""
        if self.tensor_function is None:
            # Default to Euclidean metric (identity matrix)
            self.tensor_function = lambda coords: np.eye(self.dimensions)
    
    def at(self, coords: Tuple[float, ...]) -> np.ndarray:
        """Get the metric tensor at specific coordinates.
        
        Args:
            coords: Coordinates where to evaluate the metric tensor
            
        Returns:
            Metric tensor as a square matrix
        """
        return self.tensor_function(coords)
    
    def distance(self, coords1: Tuple[float, ...], coords2: Tuple[float, ...], 
                steps: int = 10) -> float:
        """Calculate the geodesic distance between two points.
        
        In curved space, this approximates the geodesic by discretizing the path
        and summing the distance elements.
        
        Args:
            coords1: Starting coordinates
            coords2: Ending coordinates
            steps: Number of steps for numerical integration
            
        Returns:
            Approximate geodesic distance between the points
        """
        # Simple linear interpolation between points
        path = np.zeros((steps, len(coords1)))
        
        for i in range(steps):
            t = i / (steps - 1)
            for j, (start, end) in enumerate(zip(coords1, coords2)):
                path[i, j] = start + t * (end - start)
        
        # Calculate distance along the path
        distance = 0.0
        for i in range(steps - 1):
            # Get midpoint between consecutive points
            mid_point = (path[i] + path[i+1]) / 2
            
            # Get metric at midpoint
            g = self.at(tuple(mid_point))
            
            # Calculate displacement vector
            dx = path[i+1] - path[i]
            
            # Calculate distance element (ds² = g_ij dx^i dx^j)
            ds_squared = np.dot(dx, np.dot(g, dx))
            distance += np.sqrt(max(0, ds_squared))  # Ensure non-negative
        
        return distance


@dataclass
class CurvatureField:
    """Represents the curvature of the holographic substrate.
    
    Curvature affects how information flows and interacts within the substrate,
    creating potential basins and barriers that shape computational dynamics.
    
    Attributes:
        dimensions: Number of dimensions of the space
        curvature_function: Function that calculates curvature at a point
    """
    
    dimensions: int
    curvature_function: Optional[Callable] = None
    _grid: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize the curvature field with a default function if none provided."""
        if self.curvature_function is None:
            # Default to flat space (zero curvature)
            self.curvature_function = lambda coords: 0.0
    
    def initialize_grid(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize a grid representation of the curvature field.
        
        Args:
            shape: Shape of the grid
            
        Returns:
            Grid of curvature values
        """
        # Create coordinate grid
        indices = np.indices(shape)
        coords = np.zeros(shape + (len(shape),))
        
        for dim in range(len(shape)):
            # Normalize coordinates to [-1, 1]
            coords[..., dim] = 2 * indices[dim] / (shape[dim] - 1) - 1
        
        # Calculate curvature at each point
        self._grid = np.zeros(shape)
        
        # Iterate over all grid points
        it = np.nditer(self._grid, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            point_coords = tuple(coords[idx][dim] for dim in range(len(shape)))
            self._grid[idx] = self.curvature_function(point_coords)
        
        return self._grid
    
    def at(self, coords: Tuple[float, ...]) -> float:
        """Get the curvature at specific coordinates.
        
        Args:
            coords: Coordinates where to evaluate the curvature
            
        Returns:
            Curvature value
        """
        return self.curvature_function(coords)
    
    def warp(self, coords: Tuple[float, ...], displacement: np.ndarray) -> np.ndarray:
        """Warp a displacement vector based on local curvature.
        
        This simulates how the substrate's curvature affects the propagation
        of information and energy.
        
        Args:
            coords: Coordinates where to evaluate the warping
            displacement: Displacement vector to warp
            
        Returns:
            Warped displacement vector
        """
        # Simple model: scale displacement based on local curvature
        curvature = self.at(coords)
        
        # Positive curvature expands displacement, negative compresses
        scale_factor = 1.0 / (1.0 + 0.1 * curvature)
        
        return displacement * scale_factor
