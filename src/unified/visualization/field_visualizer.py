"""Field visualization for the UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

from ..field.data_field import DataField
from ..field.energy_field import EnergyField
from ..substrate.manifold import HolographicManifold


@dataclass
class FieldVisualizer:
    """Visualizer for field components of the UNIFIED Consciousness Engine.
    
    This class provides methods for visualizing the various field components
    of the system, including data fields, energy fields, and substrate curvature.
    
    Attributes:
        figsize: Default figure size for visualizations
    """
    
    figsize: Tuple[int, int] = (12, 8)
    _figures: Dict[str, Figure] = field(default_factory=dict, init=False, repr=False)
    
    def visualize_data_field(self, data_field: DataField, show_attractors: bool = True,
                           title: Optional[str] = None) -> Figure:
        """Visualize a data field and its attractors.
        
        Args:
            data_field: The data field to visualize
            show_attractors: Whether to highlight attractors
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        data_field.visualize(ax=ax, show_attractors=show_attractors)
        
        if title:
            ax.set_title(title)
        
        fig.tight_layout()
        self._figures['data_field'] = fig
        return fig
    
    def visualize_energy_field(self, energy_field: EnergyField,
                             title: Optional[str] = None) -> Figure:
        """Visualize an energy field.
        
        Args:
            energy_field: The energy field to visualize
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        energy_field.visualize(ax=ax)
        
        if title:
            ax.set_title(title)
        
        fig.tight_layout()
        self._figures['energy_field'] = fig
        return fig
    
    def visualize_substrate(self, manifold: HolographicManifold,
                          title: Optional[str] = None) -> Figure:
        """Visualize the substrate manifold.
        
        Args:
            manifold: The holographic manifold to visualize
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if manifold._grid is None:
            manifold.initialize_grid()
        
        grid = manifold._grid
        
        if len(grid.shape) == 1:
            ax.plot(grid)
            ax.set_title("Substrate Curvature")
            ax.set_xlabel("Position")
            ax.set_ylabel("Curvature")
        elif len(grid.shape) == 2:
            im = ax.imshow(grid, cmap='coolwarm', origin='lower')
            plt.colorbar(im, ax=ax, label="Curvature")
            ax.set_title("Substrate Curvature")
        else:
            # For higher dimensions, just show a slice
            middle_indices = tuple(s // 2 for s in grid.shape[2:])
            slice_indices = (slice(None), slice(None)) + middle_indices
            im = ax.imshow(grid[slice_indices], cmap='coolwarm', origin='lower')
            plt.colorbar(im, ax=ax, label="Curvature")
            ax.set_title(f"Substrate Curvature (Slice at {middle_indices})")
        
        if title:
            ax.set_title(title)
        
        fig.tight_layout()
        self._figures['substrate'] = fig
        return fig
    
    def visualize_combined_fields(self, data_field: DataField, energy_field: EnergyField,
                               manifold: Optional[HolographicManifold] = None,
                               title: Optional[str] = None) -> Figure:
        """Visualize data and energy fields side by side, with substrate if provided.
        
        Args:
            data_field: The data field to visualize
            energy_field: The energy field to visualize
            manifold: Optional holographic manifold to visualize
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        if manifold is not None:
            fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        else:
            fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Visualize data field
        data_field.visualize(ax=axes[0], show_attractors=True)
        axes[0].set_title("Data Field")
        
        # Visualize energy field
        energy_field.visualize(ax=axes[1])
        axes[1].set_title("Energy Field")
        
        # Visualize substrate if provided
        if manifold is not None:
            if manifold._grid is None:
                manifold.initialize_grid()
            
            grid = manifold._grid
            
            if len(grid.shape) == 1:
                axes[2].plot(grid)
                axes[2].set_title("Substrate Curvature")
                axes[2].set_xlabel("Position")
                axes[2].set_ylabel("Curvature")
            elif len(grid.shape) == 2:
                im = axes[2].imshow(grid, cmap='coolwarm', origin='lower')
                plt.colorbar(im, ax=axes[2], label="Curvature")
                axes[2].set_title("Substrate Curvature")
            else:
                # For higher dimensions, just show a slice
                middle_indices = tuple(s // 2 for s in grid.shape[2:])
                slice_indices = (slice(None), slice(None)) + middle_indices
                im = axes[2].imshow(grid[slice_indices], cmap='coolwarm', origin='lower')
                plt.colorbar(im, ax=axes[2], label="Curvature")
                axes[2].set_title(f"Substrate Curvature (Slice)")
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        fig.tight_layout()
        self._figures['combined'] = fig
        return fig
    
    def visualize_field_evolution(self, data_field: DataField,
                              energy_field: Optional[EnergyField] = None) -> Figure:
        """Visualize the evolution of field metrics over time.
        
        Args:
            data_field: The data field with history to visualize
            energy_field: Optional energy field to include
            
        Returns:
            Matplotlib figure with plots
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot data field structure metric
        history_len = len(data_field.history)
        times = np.arange(history_len)
        
        structure_metrics = []
        for state in data_field.history:
            # Create temporary data field with this state
            temp_field = DataField(shape=data_field.shape)
            temp_field._field = state
            structure_metrics.append(temp_field.compute_structure_metric())
        
        axes[0].plot(times, structure_metrics)
        axes[0].set_title("Data Field Structure Metric Over Time")
        axes[0].set_ylabel("Structure Metric")
        
        # Plot energy if provided
        if energy_field is not None and hasattr(energy_field, 'history'):
            energy_totals = [np.sum(state) for state in energy_field.history]
            energy_times = np.arange(len(energy_totals))
            
            # Trim to match data field history length if needed
            if len(energy_times) > len(times):
                energy_times = energy_times[-len(times):]
                energy_totals = energy_totals[-len(times):]
            
            axes[1].plot(energy_times, energy_totals)
            axes[1].set_title("Total Energy Over Time")
        else:
            # If no energy field history, plot data field energy
            field_energy = [np.sum(state**2) for state in data_field.history]
            axes[1].plot(times, field_energy)
            axes[1].set_title("Data Field Energy Over Time")
        
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Energy")
        
        fig.tight_layout()
        self._figures['evolution'] = fig
        return fig
    
    def visualize_attractor_dynamics(self, data_field: DataField,
                                 threshold: float = 0.5) -> Figure:
        """Visualize attractors and their dynamics in the data field.
        
        Args:
            data_field: The data field to analyze
            threshold: Threshold for attractor detection
            
        Returns:
            Matplotlib figure with visualization
        """
        attractors = data_field.find_attractors(threshold=threshold)
        
        if len(data_field.shape) > 2:
            # Only handle 1D and 2D fields for now
            raise ValueError("Attractor visualization only supported for 1D and 2D fields")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Visualize data field
        data_field.visualize(ax=ax, show_attractors=False)
        
        # Highlight attractors
        if len(data_field.shape) == 1:
            for attractor in attractors:
                centroid = attractor['centroid']
                strength = attractor['strength']
                size = attractor['size']
                
                # Draw marker scaled by strength and size
                marker_size = 20 + 100 * strength * np.sqrt(size)
                ax.plot(centroid[0], data_field.field[centroid], 'ro', markersize=marker_size / 10, alpha=0.7)
                
                # Annotate with strength
                ax.annotate(f"{strength:.2f}", xy=(centroid[0], data_field.field[centroid]),
                           xytext=(0, 10), textcoords="offset points",
                           ha="center")
        
        elif len(data_field.shape) == 2:
            for attractor in attractors:
                centroid = attractor['centroid']
                strength = attractor['strength']
                size = attractor['size']
                
                # Draw marker scaled by strength and size
                marker_size = 20 + 100 * strength * np.sqrt(size)
                ax.plot(centroid[1], centroid[0], 'ro', markersize=marker_size / 10, alpha=0.7)
                
                # Draw circle to show basin extent
                from matplotlib.patches import Circle
                basin_radius = np.sqrt(size / np.pi)
                circle = Circle((centroid[1], centroid[0]), basin_radius,
                               fill=False, color='r', alpha=0.5)
                ax.add_patch(circle)
                
                # Annotate with strength
                ax.annotate(f"{strength:.2f}", xy=(centroid[1], centroid[0]),
                           xytext=(0, 10), textcoords="offset points",
                           ha="center")
        
        ax.set_title(f"Data Field Attractors (Threshold: {threshold})")
        
        fig.tight_layout()
        self._figures['attractors'] = fig
        return fig
    
    def save_figures(self, base_path: str, format: str = 'png') -> List[str]:
        """Save all created figures to disk.
        
        Args:
            base_path: Base path for saving figures
            format: File format (png, pdf, svg, etc.)
            
        Returns:
            List of saved file paths
        """
        import os
        
        saved_paths = []
        
        for name, fig in self._figures.items():
            file_path = os.path.join(base_path, f"{name}.{format}")
            fig.savefig(file_path, format=format, dpi=300, bbox_inches='tight')
            saved_paths.append(file_path)
        
        return saved_paths
