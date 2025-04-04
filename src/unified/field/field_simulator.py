"""Field simulator for the UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

from ..substrate.manifold import HolographicManifold
from .energy_field import EnergyField
from .data_field import DataField


@dataclass
class FieldSimulator:
    """Integrates energy and data fields with the holographic substrate.
    
    The simulator manages the interactions between different field components
    and runs the time evolution of the entire system.
    
    Attributes:
        manifold: The holographic manifold (substrate)
        energy_field: Energy distribution across the manifold
        data_fields: One or more data fields representing information
        dt: Time step size for simulation
    """
    
    manifold: HolographicManifold
    energy_field: EnergyField
    data_fields: List[DataField] = field(default_factory=list)
    dt: float = 0.1
    time: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    
    def add_data_field(self, data_field: DataField) -> None:
        """Add a data field to the simulation.
        
        Args:
            data_field: Data field to add
        """
        self.data_fields.append(data_field)
    
    def step(self, num_steps: int = 1) -> None:
        """Run the simulation for a specified number of steps.
        
        Args:
            num_steps: Number of time steps to simulate
        """
        for _ in range(num_steps):
            # First update the energy field
            self.energy_field.update(dt=self.dt)
            
            # Get curvature from the manifold
            curvature = self.manifold._grid if self.manifold._grid is not None else None
            
            # Then warp the manifold based on energy (like mass in GR)
            if curvature is not None:
                self.manifold.warp_by_energy(self.energy_field.field, factor=0.01 * self.dt)
            
            # Finally update each data field, influenced by energy and curvature
            for data_field in self.data_fields:
                data_field.update(dt=self.dt, 
                                 energy_field=self.energy_field.field,
                                 substrate_curvature=curvature)
            
            # Record current state (simplified to avoid memory issues)
            if len(self.history) < 100:  # Limit history size
                self.history.append({
                    'time': self.time,
                    'energy_total': self.energy_field.get_total_energy(),
                    'data_structure': [df.compute_structure_metric() for df in self.data_fields]
                })
            
            # Update time
            self.time += self.dt
    
    def inject_energy_pulse(self, location: Tuple[int, ...], amount: float, radius: int = 1) -> None:
        """Inject energy at a specific location in the field.
        
        Args:
            location: Coordinates where energy is injected
            amount: Amount of energy to inject
            radius: Radius of the energy injection
        """
        self.energy_field.inject_energy(location, amount, radius)
    
    def inject_data_pattern(self, pattern: np.ndarray, location: Tuple[int, ...], 
                           field_idx: int = 0) -> None:
        """Inject a data pattern into a specific data field.
        
        Args:
            pattern: Data pattern to inject
            location: Coordinates where pattern is centered
            field_idx: Index of the data field to inject into
        """
        if field_idx >= len(self.data_fields):
            raise ValueError(f"No data field at index {field_idx}")
        
        self.data_fields[field_idx].inject_pattern(pattern, location)
    
    def find_attractors(self, field_idx: int = 0, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find attractor basins in a data field.
        
        Args:
            field_idx: Index of the data field to analyze
            threshold: Minimum value to consider as a potential attractor
            
        Returns:
            List of dictionaries containing attractor properties
        """
        if field_idx >= len(self.data_fields):
            raise ValueError(f"No data field at index {field_idx}")
        
        return self.data_fields[field_idx].find_attractors(threshold)
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the current state of the simulation.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with visualization
        """
        # Determine number of data fields to display
        n_data_fields = len(self.data_fields)
        n_rows = 1 + (n_data_fields + 1) // 2  # Energy field + data fields (2 per row)
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        if n_rows == 1:
            axes = np.array([axes])  # Ensure axes is 2D
        
        # Display energy field
        self.energy_field.visualize(ax=axes[0, 0])
        
        # Display substrate curvature
        if self.manifold._grid is not None:
            if len(self.manifold._grid.shape) == 1:
                axes[0, 1].plot(self.manifold._grid)
                axes[0, 1].set_title("Substrate Curvature")
            else:
                im = axes[0, 1].imshow(self.manifold._grid, cmap='coolwarm', origin='lower')
                plt.colorbar(im, ax=axes[0, 1], label="Curvature")
                axes[0, 1].set_title("Substrate Curvature")
        else:
            axes[0, 1].set_title("Substrate Curvature (Not Initialized)")
            axes[0, 1].set_axis_off()
        
        # Display data fields
        for i, data_field in enumerate(self.data_fields):
            row = 1 + i // 2
            col = i % 2
            if row < axes.shape[0] and col < axes.shape[1]:
                data_field.visualize(ax=axes[row, col], show_attractors=True)
        
        # Hide any unused axes
        for i in range(n_data_fields + 1, n_rows * 2):
            row = 1 + i // 2
            col = i % 2
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].set_axis_off()
        
        fig.tight_layout()
        return fig
    
    def plot_evolution(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot the evolution of key metrics over time.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with plots
        """
        if not self.history:
            raise ValueError("No history recorded yet. Run simulation first.")
        
        # Extract data from history
        times = [h['time'] for h in self.history]
        energy_totals = [h['energy_total'] for h in self.history]
        
        # Set up figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot energy evolution
        axes[0].plot(times, energy_totals)
        axes[0].set_title("Total Energy Over Time")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Total Energy")
        
        # Plot data structure metrics for each field
        n_fields = len(self.data_fields)
        for i in range(n_fields):
            field_structure = [h['data_structure'][i] if i < len(h['data_structure']) else 0 
                              for h in self.history]
            axes[1].plot(times, field_structure, label=f"Field {i}")
        
        axes[1].set_title("Data Structure Metrics Over Time")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Structure Metric")
        if n_fields > 1:
            axes[1].legend()
        
        fig.tight_layout()
        return fig
