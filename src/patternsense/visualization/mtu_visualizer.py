"""MTU visualization for the UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union

from ..mtu.mtu import MinimalThinkingUnit
from ..mtu.mtu_network import MTUNetwork
from ..trits.tryte import Tryte


@dataclass
class MTUVisualizer:
    """Visualizer for MTU components of the UNIFIED Consciousness Engine.
    
    This class provides methods for visualizing individual MTUs, 
    MTU networks, and their emergent dynamics.
    
    Attributes:
        figsize: Default figure size for visualizations
    """
    
    figsize: Tuple[int, int] = (12, 8)
    _figures: Dict[str, Figure] = field(default_factory=dict, init=False, repr=False)
    
    def visualize_mtu(self, mtu: MinimalThinkingUnit,
                    title: Optional[str] = None) -> Figure:
        """Visualize a single MTU's state field.
        
        Args:
            mtu: The MTU to visualize
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        mtu.visualize(ax=ax)
        
        if title:
            ax.set_title(title)
        
        fig.tight_layout()
        self._figures['mtu'] = fig
        return fig
    
    def visualize_mtu_network(self, network: MTUNetwork,
                           highlight_active: bool = True,
                           show_fields: bool = True,
                           title: Optional[str] = None) -> Figure:
        """Visualize an MTU network.
        
        Args:
            network: The MTU network to visualize
            highlight_active: Whether to highlight active MTUs
            show_fields: Whether to show data and energy fields
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        # Use the network's built-in visualization
        fig = network.visualize(figsize=self.figsize)
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        self._figures['network'] = fig
        return fig
    
    def visualize_mtu_states(self, mtus: List[MinimalThinkingUnit],
                          layout: Optional[Tuple[int, int]] = None,
                          title: Optional[str] = None) -> Figure:
        """Visualize the state fields of multiple MTUs in a grid layout.
        
        Args:
            mtus: List of MTUs to visualize
            layout: Grid layout as (rows, cols). If None, determined automatically
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        n_mtus = len(mtus)
        
        if n_mtus == 0:
            raise ValueError("No MTUs provided for visualization")
        
        # Determine layout if not provided
        if layout is None:
            cols = int(np.ceil(np.sqrt(n_mtus)))
            rows = int(np.ceil(n_mtus / cols))
        else:
            rows, cols = layout
        
        fig, axes = plt.subplots(rows, cols, figsize=self.figsize)
        
        # Convert to 2D array if needed
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        # Visualize each MTU
        for i, mtu in enumerate(mtus):
            if i >= rows * cols:
                break
                
            row = i // cols
            col = i % cols
            mtu.visualize(ax=axes[row, col])
            axes[row, col].set_title(f"MTU at {mtu.position}")
        
        # Hide unused axes
        for i in range(n_mtus, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_axis_off()
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        fig.tight_layout()
        self._figures['mtu_states'] = fig
        return fig
    
    def visualize_mtu_history(self, mtu: MinimalThinkingUnit,
                           metrics: Optional[List[str]] = None,
                           title: Optional[str] = None) -> Figure:
        """Visualize the history of an MTU's metrics over time.
        
        Args:
            mtu: The MTU to visualize history for
            metrics: List of metrics to plot. If None, plots all available metrics
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        if not mtu._history:
            raise ValueError("MTU has no recorded history")
        
        # Determine which metrics to plot
        available_metrics = set(mtu._history[0].keys()) - {'time'}
        
        if metrics is None:
            metrics = list(available_metrics)
        else:
            # Ensure all requested metrics are available
            for metric in metrics:
                if metric not in available_metrics:
                    raise ValueError(f"Metric '{metric}' not available in MTU history.")
        
        # Create figure with a subplot for each metric
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.figsize, sharex=True)
        
        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        times = [h['time'] for h in mtu._history]
        
        for i, metric in enumerate(metrics):
            values = [h[metric] for h in mtu._history]
            axes[i].plot(times, values)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        # Set x-label on the bottom subplot
        axes[-1].set_xlabel("Time Step")
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        fig.tight_layout()
        self._figures['mtu_history'] = fig
        return fig
    
    def visualize_network_activity(self, network: MTUNetwork,
                                time_steps: int = 10,
                                title: Optional[str] = None) -> Figure:
        """Visualize the activity patterns in an MTU network over time.
        
        This runs the network for a specified number of time steps and
        visualizes the patterns of activity that emerge.
        
        Args:
            network: The MTU network to visualize
            time_steps: Number of time steps to simulate
            title: Optional title for the plot
            
        Returns:
            Matplotlib figure with visualization
        """
        # Run the network for the specified number of steps
        outputs = network.run(time_steps)
        
        # Count outputs per MTU to measure activity
        mtu_activity = {idx: len(trytes) for idx, trytes in outputs.items()}
        
        # Get network dimensions and MTU positions
        if len(network.shape) > 3:
            raise ValueError("Activity visualization only supported for 1D, 2D, and 3D networks")
            
        mtu_positions = network.get_mtu_positions()
        connections = network.get_connections()
        
        # Create figure
        if len(network.shape) <= 2:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:  # 3D
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Visualize activity
        if len(network.shape) == 1:
            # 1D: Plot as bar chart
            positions = [mtu_positions[idx][0] for idx in mtu_activity.keys()]
            activity = list(mtu_activity.values())
            
            ax.bar(positions, activity)
            ax.set_xlabel("Position")
            ax.set_ylabel("Activity (Output Count)")
            
            # Draw connections as arcs
            max_activity = max(activity, default=1)
            for idx, connected_indices in connections.items():
                if idx in mtu_activity:
                    x1 = mtu_positions[idx][0]
                    for connected_idx in connected_indices:
                        if connected_idx in mtu_activity:
                            x2 = mtu_positions[connected_idx][0]
                            # Draw arc whose height is proportional to combined activity
                            act1 = mtu_activity[idx] / max_activity
                            act2 = mtu_activity.get(connected_idx, 0) / max_activity
                            height = 0.2 * (act1 + act2) * max_activity
                            self._draw_arc(ax, x1, 0, x2, 0, height=height)
        
        elif len(network.shape) == 2:
            # 2D: Plot as scatter with size indicating activity
            x = [mtu_positions[idx][1] for idx in mtu_activity.keys()]
            y = [mtu_positions[idx][0] for idx in mtu_activity.keys()]
            
            # Size proportional to activity
            sizes = [100 * mtu_activity[idx] for idx in mtu_activity.keys()]
            if not sizes:
                sizes = [100]  # Default if no activity
                
            # Color proportional to relative activity
            max_activity = max(mtu_activity.values(), default=1)
            colors = [mtu_activity[idx] / max_activity for idx in mtu_activity.keys()]
            
            scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="Relative Activity")
            
            # Draw connections as lines, with width proportional to combined activity
            for idx, connected_indices in connections.items():
                if idx in mtu_activity:
                    x1, y1 = mtu_positions[idx][1], mtu_positions[idx][0]
                    for connected_idx in connected_indices:
                        if connected_idx in mtu_activity:
                            x2, y2 = mtu_positions[connected_idx][1], mtu_positions[connected_idx][0]
                            # Line width proportional to combined activity
                            act1 = mtu_activity[idx] / max_activity
                            act2 = mtu_activity.get(connected_idx, 0) / max_activity
                            lw = 1 + 3 * (act1 + act2) / 2
                            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=lw)
            
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_xlim(-1, network.shape[1])
            ax.set_ylim(-1, network.shape[0])
            
        else:  # 3D
            # 3D: Plot as scatter in 3D space
            x = [mtu_positions[idx][1] for idx in mtu_activity.keys()]
            y = [mtu_positions[idx][0] for idx in mtu_activity.keys()]
            z = [mtu_positions[idx][2] for idx in mtu_activity.keys()]
            
            # Size proportional to activity
            sizes = [100 * mtu_activity[idx] for idx in mtu_activity.keys()]
            if not sizes:
                sizes = [100]  # Default if no activity
                
            # Color proportional to relative activity
            max_activity = max(mtu_activity.values(), default=1)
            colors = [mtu_activity[idx] / max_activity for idx in mtu_activity.keys()]
            
            scatter = ax.scatter(x, y, z, s=sizes, c=colors, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="Relative Activity")
            
            # Draw connections as lines
            for idx, connected_indices in connections.items():
                if idx in mtu_activity:
                    x1, y1, z1 = (mtu_positions[idx][1], mtu_positions[idx][0], mtu_positions[idx][2])
                    for connected_idx in connected_indices:
                        if connected_idx in mtu_activity:
                            x2, y2, z2 = (mtu_positions[connected_idx][1],
                                         mtu_positions[connected_idx][0],
                                         mtu_positions[connected_idx][2])
                            # Line width proportional to combined activity
                            act1 = mtu_activity[idx] / max_activity
                            act2 = mtu_activity.get(connected_idx, 0) / max_activity
                            lw = 1 + 3 * (act1 + act2) / 2
                            ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.3, linewidth=lw)
            
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.set_xlim(-1, network.shape[1])
            ax.set_ylim(-1, network.shape[0])
            ax.set_zlim(-1, network.shape[2])
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"MTU Network Activity Over {time_steps} Time Steps")
        
        fig.tight_layout()
        self._figures['network_activity'] = fig
        return fig
    
    def _draw_arc(self, ax, x1, y1, x2, y2, height=1.0):
        """Draw a arc between two points.
        
        Args:
            ax: Matplotlib axis
            x1, y1: Start point
            x2, y2: End point
            height: Height of the arc
        """
        import matplotlib.patches as patches
        import numpy as np
        
        # Calculate arc parameters
        dx = x2 - x1
        dy = y2 - y1
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Center of arc
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Parameters for the elliptical arc
        width = dist / 2
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Create and add the arc
        arc = patches.Arc((cx, cy), width, height, angle=angle, 
                         theta1=0, theta2=180, linewidth=1, color='black', alpha=0.3)
        ax.add_patch(arc)
    
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
