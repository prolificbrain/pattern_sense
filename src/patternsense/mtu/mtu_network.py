"""MTU Network implementation for UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

from .mtu import MinimalThinkingUnit
from ..trits.tryte import Tryte
from ..field.data_field import DataField
from ..field.energy_field import EnergyField


@dataclass
class MTUNetwork:
    """Represents a network of Minimal Thinking Units (MTUs) in the UNIFIED system.
    
    The MTU Network manages multiple MTUs and their connections, facilitating
    the flow of information through the field-based substrate. It orchestrates
    the processing of tryte patterns and emergent field dynamics.
    
    Attributes:
        shape: Dimensions of the MTU grid (2D or 3D)
        mtu_density: Density of MTUs in the grid (0.0-1.0)
        connections_per_mtu: Average number of connections per MTU
        connection_radius: Radius for connecting MTUs
        connection_density: Density of connections between MTUs
        data_field: Data field for communication between MTUs
        energy_field: Energy field powering the computation
        enable_hebbian_learning: Whether to enable Hebbian learning
        enable_pattern_memory: Whether to enable pattern memory
        enable_adaptive_learning: Whether to enable adaptive learning
        structured_layers: Number of layers for structured networks
        substrate: Optional substrate for the network
        field_simulator: Optional field simulator for the network
    """
    
    shape: Tuple[int, ...]
    mtu_density: float = 0.2
    connections_per_mtu: int = 4
    connection_radius: float = 3.0
    connection_density: float = 0.5
    data_field: Optional[DataField] = None
    energy_field: Optional[EnergyField] = None
    enable_hebbian_learning: bool = True
    enable_pattern_memory: bool = False
    enable_adaptive_learning: bool = False
    structured_layers: int = 0  # Number of layers for structured networks
    substrate: Optional[Any] = None
    field_simulator: Optional[Any] = None
    _mtus: List[MinimalThinkingUnit] = field(default_factory=list, init=False, repr=False)
    _connections: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)
    _connection_strengths: Dict[Tuple[int, int], float] = field(default_factory=dict, init=False, repr=False)
    _coordinates: Dict[int, Tuple[int, ...]] = field(default_factory=dict, init=False, repr=False)
    _active_mtus: List[int] = field(default_factory=list, init=False, repr=False)
    _output_history: List[Dict[int, Tryte]] = field(default_factory=list, init=False, repr=False)
    _hebbian_learning: Optional[Any] = field(default=None, init=False, repr=False)
    _adaptive_learning: Optional[Any] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize the MTU network."""
        # Create data and energy fields if not provided
        if self.data_field is None and self.field_simulator is None:
            self.data_field = DataField(shape=self.shape)
        elif self.field_simulator is not None:
            # Extract data field from simulator
            self.data_field = self.field_simulator.data_fields[0] if self.field_simulator.data_fields else None
        
        if self.energy_field is None and self.field_simulator is None:
            self.energy_field = EnergyField(shape=self.shape)
        elif self.field_simulator is not None:
            # Extract energy field from simulator
            self.energy_field = self.field_simulator.energy_field
        
        # Initialize Hebbian learning if enabled
        if self.enable_hebbian_learning:
            from ..mtu.learning import HebbianLearning
            self._hebbian_learning = HebbianLearning()
            
        # Initialize adaptive learning if enabled
        if self.enable_adaptive_learning:
            from ..mtu.learning import AdaptiveLearning
            self._adaptive_learning = AdaptiveLearning()
        else:
            self._adaptive_learning = None
            
        # Create MTUs based on density and shape
        self._generate_mtus()
        
        # Create connections between MTUs
        if self.structured_layers > 0:
            self._create_layered_connections()
        else:
            self._create_connections()
    
    def _generate_mtus(self) -> None:
        """Create MTUs based on density and shape."""
        # Option 1: Create MTUs at random positions
        if self.mtu_density > 0:
            # Calculate total number of positions in the grid
            total_positions = np.prod(self.shape)
            
            # Calculate number of MTUs to create
            num_mtus = int(total_positions * self.mtu_density)
            
            # Generate random positions for MTUs
            positions = set()
            while len(positions) < num_mtus:
                # Create random coordinates
                coords = tuple(np.random.randint(0, dim) for dim in self.shape)
                positions.add(coords)
            
            # Create MTUs at these positions
            for i, position in enumerate(positions):
                mtu = MinimalThinkingUnit(position=position, dimensions=len(self.shape))
                self._mtus.append(mtu)
                self._coordinates[i] = position
    
    def _create_layered_connections(self) -> None:
        """Create connections between MTUs in a layered structure.
        
        This creates a feedforward network structure with layers, where MTUs
        in one layer connect primarily to MTUs in the next layer.
        """
        num_mtus = len(self._mtus)
        
        if num_mtus <= 1 or self.structured_layers <= 1:
            return
            
        # Determine layer boundaries
        # For 2D networks, we use the y-coordinate to distribute MTUs in layers
        if len(self.shape) >= 2:
            # Divide the y-axis into layers
            y_min, y_max = 0, self.shape[0] - 1
            layer_height = (y_max - y_min) / (self.structured_layers - 1)
            
            # Assign each MTU to a layer based on y-coordinate
            mtu_layers = {}
            for i, coord in self._coordinates.items():
                y = coord[1]
                layer = min(self.structured_layers - 1, 
                          max(0, int((y - y_min) / layer_height)))
                if layer not in mtu_layers:
                    mtu_layers[layer] = []
                mtu_layers[layer].append(i)
            
            # Create connections between layers
            for layer in range(self.structured_layers - 1):
                if layer not in mtu_layers or layer + 1 not in mtu_layers:
                    continue
                    
                current_layer_mtus = mtu_layers[layer]
                next_layer_mtus = mtu_layers[layer + 1]
                
                for i in current_layer_mtus:
                    self._connections[i] = []
                    position_i = self._coordinates[i]
                    
                    # Calculate distances to MTUs in the next layer
                    distances = []
                    for j in next_layer_mtus:
                        position_j = self._coordinates[j]
                        distance = np.sqrt(sum((pi - pj) ** 2 for pi, pj in zip(position_i, position_j)))
                        distances.append((j, distance))
                    
                    # Sort by distance
                    distances.sort(key=lambda x: x[1])
                    
                    # Connect to closest MTUs in the next layer
                    connections_per_layer = max(1, self.connections_per_mtu // (self.structured_layers - 1))
                    for j, dist in distances[:connections_per_layer]:
                        if dist <= self.connection_radius:
                            self._connections[i].append(j)
        else:
            # For 1D networks, just divide MTUs evenly into layers
            mtus_per_layer = max(1, num_mtus // self.structured_layers)
            mtu_layers = {}
            
            for i in range(num_mtus):
                layer = min(self.structured_layers - 1, i // mtus_per_layer)
                if layer not in mtu_layers:
                    mtu_layers[layer] = []
                mtu_layers[layer].append(i)
            
            # Create connections between layers
            for layer in range(self.structured_layers - 1):
                if layer not in mtu_layers or layer + 1 not in mtu_layers:
                    continue
                    
                current_layer_mtus = mtu_layers[layer]
                next_layer_mtus = mtu_layers[layer + 1]
                
                for i in current_layer_mtus:
                    self._connections[i] = []
                    # Connect to a random subset of MTUs in the next layer
                    connections_per_layer = max(1, self.connections_per_mtu // (self.structured_layers - 1))
                    for j in np.random.choice(next_layer_mtus, 
                                             size=min(connections_per_layer, len(next_layer_mtus)), 
                                             replace=False):
                        self._connections[i].append(j)
        
        # Initialize connection strengths with default values
        self._connection_strengths = {}
        for source, targets in self._connections.items():
            for target in targets:
                self._connection_strengths[(source, target)] = 0.5  # Default strength
    
    def _create_connections(self) -> None:
        """Create connections between MTUs."""
        num_mtus = len(self._mtus)
        
        if num_mtus <= 1:
            return
        
        # For each MTU, create connections to nearby MTUs
        for i in range(num_mtus):
            self._connections[i] = []
            position_i = self._coordinates[i]
            
            # Calculate distances to all other MTUs
            distances = []
            for j in range(num_mtus):
                if i == j:
                    continue
                
                position_j = self._coordinates[j]
                distance = np.sqrt(sum((pi - pj) ** 2 for pi, pj in zip(position_i, position_j)))
                distances.append((j, distance))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Connect to closest MTUs up to connections_per_mtu
            for j, _ in distances[:self.connections_per_mtu]:
                self._connections[i].append(j)
        
        # Initialize connection strengths with default values
        if self.enable_hebbian_learning and self._hebbian_learning:
            self._connection_strengths = self._hebbian_learning.initialize_connections(self._connections)
        else:
            # Default initialization without Hebbian learning
            for source, targets in self._connections.items():
                for target in targets:
                    self._connection_strengths[(source, target)] = 0.5  # Default strength
    
    def add_mtu(self, mtu: MinimalThinkingUnit) -> int:
        """Add a new MTU to the network.
        
        Args:
            mtu: MTU to add
            
        Returns:
            Index of the new MTU
        """
        mtu_idx = len(self._mtus)
        self._mtus.append(mtu)
        self._coordinates[mtu_idx] = mtu.position
        self._connections[mtu_idx] = []
        
        return mtu_idx
    
    def connect_mtu(self, source_idx: int, target_idx: int, weight: float = 0.5) -> None:
        """Connect two MTUs in the network.
        
        Args:
            source_idx: Index of source MTU
            target_idx: Index of target MTU
            weight: Initial connection weight
        """
        if source_idx < 0 or source_idx >= len(self._mtus) or target_idx < 0 or target_idx >= len(self._mtus):
            raise ValueError(f"MTU indices out of range: {source_idx}, {target_idx}")
        
        if target_idx not in self._connections[source_idx]:
            self._connections[source_idx].append(target_idx)
        
        self._connection_strengths[(source_idx, target_idx)] = weight
    
    def disconnect_mtu(self, source_idx: int, target_idx: int) -> None:
        """Remove connection between two MTUs.
        
        Args:
            source_idx: Index of source MTU
            target_idx: Index of target MTU
        """
        if source_idx in self._connections and target_idx in self._connections[source_idx]:
            self._connections[source_idx].remove(target_idx)
            
            # Remove connection strength
            if (source_idx, target_idx) in self._connection_strengths:
                del self._connection_strengths[(source_idx, target_idx)]
    
    def inject_input(self, tryte: Tryte, position: Optional[Tuple[int, ...]] = None) -> None:
        """Inject a tryte into the network at a specific position or random position.
        
        Args:
            tryte: Tryte to inject
            position: Position in the grid to inject at (None for random)
        """
        if not self._mtus:
            return
        
        if position is None:
            # Choose a random MTU
            mtu_idx = np.random.randint(0, len(self._mtus))
        else:
            # Find the closest MTU to the given position
            closest_idx = 0
            min_distance = float('inf')
            
            for idx, coords in self._coordinates.items():
                distance = np.sqrt(sum((p - c) ** 2 for p, c in zip(position, coords)))
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = idx
            
            mtu_idx = closest_idx
        
        # Inject the tryte into the selected MTU
        self._mtus[mtu_idx].receive_input(tryte)
        
        # Also inject the tryte pattern into the data field at the MTU's position
        if self.data_field is not None:
            pattern = tryte.to_wave_pattern(self.shape, self._coordinates[mtu_idx])
            self.data_field.inject_pattern(pattern, self._coordinates[mtu_idx])
        
        # Add to active MTUs list
        if mtu_idx not in self._active_mtus:
            self._active_mtus.append(mtu_idx)
    
    def _update_fields(self) -> None:
        """Update data and energy fields."""
        # If using field simulator, let it handle the updates
        if self.field_simulator is not None:
            self.field_simulator.step()
            return
            
        # Otherwise update fields directly
        if self.energy_field is not None:
            self.energy_field.update(dt=0.1)
        
        if self.data_field is not None and self.energy_field is not None:
            self.data_field.update(dt=0.1, energy_field=self.energy_field.field)
        elif self.data_field is not None:
            self.data_field.update(dt=0.1)
    
    def _update_mtu_local_fields(self) -> None:
        """Update the local field state for each MTU."""
        for i, mtu in enumerate(self._mtus):
            # Get MTU position
            position = self._coordinates[i]
            
            # Extract local field around MTU
            local_size = mtu.state_field_size
            half_size = local_size // 2
            
            if self.data_field is None:
                # No data field available, just update with zeros
                mtu.update_local_field(np.zeros((local_size,) * len(self.shape)))
                continue
            
            # Calculate extraction bounds
            start_idx = tuple(max(0, p - half_size) for p in position)
            end_idx = tuple(min(dim, p + half_size + 1) for p, dim in zip(position, self.shape))
            
            # Extract local data field
            slices = tuple(slice(s, e) for s, e in zip(start_idx, end_idx))
            local_field = self.data_field.field[slices]
            
            # Pad if necessary to maintain consistent size
            pad_widths = []
            for dim, (s, e, p) in enumerate(zip(start_idx, end_idx, position)):
                pad_left = max(0, half_size - (p - s))
                pad_right = max(0, half_size - (e - p - 1))
                pad_widths.append((pad_left, pad_right))
            
            if any(sum(pw) > 0 for pw in pad_widths):
                local_field = np.pad(local_field, pad_widths, mode='constant')
            
            # Ensure local field has correct shape
            if local_field.shape != (local_size,) * len(self.shape):
                # Resize to correct shape
                from scipy.ndimage import zoom
                zoom_factors = tuple(local_size / s for s in local_field.shape)
                local_field = zoom(local_field, zoom_factors, order=1)
            
            # Update MTU's local field state
            mtu.update_local_field(local_field)
    
    def _propagate_outputs(self, outputs: List[Tuple[int, Tryte]]) -> None:
        """Propagate outputs from MTUs to their connected targets.
        
        Args:
            outputs: List of (MTU index, output tryte) pairs
        """
        for source_idx, tryte in outputs:
            # Get connected MTUs
            if source_idx not in self._connections:
                continue
                
            for target_idx in self._connections[source_idx]:
                # Get connection strength
                strength = self._connection_strengths.get((source_idx, target_idx), 0.5)
                
                # Only propagate if connection is strong enough
                if strength > 0.2:
                    # Scale tryte energy by connection strength
                    scaled_tryte = tryte.copy()
                    scaled_tryte.scale_energy(strength)
                    
                    # Pass to target MTU
                    self._mtus[target_idx].receive_input(scaled_tryte)
                    
                    # Add target to active list
                    if target_idx not in self._active_mtus:
                        self._active_mtus.append(target_idx)
    
    def step(self) -> List[Tuple[int, Tryte]]:
        """Run one step of the network simulation.
        
        Returns:
            List of (MTU index, output tryte) pairs for MTUs that produced output
        """
        # Update fields
        self._update_fields()
        
        # Update MTU local fields
        self._update_mtu_local_fields()
        
        # Process MTUs and collect outputs
        outputs = []
        mtu_outputs = {}  # Track outputs for Hebbian learning
        
        for idx in self._active_mtus.copy():
            mtu = self._mtus[idx]
            output = mtu.process()
            
            if output is not None:
                outputs.append((idx, output))
                mtu_outputs[idx] = output.get_average_energy()  # Use average energy as output strength
            elif not mtu._input_buffer:  # No input and no output
                # MTU is no longer active
                self._active_mtus.remove(idx)
        
        # Apply Hebbian learning if enabled
        if self.enable_hebbian_learning and self._hebbian_learning and mtu_outputs:
            self._connection_strengths = self._hebbian_learning.update_connections(
                self._active_mtus, mtu_outputs, self._connections
            )
        
        # Store output history
        output_dict = {idx: tryte for idx, tryte in outputs}
        if output_dict:
            self._output_history.append(output_dict)
            if len(self._output_history) > 100:
                self._output_history.pop(0)
        
        # Propagate outputs to connected MTUs
        self._propagate_outputs(outputs)
        
        return outputs
    
    def run_steps(self, num_steps: int) -> List[List[Tuple[int, Tryte]]]:
        """Run multiple simulation steps.
        
        Args:
            num_steps: Number of steps to run
            
        Returns:
            List of outputs for each step
        """
        outputs = []
        for _ in range(num_steps):
            step_outputs = self.step()
            outputs.append(step_outputs)
        return outputs
    
    def get_active_mtu_count(self) -> int:
        """Get the number of currently active MTUs.
        
        Returns:
            Number of active MTUs
        """
        return len(self._active_mtus)
    
    def get_connection_strength(self, source_idx: int, target_idx: int) -> float:
        """Get the strength of a connection between two MTUs.
        
        Args:
            source_idx: Index of source MTU
            target_idx: Index of target MTU
            
        Returns:
            Connection strength between 0 and 1
        """
        return self._connection_strengths.get((source_idx, target_idx), 0.0)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning process.
        
        Returns:
            Dictionary with learning statistics
        """
        if not self.enable_hebbian_learning or self._hebbian_learning is None:
            return {'enabled': False}
        
        # Calculate average connection strength
        if self._connection_strengths:
            avg_strength = sum(self._connection_strengths.values()) / len(self._connection_strengths)
        else:
            avg_strength = 0.0
        
        # Count strong connections (>0.7)
        strong_connections = sum(1 for s in self._connection_strengths.values() if s > 0.7)
        
        # Calculate pattern memory stats for all MTUs
        total_patterns = 0
        for mtu in self._mtus:
            stats = mtu.get_pattern_memory_stats()
            if stats['enabled']:
                total_patterns += stats['pattern_count']
        
        return {
            'enabled': True,
            'avg_connection_strength': avg_strength,
            'strong_connection_count': strong_connections,
            'total_patterns_learned': total_patterns
        }
    
    def visualize(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualize the network state.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with visualization
        """
        if len(self.shape) > 3:
            raise ValueError("Cannot visualize networks with more than 3 dimensions")
        
        fig = plt.figure(figsize=figsize)
        
        # For 1D network
        if len(self.shape) == 1:
            # 1D network: plot data field, energy field, and MTU positions
            gs = plt.GridSpec(3, 1)
            
            # Plot data field
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(self.data_field.field)
            ax1.set_title("Data Field")
            
            # Plot energy field
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(self.energy_field.field)
            ax2.set_title("Energy Field")
            
            # Plot MTU positions
            ax3 = fig.add_subplot(gs[2, 0])
            for idx, coords in self._coordinates.items():
                x = coords[0]
                is_active = idx in self._active_mtus
                color = 'r' if is_active else 'b'
                ax3.scatter(x, 0, color=color, s=50)
                
                # Draw connections
                for connected_idx in self._connections.get(idx, []):
                    connected_x = self._coordinates[connected_idx][0]
                    ax3.plot([x, connected_x], [0, 0], 'k-', alpha=0.3)
            
            ax3.set_title("MTU Network")
            ax3.set_yticks([])
            ax3.set_xlabel("Position")
        
        # For 2D network
        elif len(self.shape) == 2:
            if len(self.shape) == 2:
                # Use GridSpec for layout
                gs = plt.GridSpec(2, 2, width_ratios=[2, 1])
                
                # Plot data field
                ax1 = fig.add_subplot(gs[0, 0])
                im1 = ax1.imshow(self.data_field.field, cmap='viridis', origin='lower')
                plt.colorbar(im1, ax=ax1, label="Data Value")
                ax1.set_title("Data Field")
                
                # Plot energy field
                ax2 = fig.add_subplot(gs[1, 0])
                im2 = ax2.imshow(self.energy_field.field, cmap='inferno', origin='lower')
                plt.colorbar(im2, ax=ax2, label="Energy")
                ax2.set_title("Energy Field")
                
                # Plot MTU network
                ax3 = fig.add_subplot(gs[:, 1])
                
                # Plot MTU positions
                for idx, coords in self._coordinates.items():
                    x, y = coords
                    is_active = idx in self._active_mtus
                    color = 'r' if is_active else 'b'
                    ax3.scatter(x, y, color=color, s=50)
                
                # Draw connections
                for idx, connected_indices in self._connections.items():
                    x1, y1 = self._coordinates[idx]
                    for connected_idx in connected_indices:
                        x2, y2 = self._coordinates[connected_idx]
                        ax3.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
                
                ax3.set_xlim(-1, self.shape[0])
                ax3.set_ylim(-1, self.shape[1])
                ax3.set_title("MTU Network")
                ax3.set_aspect('equal')
        
        # For 3D network
        else:  # len(self.shape) == 3
            from mpl_toolkits.mplot3d import Axes3D
            
            # Plot MTU network
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot MTU positions
            for idx, coords in self._coordinates.items():
                x, y, z = coords
                is_active = idx in self._active_mtus
                color = 'r' if is_active else 'b'
                ax.scatter(x, y, z, color=color, s=50)
            
            # Draw connections
            for idx, connected_indices in self._connections.items():
                x1, y1, z1 = self._coordinates[idx]
                for connected_idx in connected_indices:
                    x2, y2, z2 = self._coordinates[connected_idx]
                    ax.plot([x1, x2], [y1, y2], [z1, z2], 'k-', alpha=0.3)
            
            ax.set_xlim(-1, self.shape[0])
            ax.set_ylim(-1, self.shape[1])
            ax.set_zlim(-1, self.shape[2])
            ax.set_title("3D MTU Network")
        
        fig.tight_layout()
        return fig
    
    def get_mtus(self) -> List[MinimalThinkingUnit]:
        """Get all MTUs in the network.
        
        Returns:
            List of all MTUs
        """
        return self._mtus
    
    def get_mtu_positions(self) -> Dict[int, Tuple[int, ...]]:
        """Get positions of all MTUs.
        
        Returns:
            Dictionary mapping MTU indices to positions
        """
        return self._coordinates
    
    def get_connections(self) -> Dict[int, List[int]]:
        """Get connections between MTUs.
        
        Returns:
            Dictionary mapping MTU indices to lists of connected MTU indices
        """
        return self._connections
    
    def find_active_circuits(self) -> List[List[int]]:
        """Find active circuits in the network.
        
        An active circuit is a connected component of active MTUs.
        
        Returns:
            List of circuits, each represented as a list of MTU indices
        """
        if not self._active_mtus:
            return []
        
        # Create graph of active MTUs and their connections
        active_graph = {}
        for idx in self._active_mtus:
            active_graph[idx] = []
            for connected_idx in self._connections.get(idx, []):
                if connected_idx in self._active_mtus:
                    active_graph[idx].append(connected_idx)
        
        # Find connected components (circuits)
        circuits = []
        visited = set()
        
        def dfs(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in active_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for node in active_graph:
            if node not in visited:
                component = []
                dfs(node, component)
                circuits.append(component)
        
        return circuits
    
    def find_closest_mtu(self, position):
        """Find the MTU closest to a given position.
        
        Args:
            position: Position coordinates (x, y, ...)
            
        Returns:
            mtu_idx: Index of the closest MTU, or None if no MTUs exist
        """
        if not self._coordinates:
            return None
            
        # Calculate distances to all MTUs
        min_dist = float('inf')
        closest_idx = None
        
        for idx, coord in self._coordinates.items():
            # Calculate Euclidean distance
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(position, coord)))
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx
                
        return closest_idx
    
    def get_mtu_layer(self, mtu_idx):
        """Get the layer of a specific MTU.
        
        This is primarily useful for structured networks with defined layers.
        For non-structured networks, this will return 0 for all MTUs.
        
        Args:
            mtu_idx: Index of the MTU
            
        Returns:
            layer: Layer index (0 for input layer)
        """
        if mtu_idx not in self._coordinates:
            return 0
            
        # For 2D networks, we use the y-coordinate to determine the layer
        if len(self._coordinates[mtu_idx]) >= 2:
            # Map y-coordinate to a layer
            y_values = {coord[1] for coord in self._coordinates.values()}
            sorted_y = sorted(y_values)
            
            y = self._coordinates[mtu_idx][1]
            return sorted_y.index(y)
        
        return 0
    
    def get_mtu_index_in_layer(self, mtu_idx, layer):
        """Get the relative index of an MTU within its layer.
        
        This is useful for mapping output MTUs to specific classes/categories.
        
        Args:
            mtu_idx: Index of the MTU
            layer: Layer index
            
        Returns:
            idx_in_layer: Relative index within the layer
        """
        if mtu_idx not in self._coordinates:
            return 0
            
        # Get all MTUs in the specified layer
        mtu_indices_in_layer = [
            idx for idx in self._coordinates
            if self.get_mtu_layer(idx) == layer
        ]
        
        # Sort by x-coordinate (assuming 2D network)
        if len(self._coordinates[mtu_idx]) >= 2:
            sorted_indices = sorted(
                mtu_indices_in_layer,
                key=lambda idx: self._coordinates[idx][0]
            )
            
            if mtu_idx in sorted_indices:
                return sorted_indices.index(mtu_idx)
        
        return 0
    
    @property
    def num_layers(self):
        """Get the number of layers in the network.
        
        Returns:
            num_layers: Number of distinct layers
        """
        # For 2D networks, count unique y-coordinates
        if self._coordinates and len(next(iter(self._coordinates.values()))) >= 2:
            y_values = {coord[1] for coord in self._coordinates.values()}
            return len(y_values)
        
        return 1
    
    def reset(self):
        """Reset all MTUs to inactive state and clear buffers.
        
        This is useful between processing different inputs.
        """
        for mtu in self._mtus:
            mtu.clear_buffers()
            mtu._is_active = False
            mtu._state_field *= 0.1  # Reduce state without completely clearing
    
    @property
    def total_connections(self):
        """Get the total number of connections in the network.
        
        Returns:
            total: Total number of connections
        """
        total = 0
        for mtu_idx in self._connections:
            total += len(self._connections[mtu_idx])
        return total
    
    def hebbian_update(self, performance_metric=None):
        """Apply Hebbian learning to update connection strengths based on activity.
        
        This implements the principle that "neurons that fire together, wire together"
        by strengthening connections between co-active MTUs.
        
        Args:
            performance_metric: Optional performance metric for adaptive learning
        """
        if not self.enable_hebbian_learning or self._hebbian_learning is None:
            return
            
        # Update adaptive learning rate if enabled and performance metric provided
        learning_scale = 1.0
        if self.enable_adaptive_learning and self._adaptive_learning and performance_metric is not None:
            learning_scale = self._adaptive_learning.update_rate(performance_metric) / 0.01  # Normalize to base rate of 0.01
            
        # Find active MTUs
        active_mtus = [i for i, mtu in enumerate(self._mtus) if getattr(mtu, '_is_active', False)]
        
        # Apply Hebbian learning to connections between active MTUs
        for i in active_mtus:
            if i not in self._connections:
                continue
                
            for j in self._connections[i]:
                if j in active_mtus and (i, j) in self._connection_strengths:
                    # Strengthen connection between co-active MTUs
                    old_strength = self._connection_strengths[(i, j)]
                    # Apply learning rate scaling factor
                    new_strength = self._hebbian_learning.update_connection(
                        old_strength, True, learning_scale=learning_scale)
                    self._connection_strengths[(i, j)] = new_strength
                elif (i, j) in self._connection_strengths:
                    # Slightly weaken connections where target is not active
                    old_strength = self._connection_strengths[(i, j)]
                    # Apply learning rate scaling factor
                    new_strength = self._hebbian_learning.update_connection(
                        old_strength, False, learning_scale=learning_scale)
                    self._connection_strengths[(i, j)] = new_strength
