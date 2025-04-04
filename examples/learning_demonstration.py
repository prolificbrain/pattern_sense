#!/usr/bin/env python
"""
Learning demonstration for the UNIFIED Consciousness Engine.

This script demonstrates the learning capabilities of the UNIFIED Consciousness Engine,
showing how MTUs can learn and recognize patterns through Hebbian learning and
attractor dynamics.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.unified.substrate.manifold import HolographicManifold
from src.unified.substrate.metrics import MetricTensor, CurvatureField
from src.unified.substrate.field import SubstrateField
from src.unified.field.energy_field import EnergyField
from src.unified.field.data_field import DataField
from src.unified.field.field_simulator import FieldSimulator
from src.unified.trits.trit import Trit, TritState
from src.unified.trits.tryte import Tryte
from src.unified.trits.trit_patterns import create_pulse_pattern, create_wave_pattern
from src.unified.mtu.mtu import MinimalThinkingUnit
from src.unified.mtu.mtu_network import MTUNetwork
from src.unified.visualization.field_visualizer import FieldVisualizer
from src.unified.visualization.mtu_visualizer import MTUVisualizer
from src.unified.visualization.animation import create_network_animation


def create_simulation_environment(
    shape: Tuple[int, int] = (32, 32)
) -> Tuple[HolographicManifold, FieldSimulator, MTUNetwork]:
    """
    Create the basic simulation environment for the learning demonstration.
    
    Args:
        shape: Shape of the simulation grid
        
    Returns:
        Tuple of (manifold, field_simulator, mtu_network)
    """
    # Create a holographic manifold with a non-uniform curvature function
    def curvature_function(coords):
        # Get the shape of the input coordinates
        grid_shape = coords.shape[:-1]
        
        # Extract x and y coordinates
        x = coords[..., 0]
        y = coords[..., 1]
        
        # Primary central basin (negative curvature at origin)
        sq_dist = x**2 + y**2
        curvature = -0.5 * np.exp(-sq_dist / 0.5)
        
        # Add some secondary basins at specific locations
        for i in range(3):
            # Calculate offset from origin
            angle = i * 2.0 * np.pi / 3
            offset_x = 0.5 * np.cos(angle)
            offset_y = 0.5 * np.sin(angle)
            
            # Calculate square distance to this offset point
            local_sq_dist = (x - offset_x)**2 + (y - offset_y)**2
            
            # Add negative curvature (basin) at this location
            curvature -= 0.3 * np.exp(-local_sq_dist / 0.3)
        
        return curvature
    
    manifold = HolographicManifold(
        dimensions=2,
        time_dimensions=1,
        curvature_function=curvature_function
    )
    
    # Initialize the manifold grid
    manifold.initialize_grid(resolution=shape[0])
    
    # Create energy and data fields
    energy_field = EnergyField(shape=shape, dissipation_rate=0.05)
    data_field = DataField(shape=shape, persistence=0.90)
    
    # Initialize with some energy
    initial_energy = np.zeros(shape)
    initial_energy[12:20, 12:20] = 1.0  # Add energy to a central region
    energy_field._field = initial_energy.copy()  # Directly set the field
    
    # Create a field simulator
    simulator = FieldSimulator(
        manifold=manifold,
        energy_field=energy_field,
        data_fields=[data_field],
        dt=0.1  # Time step size
    )
    
    # Create an MTU network
    network = MTUNetwork(
        shape=shape,
        mtu_density=0.0,  # Start with 0 density, we'll add MTUs manually
        enable_hebbian_learning=True,
        substrate=manifold,
        field_simulator=simulator
    )
    
    return manifold, simulator, network


def create_structured_mtu_network(
    network: MTUNetwork,
    num_layers: int = 3,
    mtu_per_layer: int = 4
) -> None:
    """
    Create a structured MTU network with layers.
    
    Args:
        network: MTU network to populate
        num_layers: Number of layers in the network
        mtu_per_layer: Number of MTUs per layer
    """
    shape = network.shape
    
    # Calculate spacing along x-axis
    x_spacing = shape[0] // (mtu_per_layer + 1)
    
    # Calculate spacing along y-axis
    y_spacing = shape[1] // (num_layers + 1)
    
    # Create MTUs in layers
    layer_mtus = []
    
    for layer in range(num_layers):
        layer_y = (layer + 1) * y_spacing
        layer_mtu_indices = []
        
        for i in range(mtu_per_layer):
            x_pos = (i + 1) * x_spacing
            
            # Create MTU
            mtu = MinimalThinkingUnit(
                position=(x_pos, layer_y),
                dimensions=2,
                state_field_size=5,
                learning_rate=0.3 if layer == 0 else 0.2,  # Input layer learns faster
                pattern_memory_enabled=True,
                attractor_dynamics_enabled=True
            )
            
            # Add to network
            mtu_idx = network.add_mtu(mtu)
            layer_mtu_indices.append(mtu_idx)
        
        layer_mtus.append(layer_mtu_indices)
    
    # Connect MTUs between layers
    for layer in range(num_layers - 1):
        for i, source_idx in enumerate(layer_mtus[layer]):
            # Connect to all MTUs in the next layer
            for target_idx in layer_mtus[layer + 1]:
                # Initial weight varies based on proximity
                source_pos = network._coordinates[source_idx]
                target_pos = network._coordinates[target_idx]
                distance = np.sqrt(sum((s - t) ** 2 for s, t in zip(source_pos, target_pos)))
                max_distance = np.sqrt(sum(s ** 2 for s in network.shape))
                weight = 0.5 * (1.0 - distance / max_distance)
                
                network.connect_mtu(source_idx, target_idx, weight=weight)


def create_test_patterns() -> List[Tryte]:
    """
    Create test patterns for learning.
    
    Returns:
        List of test patterns as Trytes
    """
    patterns = []
    
    # Pattern 1: Simple positive pulse
    trits1 = [
        Trit(state=TritState.POSITIVE, energy=1.0, phase=0.0),
        Trit(state=TritState.POSITIVE, energy=0.8, phase=0.0),
        Trit(state=TritState.NEUTRAL, energy=0.3, phase=0.0)
    ]
    patterns.append(Tryte(trits=trits1, coherence=0.9))
    
    # Pattern 2: Simple negative pulse
    trits2 = [
        Trit(state=TritState.NEGATIVE, energy=1.0, phase=0.0),
        Trit(state=TritState.NEGATIVE, energy=0.8, phase=0.0),
        Trit(state=TritState.NEUTRAL, energy=0.3, phase=0.0)
    ]
    patterns.append(Tryte(trits=trits2, coherence=0.9))
    
    # Pattern 3: Mixed positive/negative
    trits3 = [
        Trit(state=TritState.POSITIVE, energy=1.0, phase=0.0),
        Trit(state=TritState.NEGATIVE, energy=0.8, phase=0.0),
        Trit(state=TritState.POSITIVE, energy=0.6, phase=0.0)
    ]
    patterns.append(Tryte(trits=trits3, coherence=0.8))
    
    # Pattern 4: Triangle wave (more complex)
    trits4 = [
        Trit(state=TritState.POSITIVE, energy=1.0, phase=0.0),
        Trit(state=TritState.NEGATIVE, energy=1.0, phase=np.pi/3),
        Trit(state=TritState.POSITIVE, energy=1.0, phase=2*np.pi/3)
    ]
    patterns.append(Tryte(trits=trits4, coherence=0.7))
    
    return patterns


def train_network(
    network: MTUNetwork,
    patterns: List[Tryte],
    training_cycles: int = 10,
    steps_per_pattern: int = 20
) -> Dict[str, List[Any]]:
    """
    Train the network on the provided patterns.
    
    Args:
        network: MTU network to train
        patterns: List of input patterns
        training_cycles: Number of training cycles
        steps_per_pattern: Steps to run for each pattern
        
    Returns:
        Dictionary with training statistics
    """
    # Training statistics
    stats = {
        'connection_strengths': [],
        'active_mtus': [],
        'outputs_per_step': [],
        'pattern_recognition': [0] * len(patterns)
    }
    
    # Run training cycles
    for cycle in range(training_cycles):
        print(f"Training cycle {cycle+1}/{training_cycles}")
        
        # For each pattern
        for i, pattern in enumerate(patterns):
            # Inject pattern into first layer MTUs
            input_layer_mtus = [idx for idx, mtu in enumerate(network._mtus) 
                             if network._coordinates[idx][1] == network._coordinates[0][1]]
            
            for mtu_idx in input_layer_mtus:
                network.inject_input(pattern, network._coordinates[mtu_idx])
            
            # Run steps
            outputs_for_pattern = []
            for step in range(steps_per_pattern):
                step_outputs = network.step()
                outputs_for_pattern.append(len(step_outputs))
                
                # Check for pattern recognition (output from last layer)
                last_layer_y = max(coord[1] for coord in network._coordinates.values())
                last_layer_mtus = [idx for idx, mtu in enumerate(network._mtus) 
                                if network._coordinates[idx][1] == last_layer_y]
                
                for mtu_idx, _ in step_outputs:
                    if mtu_idx in last_layer_mtus:
                        stats['pattern_recognition'][i] += 1
                
            # Record statistics
            stats['outputs_per_step'].append(outputs_for_pattern)
            stats['active_mtus'].append(network.get_active_mtu_count())
                
        # Record connection strengths after each cycle
        avg_strength = sum(network._connection_strengths.values()) / max(1, len(network._connection_strengths))
        stats['connection_strengths'].append(avg_strength)
    
    return stats


def test_network(
    network: MTUNetwork,
    patterns: List[Tryte],
    steps_per_pattern: int = 20,
    include_variants: bool = True
) -> Dict[str, List[Any]]:
    """
    Test the network with patterns and their variants.
    
    Args:
        network: Trained MTU network
        patterns: Original training patterns
        steps_per_pattern: Steps to run for each pattern
        include_variants: Whether to include pattern variants
        
    Returns:
        Dictionary with test results
    """
    test_results = {
        'pattern_activations': [],
        'pattern_recognition': [],
        'response_times': []
    }
    
    # Create test set
    test_patterns = list(patterns)  # Original patterns
    
    # Add variants if requested
    if include_variants:
        for pattern in patterns:
            # Create a noisy variant (reduce coherence)
            noisy_pattern = pattern.copy()
            noisy_pattern.coherence *= 0.7
            test_patterns.append(noisy_pattern)
            
            # Create a weaker variant (reduce energy)
            weak_pattern = pattern.copy()
            weak_pattern.scale_energy(0.6)
            test_patterns.append(weak_pattern)
    
    # Run tests
    for i, pattern in enumerate(test_patterns):
        print(f"Testing pattern {i+1}/{len(test_patterns)}")
        
        # Inject pattern into first layer MTUs
        input_layer_mtus = [idx for idx, mtu in enumerate(network._mtus) 
                         if network._coordinates[idx][1] == network._coordinates[0][1]]
        
        for mtu_idx in input_layer_mtus:
            network.inject_input(pattern, network._coordinates[mtu_idx])
        
        # Track response
        recognized = False
        response_time = steps_per_pattern  # Default if not recognized
        
        # Run steps and check for recognition
        activations = []
        for step in range(steps_per_pattern):
            step_outputs = network.step()
            active_count = network.get_active_mtu_count()
            activations.append(active_count)
            
            # Check for pattern recognition (output from last layer)
            if not recognized:
                last_layer_y = max(coord[1] for coord in network._coordinates.values())
                last_layer_mtus = [idx for idx, mtu in enumerate(network._mtus) 
                                if network._coordinates[idx][1] == last_layer_y]
                
                for mtu_idx, _ in step_outputs:
                    if mtu_idx in last_layer_mtus:
                        recognized = True
                        response_time = step
                        break
        
        # Record results
        test_results['pattern_activations'].append(activations)
        test_results['pattern_recognition'].append(recognized)
        test_results['response_times'].append(response_time)
    
    return test_results


def visualize_training_results(stats: Dict[str, List[Any]], filename: str = "training_results.png") -> None:
    """
    Visualize training results.
    
    Args:
        stats: Training statistics
        filename: Output filename for the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot connection strength evolution
    axes[0, 0].plot(stats['connection_strengths'])
    axes[0, 0].set_title('Average Connection Strength')
    axes[0, 0].set_xlabel('Training Cycle')
    axes[0, 0].set_ylabel('Strength')
    
    # Plot active MTUs over training
    axes[0, 1].plot(stats['active_mtus'])
    axes[0, 1].set_title('Active MTUs')
    axes[0, 1].set_xlabel('Pattern Presentation')
    axes[0, 1].set_ylabel('Count')
    
    # Plot outputs per step for a subset of patterns
    for i in range(min(4, len(stats['outputs_per_step']))):
        axes[1, 0].plot(stats['outputs_per_step'][i], label=f'Pattern {i+1}')
    axes[1, 0].set_title('Outputs per Step')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Output Count')
    axes[1, 0].legend()
    
    # Plot pattern recognition counts
    patterns = range(1, len(stats['pattern_recognition']) + 1)
    axes[1, 1].bar(patterns, stats['pattern_recognition'])
    axes[1, 1].set_title('Pattern Recognition Counts')
    axes[1, 1].set_xlabel('Pattern')
    axes[1, 1].set_ylabel('Recognition Count')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_test_results(test_results: Dict[str, List[Any]], filename: str = "test_results.png") -> None:
    """
    Visualize test results.
    
    Args:
        test_results: Test results
        filename: Output filename for the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot pattern activations
    for i, activations in enumerate(test_results['pattern_activations']):
        if i < 4:  # Only show first few patterns for clarity
            axes[0].plot(activations, label=f'Pattern {i+1}')
            
    axes[0].set_title('MTU Activations During Testing')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Active MTUs')
    axes[0].legend()
    
    # Plot recognition rate and response time
    pattern_numbers = list(range(1, len(test_results['pattern_recognition']) + 1))
    
    ax1 = axes[1]
    ax1.bar(pattern_numbers, test_results['response_times'], alpha=0.7)
    ax1.set_title('Pattern Recognition Results')
    ax1.set_xlabel('Pattern')
    ax1.set_ylabel('Response Time (steps)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Add recognition markers as a secondary axis
    ax2 = ax1.twinx()
    recognition = [1 if r else 0 for r in test_results['pattern_recognition']]
    ax2.scatter(pattern_numbers, recognition, color='red', s=100, marker='o')
    ax2.set_ylabel('Recognized (1=yes, 0=no)', color='red')
    ax2.set_yticks([0, 1])
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def visualize_network_learning(network: MTUNetwork, filename: str = "mtu_network_learned.png") -> None:
    """
    Visualize the network with connection strengths after learning.
    
    Args:
        network: Trained MTU network
        filename: Output filename for the plot
    """
    # Use the network's own visualization method
    fig = network.visualize(figsize=(12, 8))
    
    plt.savefig(filename)
    plt.close()


def run_learning_demonstration():
    """
    Run the complete learning demonstration.
    
    Returns:
        Statistics and results from the demonstration
    """
    print("UNIFIED Consciousness Engine - Learning Demonstration")
    print("==================================================")
    
    print("\nInitializing simulation environment...")
    manifold, simulator, network = create_simulation_environment()
    
    print("Creating structured MTU network...")
    create_structured_mtu_network(network)
    
    print("\nNetwork Structure:")
    print(f"- {len(network._mtus)} MTUs created")
    print(f"- {len(network._connections)} MTUs with connections")
    print(f"- {sum(len(targets) for targets in network._connections.values())} total connections")
    
    # Visualize initial network
    visualize_network_learning(network, "mtu_network_initial.png")
    print("Initial network visualization saved as 'mtu_network_initial.png'")
    
    print("\nCreating test patterns...")
    patterns = create_test_patterns()
    print(f"- {len(patterns)} patterns created")
    
    print("\nTraining network...")
    training_stats = train_network(network, patterns)
    
    # Visualize training results
    visualize_training_results(training_stats)
    print("Training results visualization saved as 'training_results.png'")
    
    # Visualize learned network
    visualize_network_learning(network, "mtu_network_learned.png")
    print("Learned network visualization saved as 'mtu_network_learned.png'")
    
    print("\nTesting network...")
    test_results = test_network(network, patterns)
    
    # Visualize test results
    visualize_test_results(test_results)
    print("Test results visualization saved as 'test_results.png'")
    
    print("\nLearning demonstration complete.")
    
    return training_stats, test_results


if __name__ == "__main__":
    run_learning_demonstration()
