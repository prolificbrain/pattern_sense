#!/usr/bin/env python
"""Basic simulation example for the UNIFIED Consciousness Engine.

This example demonstrates the core components of the system working together,
including the holographic substrate, field dynamics, and trit-based computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unified.substrate.manifold import HolographicManifold
from src.unified.field.energy_field import EnergyField
from src.unified.field.data_field import DataField
from src.unified.field.field_simulator import FieldSimulator
from src.unified.trits.trit import Trit, TritState
from src.unified.trits.tryte import Tryte
from src.unified.trits.trit_patterns import create_pulse_pattern, create_wave_pattern, create_tryte_pattern
from src.unified.mtu.mtu import MinimalThinkingUnit
from src.unified.mtu.mtu_network import MTUNetwork
from src.unified.visualization.field_visualizer import FieldVisualizer
from src.unified.visualization.mtu_visualizer import MTUVisualizer
from src.unified.visualization.animation import create_network_animation, create_interactive_animation


def curvature_function(coords):
    """Define a non-uniform curvature for the holographic substrate."""
    # Create a curved space with a central basin (attractor)
    center = np.zeros(coords.shape[-1])
    sq_dist = np.sum((coords - center) ** 2, axis=-1)
    return -0.5 * np.exp(-sq_dist / 0.5)


def run_basic_simulation():
    """Run a basic simulation of the UNIFIED Consciousness Engine."""
    print("Initializing UNIFIED Consciousness Engine components...")
    
    # Create the holographic substrate with non-uniform curvature
    manifold = HolographicManifold(
        dimensions=2,  # 2D spatial dimensions
        time_dimensions=1,  # 1 time dimension
        curvature_function=curvature_function
    )
    
    # Initialize the manifold grid
    manifold.initialize_grid(resolution=32)
    
    # Create energy and data fields
    shape = (32, 32)  # Field dimensions
    energy_field = EnergyField(shape=shape)
    data_field = DataField(shape=shape)
    
    # Create a field simulator to integrate the components
    simulator = FieldSimulator(
        manifold=manifold,
        energy_field=energy_field,
        data_fields=[data_field],
        dt=0.1  # Time step size
    )
    
    # Create some field visualizations
    print("Creating field visualizations...")
    field_viz = FieldVisualizer()
    combined_fig = field_viz.visualize_combined_fields(
        data_field=data_field,
        energy_field=energy_field,
        manifold=manifold,
        title="Initial Field State"
    )
    combined_fig.savefig("initial_fields.png", dpi=300, bbox_inches='tight')
    
    # Create and inject trytes into the field
    print("Creating trit patterns...")
    
    # Create a positive tryte (pattern of positive trits)
    positive_tryte = Tryte.from_pattern(
        pattern=[1, 1, 1],
        energy=1.0,
        geometry='triangle',
        coherence=0.9
    )
    
    # Create a mixed tryte (positive, neutral, negative)
    mixed_tryte = Tryte.from_pattern(
        pattern=[1, 0, -1],
        energy=1.0,
        geometry='triangle',
        coherence=0.7
    )
    
    # Inject the trytes into different locations in the field
    print("Injecting patterns into the field...")
    
    # Use simpler pattern creation to avoid shape mismatch
    positive_pattern = create_pulse_pattern(shape=shape, center=(8, 8), value=1, energy=1.0, width=3.0)
    mixed_pattern = create_pulse_pattern(shape=shape, center=(24, 24), value=-1, energy=1.0, width=3.0)
    
    simulator.inject_data_pattern(positive_pattern, (8, 8), field_idx=0)
    simulator.inject_data_pattern(mixed_pattern, (24, 24), field_idx=0)
    
    # Inject some energy to power the computation
    simulator.inject_energy_pulse(location=(16, 16), amount=2.0, radius=5)
    
    # Run the simulation for a few steps
    print("Running simulation...")
    for i in range(20):
        simulator.step()
        if i % 5 == 0:
            print(f"Simulation step {i}, time: {simulator.time:.2f}")
    
    # Visualize the final state
    print("Creating final state visualizations...")
    final_fig = field_viz.visualize_combined_fields(
        data_field=data_field,
        energy_field=energy_field,
        manifold=manifold,
        title="Field State After Simulation"
    )
    final_fig.savefig("final_fields.png", dpi=300, bbox_inches='tight')
    
    # Visualize attractors in the data field
    attractor_fig = field_viz.visualize_attractor_dynamics(
        data_field=data_field,
        threshold=0.3
    )
    attractor_fig.savefig("attractors.png", dpi=300, bbox_inches='tight')
    
    print("Basic field simulation complete.")
    
    # Return the simulator for further analysis if needed
    return simulator


def run_mtu_network_simulation():
    """Run a simulation with a network of Minimal Thinking Units."""
    print("Initializing MTU network...")
    
    # Define the network shape
    shape = (20, 20)  # 20x20 grid
    
    # Create the MTU network
    network = MTUNetwork(
        shape=shape,
        mtu_density=0.2,  # 20% of positions will have MTUs
        connections_per_mtu=4  # Each MTU connects to ~4 others
    )
    
    # Create some trytes to inject as input
    print("Creating input patterns...")
    
    # Create a balanced tryte
    balanced_tryte = Tryte.from_pattern(
        pattern=[1, -1, 1, -1],
        energy=1.0,
        geometry='triangle',
        coherence=0.8
    )
    
    # Create a primarily positive tryte
    positive_tryte = Tryte.from_pattern(
        pattern=[1, 1, 0, 1],
        energy=1.0,
        geometry='triangle',
        coherence=0.9
    )
    
    # Create a primarily negative tryte
    negative_tryte = Tryte.from_pattern(
        pattern=[-1, -1, 0, -1],
        energy=1.0,
        geometry='triangle',
        coherence=0.9
    )
    
    # Visualize the initial network state
    print("Creating network visualizations...")
    mtu_viz = MTUVisualizer()
    initial_network_fig = mtu_viz.visualize_mtu_network(
        network=network,
        title="Initial MTU Network"
    )
    initial_network_fig.savefig("initial_network.png", dpi=300, bbox_inches='tight')
    
    # Inject inputs at different locations
    print("Injecting inputs into the network...")
    
    # Create simpler patterns for network injection to avoid shape mismatches
    network.data_field.inject_pattern(
        create_pulse_pattern(shape=shape, center=(5, 5), value=1, energy=1.0, width=2.0),
        (5, 5)
    )
    network.data_field.inject_pattern(
        create_pulse_pattern(shape=shape, center=(15, 5), value=1, energy=1.0, width=2.0),
        (15, 5)
    )
    network.data_field.inject_pattern(
        create_pulse_pattern(shape=shape, center=(10, 15), value=-1, energy=1.0, width=2.0),
        (10, 15)
    )
    
    # Get MTU at positions close to injection points
    mtu_positions = network.get_mtu_positions()
    for idx, pos in mtu_positions.items():
        if np.linalg.norm(np.array(pos) - np.array((5, 5))) < 3:
            network._mtus[idx].receive_input(balanced_tryte)
        elif np.linalg.norm(np.array(pos) - np.array((15, 5))) < 3:
            network._mtus[idx].receive_input(positive_tryte)
        elif np.linalg.norm(np.array(pos) - np.array((10, 15))) < 3:
            network._mtus[idx].receive_input(negative_tryte)
    
    # Run the network for some steps
    print("Running network simulation...")
    outputs = network.run(steps=30)
    
    # Print output summary
    print(f"Network produced outputs from {len(outputs)} MTUs:")
    for mtu_idx, trytes in outputs.items():
        print(f"  MTU {mtu_idx} at {network.get_mtu_positions()[mtu_idx]}: {len(trytes)} outputs")
    
    # Visualize the final network state
    final_network_fig = mtu_viz.visualize_mtu_network(
        network=network,
        title="Final MTU Network State"
    )
    final_network_fig.savefig("final_network.png", dpi=300, bbox_inches='tight')
    
    # Visualize network activity
    activity_fig = mtu_viz.visualize_network_activity(
        network=network,
        time_steps=10,
        title="MTU Network Activity"
    )
    activity_fig.savefig("network_activity.png", dpi=300, bbox_inches='tight')
    
    print("MTU network simulation complete.")
    
    # Return the network for further analysis if needed
    return network


def create_network_animation_demo(steps=100):
    """Create an animated demonstration of the MTU network."""
    print("Creating network animation...")
    
    # Define the network shape
    shape = (20, 20)  # 20x20 grid
    
    # Create the MTU network
    network = MTUNetwork(
        shape=shape,
        mtu_density=0.2,
        connections_per_mtu=4
    )
    
    # Define a function to inject inputs at specific frames
    def input_function(frame):
        if frame == 0:
            # Inject a balanced tryte at the start
            tryte = Tryte.from_pattern(
                pattern=[1, -1, 1, -1],
                energy=1.0,
                geometry='triangle',
                coherence=0.8
            )
            return {"tryte": tryte, "position": (5, 5), "strength": 1.0}
        elif frame == 25:
            # Inject a positive tryte
            tryte = Tryte.from_pattern(
                pattern=[1, 1, 0, 1],
                energy=1.0,
                geometry='triangle',
                coherence=0.9
            )
            return {"tryte": tryte, "position": (15, 5), "strength": 1.0}
        elif frame == 50:
            # Inject a negative tryte
            tryte = Tryte.from_pattern(
                pattern=[-1, -1, 0, -1],
                energy=1.0,
                geometry='triangle',
                coherence=0.9
            )
            return {"tryte": tryte, "position": (10, 15), "strength": 1.0}
        elif frame == 75:
            # Inject a mixed tryte
            tryte = Tryte.from_pattern(
                pattern=[1, 0, -1, 1, -1],
                energy=1.5,  # Stronger input
                geometry='triangle',
                coherence=0.7
            )
            return {"tryte": tryte, "position": (10, 10), "strength": 1.5}
        return None
    
    # Create the interactive animation
    anim = create_interactive_animation(
        simulator_or_network=network,
        input_function=input_function,
        steps=steps,
        interval=100,  # milliseconds between frames
        figsize=(12, 8)
    )
    
    # Save the animation if possible
    try:
        anim.save("network_animation.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
        print("Animation saved as 'network_animation.mp4'")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("You can display the animation in a Jupyter notebook with:")
        print("from IPython.display import HTML")
        print("HTML(anim.to_jshtml())")
    
    return anim


if __name__ == "__main__":
    print("UNIFIED Consciousness Engine - Demonstration")
    print("===========================================\n")
    
    # Run basic field simulation
    simulator = run_basic_simulation()
    print("\n")
    
    # Run MTU network simulation
    network = run_mtu_network_simulation()
    print("\n")
    
    # Create animation demo
    animation = create_network_animation_demo(steps=100)
    
    print("\nAll simulations complete. Visualizations saved as PNG files.")
    print("To view the animation, run this script in a Jupyter notebook.")
    
    # Keep the plots open if run interactively
    plt.show()
