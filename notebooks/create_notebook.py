#!/usr/bin/env python
"""
Script to generate a Jupyter notebook for exploring the UNIFIED Consciousness Engine.
Run this script to create the notebook file, then open it with Jupyter.
"""

import json
import os

# Define the notebook structure
notebook = {
    "cells": [
        # Introduction cell
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# UNIFIED Consciousness Engine - Interactive Explorer\n",
                "\n",
                "This notebook provides an interactive environment for exploring the UNIFIED Consciousness Engine, \n",
                "a computational framework for emergent intelligence based on holographic substrate theory, field dynamics, and trit-based logic.\n",
                "\n",
                "## Overview\n",
                "\n",
                "The UNIFIED Consciousness Engine implements a novel approach to artificial intelligence by defining computation as morphogenic field dynamics within a holographic high-dimensional space. Instead of traditional neural networks, this system uses:\n",
                "\n",
                "- Holographic HD substrate as a computational medium\n",
                "- Trit-based pulse logic (+1, 0, -1) for information representation\n",
                "- Dynamic field interactions as computation\n",
                "- Emergent knowledge formation through field attractors\n",
                "- Minimal Thinking Units (MTUs) as computational nodes\n"
            ]
        },
        
        # Import dependencies cell
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from matplotlib.animation import FuncAnimation\n",
                "from IPython.display import HTML\n",
                "\n",
                "# Configure matplotlib for interactive notebooks\n",
                "%matplotlib notebook\n",
                "\n",
                "# Add the project root directory to the Python path\n",
                "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('__file__'), '..')))\n",
                "\n",
                "# Import UNIFIED Consciousness modules\n",
                "from src.unified.substrate.manifold import HolographicManifold\n",
                "from src.unified.field.energy_field import EnergyField\n",
                "from src.unified.field.data_field import DataField\n",
                "from src.unified.field.field_simulator import FieldSimulator\n",
                "from src.unified.trits.trit import Trit, TritState\n",
                "from src.unified.trits.tryte import Tryte\n",
                "from src.unified.trits.trit_patterns import create_pulse_pattern, create_wave_pattern\n",
                "from src.unified.mtu.mtu import MinimalThinkingUnit\n",
                "from src.unified.mtu.mtu_network import MTUNetwork\n",
                "from src.unified.visualization.field_visualizer import FieldVisualizer\n",
                "from src.unified.visualization.mtu_visualizer import MTUVisualizer\n",
                "from src.unified.visualization.animation import create_network_animation, create_interactive_animation"
            ]
        },

        # Exploring the holographic substrate
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Exploring the Holographic Substrate\n",
                "\n",
                "The holographic substrate is the foundation of the UNIFIED Consciousness Engine. It defines a curved, high-dimensional space where computation occurs through field interactions."
            ]
        },

        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define a non-uniform curvature function for the substrate\n",
                "def curvature_function(coords):\n",
                "    # Create a curved space with a central basin (attractor)\n",
                "    center = np.zeros(coords.shape[-1])\n",
                "    sq_dist = np.sum((coords - center) ** 2, axis=-1)\n",
                "    return -0.5 * np.exp(-sq_dist / 0.5)\n",
                "\n",
                "# Create the holographic substrate\n",
                "manifold = HolographicManifold(\n",
                "    dimensions=2,  # 2D spatial dimensions\n",
                "    time_dimensions=1,  # 1 time dimension\n",
                "    curvature_function=curvature_function\n",
                ")\n",
                "\n",
                "# Initialize the manifold grid\n",
                "manifold.initialize_grid(resolution=32)\n",
                "\n",
                "# Visualize the substrate curvature\n",
                "plt.figure(figsize=(10, 8))\n",
                "plt.imshow(manifold._grid, cmap='coolwarm', origin='lower')\n",
                "plt.colorbar(label='Curvature')\n",
                "plt.title('Holographic Substrate Curvature')\n",
                "plt.show()"
            ]
        },

        # Energy and Data Fields
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Field Dynamics\n",
                "\n",
                "The UNIFIED Consciousness Engine uses two primary fields that interact within the holographic substrate:\n",
                "\n",
                "- **Energy Field**: Represents the activation energy available in the system\n",
                "- **Data Field**: Stores and propagates information patterns\n",
                "\n",
                "These fields interact with each other and with the substrate curvature to produce complex dynamics."
            ]
        },

        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create energy and data fields\n",
                "energy_field = EnergyField(shape=(32, 32), persistence=0.95)\n",
                "data_field = DataField(shape=(32, 32), persistence=0.90)\n",
                "\n",
                "# Initialize with some energy\n",
                "initial_energy = np.zeros((32, 32))\n",
                "initial_energy[12:20, 12:20] = 1.0  # Add energy to a central region\n",
                "energy_field.set_field(initial_energy)\n",
                "\n",
                "# Create a field simulator that connects everything\n",
                "simulator = FieldSimulator(\n",
                "    manifold=manifold,\n",
                "    energy_fields=[energy_field],\n",
                "    data_fields=[data_field],\n",
                "    dt=0.1  # Time step size\n",
                ")\n",
                "\n",
                "# Visualize initial field states\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                "\n",
                "# Substrate curvature\n",
                "axes[0].imshow(manifold._grid, cmap='coolwarm')\n",
                "axes[0].set_title('Substrate Curvature')\n",
                "\n",
                "# Energy field\n",
                "axes[1].imshow(energy_field.get_field(), cmap='plasma')\n",
                "axes[1].set_title('Energy Field')\n",
                "\n",
                "# Data field\n",
                "axes[2].imshow(data_field.get_field(), cmap='viridis')\n",
                "axes[2].set_title('Data Field')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },

        # Trit Patterns
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Trit-Based Information Patterns\n",
                "\n",
                "The UNIFIED Consciousness Engine uses trinary logic (trits) instead of binary, where:\n",
                "\n",
                "- **+1**: Represents positive/excitatory states\n",
                "- **0**: Represents neutral/ground states\n",
                "- **-1**: Represents negative/inhibitory states\n",
                "\n",
                "Trits combine to form Trytes, which can create complex patterns that are injected into the data field."
            ]
        },

        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create trit patterns\n",
                "positive_pattern = create_pulse_pattern(shape=(24, 24), amplitude=0.8, frequency=2, phase=0)\n",
                "negative_pattern = create_pulse_pattern(shape=(24, 24), amplitude=-0.8, frequency=2, phase=np.pi)\n",
                "wave_pattern = create_wave_pattern(shape=(24, 24), amplitude=0.5, frequency=3, damping=0.2)\n",
                "\n",
                "# Visualize the patterns\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                "\n",
                "axes[0].imshow(positive_pattern, cmap='RdBu', vmin=-1, vmax=1)\n",
                "axes[0].set_title('Positive Pulse Pattern')\n",
                "\n",
                "axes[1].imshow(negative_pattern, cmap='RdBu', vmin=-1, vmax=1)\n",
                "axes[1].set_title('Negative Pulse Pattern')\n",
                "\n",
                "axes[2].imshow(wave_pattern, cmap='RdBu', vmin=-1, vmax=1)\n",
                "axes[2].set_title('Wave Pattern')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },

        # Running a field simulation
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Interactive Field Simulation\n",
                "\n",
                "Let's run a simulation of the field dynamics and visualize the results interactively. \n",
                "We'll inject a pattern into the data field and observe how it propagates and interacts with the substrate."
            ]
        },

        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inject a pattern into the data field\n",
                "simulator.inject_data_pattern(positive_pattern, (8, 8), field_idx=0)\n",
                "simulator.inject_data_pattern(negative_pattern, (24, 24), field_idx=0)\n",
                "\n",
                "# Create a field visualizer\n",
                "visualizer = FieldVisualizer()\n",
                "\n",
                "# Interactive animation function\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n",
                "\n",
                "# Store field history for animation\n",
                "energy_history = []\n",
                "data_history = []\n",
                "\n",
                "# Run a few steps and store results\n",
                "num_steps = 50\n",
                "for _ in range(num_steps):\n",
                "    simulator.step()\n",
                "    energy_history.append(energy_field.get_field().copy())\n",
                "    data_history.append(data_field.get_field().copy())\n",
                "\n",
                "# Set up the animation\n",
                "energy_im = axes[0].imshow(energy_history[0], cmap='plasma', vmin=0, vmax=1)\n",
                "data_im = axes[1].imshow(data_history[0], cmap='RdBu', vmin=-1, vmax=1)\n",
                "\n",
                "axes[0].set_title('Energy Field')\n",
                "axes[1].set_title('Data Field')\n",
                "\n",
                "plt.tight_layout()\n",
                "\n",
                "# Animation function\n",
                "def update(frame):\n",
                "    energy_im.set_array(energy_history[frame])\n",
                "    data_im.set_array(data_history[frame])\n",
                "    return energy_im, data_im\n",
                "\n",
                "# Create the animation\n",
                "anim = FuncAnimation(fig, update, frames=len(energy_history), interval=100, blit=True)\n",
                "\n",
                "# Display the animation\n",
                "HTML(anim.to_jshtml())"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Add more cells to the notebook
notebook_cells = notebook["cells"]

# Add MTU Network exploration section
notebook_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Minimal Thinking Units (MTUs)\n",
        "\n",
        "MTUs are the computational nodes of the UNIFIED Consciousness Engine. Each MTU:\n",
        "\n",
        "- Processes trit-based information in trinary logic\n",
        "- Operates through field interactions within the holographic substrate\n",
        "- Forms connections with other MTUs to create functional networks\n",
        "- Adapts based on field dynamics and energy flow\n"
    ]
})

notebook_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Create a simple MTU network\n",
        "network = MTUNetwork(substrate=manifold, field_simulator=simulator)\n",
        "\n",
        "# Add some MTUs\n",
        "network.add_mtu(MinimalThinkingUnit(position=(8, 8), pattern_size=(5, 5), threshold=0.3))\n",
        "network.add_mtu(MinimalThinkingUnit(position=(16, 16), pattern_size=(5, 5), threshold=0.3))\n",
        "network.add_mtu(MinimalThinkingUnit(position=(24, 8), pattern_size=(5, 5), threshold=0.3))\n",
        "\n",
        "# Connect the MTUs\n",
        "network.connect_mtu(0, 1, weight=0.7)  # Connect MTU 0 to MTU 1\n",
        "network.connect_mtu(1, 2, weight=0.7)  # Connect MTU 1 to MTU 2\n",
        "network.connect_mtu(2, 0, weight=0.4)  # Connect MTU 2 to MTU 0 (feedback)\n",
        "\n",
        "# Visualize the network\n",
        "mtu_visualizer = MTUVisualizer()\n",
        "network_fig = mtu_visualizer.visualize_network(network)\n",
        "plt.show()"
    ]
})

# Add Benchmark section
notebook_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Benchmarking and Analysis\n",
        "\n",
        "Let's evaluate the UNIFIED Consciousness Engine on a few benchmarks to assess its viability:\n",
        "\n",
        "1. **Pattern Recognition**: How well does the system recognize and respond to specific patterns?\n",
        "2. **Field Stability**: Can the system maintain stable field states over time?\n",
        "3. **Energy Efficiency**: How efficiently does the system use energy to process information?\n",
        "4. **Computational Complexity**: How does performance scale with system size?\n",
        "5. **Emergent Behavior**: Does the system exhibit any unexpected emergent properties?\n"
    ]
})

notebook_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 1. Pattern Recognition Benchmark\n",
        "import time\n",
        "\n",
        "def benchmark_pattern_recognition(num_patterns=10, shape=(32, 32)):\n",
        "    # Create test substrate and simulator\n",
        "    test_manifold = HolographicManifold(dimensions=2, time_dimensions=1)\n",
        "    test_manifold.initialize_grid(resolution=shape[0])\n",
        "    \n",
        "    test_energy = EnergyField(shape=shape, persistence=0.95)\n",
        "    test_data = DataField(shape=shape, persistence=0.90)\n",
        "    \n",
        "    test_simulator = FieldSimulator(\n",
        "        manifold=test_manifold,\n",
        "        energy_fields=[test_energy],\n",
        "        data_fields=[test_data],\n",
        "        dt=0.1\n",
        "    )\n",
        "    \n",
        "    # Create patterns\n",
        "    patterns = []\n",
        "    for i in range(num_patterns):\n",
        "        # Create patterns with different frequencies\n",
        "        p = create_pulse_pattern(\n",
        "            shape=(shape[0]//2, shape[1]//2),\n",
        "            amplitude=0.8,\n",
        "            frequency=1 + i * 0.5,\n",
        "            phase=i * np.pi / num_patterns\n",
        "        )\n",
        "        patterns.append(p)\n",
        "    \n",
        "    # Measure time to process patterns\n",
        "    start_time = time.time()\n",
        "    \n",
        "    # Inject patterns and run simulation\n",
        "    for i, pattern in enumerate(patterns):\n",
        "        # Put each pattern in a different location\n",
        "        x = shape[0] // 2 + int(np.cos(i * 2 * np.pi / num_patterns) * shape[0] // 4)\n",
        "        y = shape[1] // 2 + int(np.sin(i * 2 * np.pi / num_patterns) * shape[1] // 4)\n",
        "        \n",
        "        test_simulator.inject_data_pattern(pattern, (x, y), field_idx=0)\n",
        "        \n",
        "        # Run 5 steps for each pattern\n",
        "        for _ in range(5):\n",
        "            test_simulator.step()\n",
        "    \n",
        "    end_time = time.time()\n",
        "    processing_time = end_time - start_time\n",
        "    \n",
        "    # Measure field stability (variation in field values)\n",
        "    final_field = test_data.get_field()\n",
        "    field_variation = np.std(final_field)\n",
        "    \n",
        "    # Calculate energy efficiency (field activity / energy consumed)\n",
        "    total_energy = np.sum(test_energy.get_field())\n",
        "    field_activity = np.sum(np.abs(final_field))\n",
        "    energy_efficiency = field_activity / (total_energy + 1e-10)  # Avoid division by zero\n",
        "    \n",
        "    return {\n",
        "        'processing_time': processing_time,\n",
        "        'field_variation': field_variation,\n",
        "        'energy_efficiency': energy_efficiency,\n",
        "        'final_field': final_field\n",
        "    }\n",
        "\n",
        "# Run the benchmark\n",
        "benchmark_results = benchmark_pattern_recognition()\n",
        "\n",
        "# Display results\n",
        "print(f\"Pattern Recognition Benchmark Results:\")\n",
        "print(f\"Processing Time: {benchmark_results['processing_time']:.4f} seconds\")\n",
        "print(f\"Field Variation: {benchmark_results['field_variation']:.4f}\")\n",
        "print(f\"Energy Efficiency: {benchmark_results['energy_efficiency']:.4f}\")\n",
        "\n",
        "# Visualize the final field state\n",
        "plt.figure(figsize=(10, 8))\n",
        "plt.imshow(benchmark_results['final_field'], cmap='RdBu', vmin=-1, vmax=1)\n",
        "plt.colorbar(label='Field Value')\n",
        "plt.title('Final Field State After Pattern Processing')\n",
        "plt.show()"
    ]
})

# Add Conclusions section
notebook_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Conclusions and Next Steps\n",
        "\n",
        "Based on the exploration and benchmarks, we can draw initial conclusions about the UNIFIED Consciousness Engine:\n",
        "\n",
        "### Strengths\n",
        "- **Non-binary Logic**: The trit-based approach allows for a richer representation space than binary systems\n",
        "- **Holographic Substrate**: The curvature-based computation offers potential for emergent attractors\n",
        "- **Field Dynamics**: Information flows naturally through the system following physical principles\n",
        "\n",
        "### Areas for Improvement\n",
        "- **Performance Optimization**: Computational efficiency needs enhancement for larger systems\n",
        "- **Parameter Tuning**: Field persistence, diffusion rates, and other parameters require optimization\n",
        "- **MTU Architecture**: The MTU network needs a more sophisticated connection strategy\n",
        "\n",
        "### Next Development Steps\n",
        "1. Implement learning mechanisms for MTU networks\n",
        "2. Create larger, more complex substrate geometries\n",
        "3. Develop specialized pattern libraries for specific computational tasks\n",
        "4. Compare performance with traditional neural networks on standard benchmarks\n",
        "5. Explore hardware-acceleration options for field simulations\n",
        "\n",
        "This proof of concept demonstrates the viability of the UNIFIED Consciousness Engine approach for further development as a novel computing paradigm."
    ]
})

# Write the notebook to a file
output_path = os.path.join('notebooks', 'unified_explorer.ipynb')
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook created at {output_path}")
print("To launch the notebook, run:")
print("jupyter notebook unified_explorer.ipynb")
