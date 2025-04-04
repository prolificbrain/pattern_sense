"""Animation utilities for the UNIFIED Consciousness Engine."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

from ..field.field_simulator import FieldSimulator
from ..mtu.mtu_network import MTUNetwork


def create_field_animation(simulator: FieldSimulator, steps: int = 50, 
                          interval: int = 100, figsize: Tuple[int, int] = (10, 8)):
    """Create an animation of the field simulator running over time.
    
    Args:
        simulator: The field simulator to animate
        steps: Number of simulation steps to animate
        interval: Time between frames in milliseconds
        figsize: Figure size
        
    Returns:
        Matplotlib animation object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Initialize plot with current state
    data_im = axes[0].imshow(simulator.data_fields[0].field, cmap='viridis', origin='lower',
                          vmin=-1, vmax=1, animated=True)
    axes[0].set_title("Data Field")
    plt.colorbar(data_im, ax=axes[0], label="Data Value")
    
    energy_im = axes[1].imshow(simulator.energy_field.field, cmap='inferno', origin='lower',
                            animated=True)
    axes[1].set_title("Energy Field")
    plt.colorbar(energy_im, ax=axes[1], label="Energy")
    
    # For attractors
    attractor_plots = []
    
    # Status text for time step
    time_text = fig.text(0.02, 0.02, f"Time: {simulator.time:.2f}", fontsize=10)
    
    fig.tight_layout()
    
    def init():
        """Initialize animation frame."""
        return [data_im, energy_im, time_text] + attractor_plots
    
    def update(frame):
        """Update animation frame."""
        # Run one simulation step
        simulator.step()
        
        # Update data field plot
        data_im.set_array(simulator.data_fields[0].field)
        
        # Update energy field plot
        energy_im.set_array(simulator.energy_field.field)
        
        # Update time text
        time_text.set_text(f"Time: {simulator.time:.2f}")
        
        # Find and plot attractors if any
        for plot in attractor_plots:
            plot.remove()
        attractor_plots.clear()
        
        attractors = simulator.find_attractors()
        for attractor in attractors:
            centroid = attractor['centroid']
            # Add marker for attractor
            plot = axes[0].plot(centroid[1], centroid[0], 'ro', markersize=5)[0]
            attractor_plots.append(plot)
        
        return [data_im, energy_im, time_text] + attractor_plots
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=steps, init_func=init,
                         interval=interval, blit=True)
    
    return anim


def create_network_animation(network: MTUNetwork, steps: int = 50,
                           interval: int = 100, figsize: Tuple[int, int] = (10, 8)):
    """Create an animation of the MTU network running over time.
    
    Args:
        network: The MTU network to animate
        steps: Number of simulation steps to animate
        interval: Time between frames in milliseconds
        figsize: Figure size
        
    Returns:
        Matplotlib animation object
    """
    if len(network.shape) > 2:
        raise ValueError("Network animation currently only supports 1D and 2D networks")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Initialize plot with current state
    data_im = axes[0].imshow(network.data_field.field, cmap='viridis', origin='lower',
                          animated=True)
    axes[0].set_title("Data Field")
    plt.colorbar(data_im, ax=axes[0], label="Data Value")
    
    # Get network dimensions and MTU positions
    mtu_positions = network.get_mtu_positions()
    connections = network.get_connections()
    
    # For 2D networks, create scatter plot of MTUs
    if len(network.shape) == 2:
        # Extract MTU x, y coordinates
        x = [pos[1] for pos in mtu_positions.values()]
        y = [pos[0] for pos in mtu_positions.values()]
        
        # Initial scatter with all MTUs inactive
        mtu_scatter = axes[1].scatter(x, y, c='blue', s=50, animated=True)
        
        # Draw connections
        for idx, connected_indices in connections.items():
            pos1 = mtu_positions[idx]
            x1, y1 = pos1[1], pos1[0]
            for connected_idx in connected_indices:
                pos2 = mtu_positions[connected_idx]
                x2, y2 = pos2[1], pos2[0]
                axes[1].plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
        
        axes[1].set_title("MTU Network")
        axes[1].set_xlim(-1, network.shape[1])
        axes[1].set_ylim(-1, network.shape[0])
    
    # For 1D networks, create bar plot of MTU states
    else:  # 1D network
        # Extract MTU positions
        positions = [pos[0] for pos in mtu_positions.values()]
        # Initial bars with zero height
        mtu_bars = axes[1].bar(positions, np.zeros_like(positions), animated=True)
        
        axes[1].set_title("MTU Network Activity")
        axes[1].set_xlim(-1, network.shape[0])
        axes[1].set_ylim(0, 1)  # Will be adjusted during animation
    
    # Status text for time step
    step_text = fig.text(0.02, 0.02, "Step: 0", fontsize=10)
    
    fig.tight_layout()
    
    # Keep track of active MTUs and their outputs
    active_mtus = set()
    mtu_activity = {idx: 0 for idx in mtu_positions.keys()}
    
    def init():
        """Initialize animation frame."""
        if len(network.shape) == 2:
            return [data_im, mtu_scatter, step_text]
        else:  # 1D
            return [data_im] + list(mtu_bars) + [step_text]
    
    def update(frame):
        """Update animation frame."""
        # Run one network step
        outputs = network.step()
        
        # Update data field plot
        data_im.set_array(network.data_field.field)
        
        # Update activity counters
        for mtu_idx, _ in outputs:
            active_mtus.add(mtu_idx)
            mtu_activity[mtu_idx] += 1
        
        # Update MTU visualization
        if len(network.shape) == 2:
            # Create colors and sizes based on activity
            colors = ['red' if idx in active_mtus else 'blue' for idx in mtu_positions.keys()]
            sizes = [50 + 10 * mtu_activity.get(idx, 0) for idx in mtu_positions.keys()]
            
            # Update scatter
            mtu_scatter.set_color(colors)
            mtu_scatter.set_sizes(sizes)
            
        else:  # 1D
            # Update bar heights based on activity
            heights = [mtu_activity.get(idx, 0) for idx in mtu_positions.keys()]
            
            # Normalize if needed
            max_height = max(heights) if heights else 1
            if max_height > 0:
                normalized_heights = [h / max_height for h in heights]
            else:
                normalized_heights = heights
                
            # Update bars
            for bar, h in zip(mtu_bars, normalized_heights):
                bar.set_height(h)
            
            # Adjust y limit
            axes[1].set_ylim(0, max(1, max_height))
        
        # Update step text
        step_text.set_text(f"Step: {frame}")
        
        if len(network.shape) == 2:
            return [data_im, mtu_scatter, step_text]
        else:  # 1D
            return [data_im] + list(mtu_bars) + [step_text]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=steps, init_func=init,
                         interval=interval, blit=True)
    
    return anim


def create_interactive_animation(simulator_or_network: Union[FieldSimulator, MTUNetwork],
                               input_function: Callable[[int], Optional[Dict[str, Any]]],
                               steps: int = 50, interval: int = 100,
                               figsize: Tuple[int, int] = (12, 8)):
    """Create an interactive animation where inputs can be injected based on a function.
    
    Args:
        simulator_or_network: The simulator or network to animate
        input_function: Function that takes a frame number and returns input parameters
                       (or None for no input on that frame)
        steps: Number of simulation steps to animate
        interval: Time between frames in milliseconds
        figsize: Figure size
        
    Returns:
        Matplotlib animation object
    """
    is_network = isinstance(simulator_or_network, MTUNetwork)
    
    if is_network:
        network = simulator_or_network
        shape = network.shape
        data_field = network.data_field
    else:
        simulator = simulator_or_network
        shape = simulator.data_fields[0].shape
        data_field = simulator.data_fields[0]
    
    # Set up figure with three subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, width_ratios=[2, 1])
    
    # Data field plot
    ax_data = fig.add_subplot(gs[0, 0])
    data_im = ax_data.imshow(data_field.field, cmap='viridis', origin='lower',
                         vmin=-1, vmax=1, animated=True)
    plt.colorbar(data_im, ax=ax_data, label="Data Value")
    ax_data.set_title("Data Field")
    
    # Energy field or network plot
    ax_energy_network = fig.add_subplot(gs[1, 0])
    
    if is_network:
        # Extract MTU positions
        mtu_positions = network.get_mtu_positions()
        connections = network.get_connections()
        
        # For 2D networks, create scatter plot of MTUs
        if len(shape) == 2:
            # Extract MTU x, y coordinates
            x = [pos[1] for pos in mtu_positions.values()]
            y = [pos[0] for pos in mtu_positions.values()]
            
            # Initial scatter with all MTUs inactive
            mtu_scatter = ax_energy_network.scatter(x, y, c='blue', s=50, animated=True)
            
            # Draw connections
            for idx, connected_indices in connections.items():
                pos1 = mtu_positions[idx]
                x1, y1 = pos1[1], pos1[0]
                for connected_idx in connected_indices:
                    pos2 = mtu_positions[connected_idx]
                    x2, y2 = pos2[1], pos2[0]
                    ax_energy_network.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)
            
            energy_network_obj = mtu_scatter
            ax_energy_network.set_title("MTU Network")
            ax_energy_network.set_xlim(-1, shape[1])
            ax_energy_network.set_ylim(-1, shape[0])
        else:
            # For non-2D networks, show data field structure over time
            energy_network_obj = ax_energy_network.plot([], [])[0]
            ax_energy_network.set_title("Data Structure Over Time")
            ax_energy_network.set_xlabel("Time Step")
            ax_energy_network.set_ylabel("Structure Metric")
            
    else:  # Field simulator
        energy_im = ax_energy_network.imshow(simulator.energy_field.field, 
                                         cmap='inferno', origin='lower',
                                         animated=True)
        plt.colorbar(energy_im, ax=ax_energy_network, label="Energy")
        ax_energy_network.set_title("Energy Field")
        energy_network_obj = energy_im
    
    # Input visualization plot
    ax_input = fig.add_subplot(gs[:, 1])
    ax_input.set_title("Input Activity")
    ax_input.set_xlabel("Time Step")
    ax_input.set_ylabel("Input Strength")
    input_line = ax_input.plot([], [], 'r-')[0]
    input_scatter = ax_input.scatter([], [], c='r', s=50)
    
    # Status text
    info_text = fig.text(0.02, 0.02, "Step: 0", fontsize=10)
    
    fig.tight_layout()
    
    # Data for tracking inputs and structure over time
    time_steps = []
    input_strengths = []
    structure_metrics = []
    
    # Optional network tracking
    active_mtus = set()
    mtu_activity = {} if is_network else None
    
    def init():
        """Initialize animation frame."""
        if is_network and len(shape) == 2:
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
        elif is_network:
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
        else:  # Field simulator
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
    
    def update(frame):
        """Update animation frame."""
        # Get input from function for this frame
        input_params = input_function(frame)
        input_strength = 0
        
        # Apply input if provided
        if input_params is not None:
            if is_network:
                if 'tryte' in input_params and 'position' in input_params:
                    network.inject_input(input_params['tryte'], input_params['position'])
                    input_strength = input_params.get('strength', 1.0)
            else:  # Field simulator
                if 'pattern' in input_params and 'location' in input_params:
                    simulator.inject_data_pattern(
                        input_params['pattern'],
                        input_params['location']
                    )
                    input_strength = np.max(np.abs(input_params['pattern']))
                
                if 'energy_amount' in input_params and 'energy_location' in input_params:
                    simulator.inject_energy_pulse(
                        input_params['energy_location'],
                        input_params['energy_amount']
                    )
        
        # Run simulation step
        if is_network:
            outputs = network.step()
            
            # Update activity tracking
            for mtu_idx, _ in outputs:
                active_mtus.add(mtu_idx)
                mtu_activity[mtu_idx] = mtu_activity.get(mtu_idx, 0) + 1
        else:
            simulator.step()
        
        # Update data field plot
        data_im.set_array(data_field.field)
        
        # Calculate structure metric
        structure_metric = data_field.compute_structure_metric()
        
        # Update tracking data
        time_steps.append(frame)
        input_strengths.append(input_strength)
        structure_metrics.append(structure_metric)
        
        # Update input visualization
        input_line.set_data(time_steps, input_strengths)
        input_scatter.set_offsets(np.column_stack([time_steps, input_strengths]))
        ax_input.relim()
        ax_input.autoscale_view()
        
        # Update energy/network plot
        if is_network:
            if len(shape) == 2:
                # Update MTU colors and sizes based on activity
                if mtu_activity:
                    max_activity = max(mtu_activity.values()) if mtu_activity else 1
                    colors = ['red' if idx in active_mtus else 'blue' 
                             for idx in network.get_mtu_positions().keys()]
                    sizes = [50 + 50 * mtu_activity.get(idx, 0) / max_activity 
                            for idx in network.get_mtu_positions().keys()]
                    
                    # Update scatter
                    energy_network_obj.set_color(colors)
                    energy_network_obj.set_sizes(sizes)
            else:
                # Update structure metric plot
                energy_network_obj.set_data(time_steps, structure_metrics)
                ax_energy_network.relim()
                ax_energy_network.autoscale_view()
        else:  # Field simulator
            energy_network_obj.set_array(simulator.energy_field.field)
        
        # Update info text
        if is_network:
            active_count = len(active_mtus)
            total_count = len(network.get_mtu_positions())
            info_text.set_text(f"Step: {frame} | Active MTUs: {active_count}/{total_count} | Structure: {structure_metric:.3f}")
        else:
            info_text.set_text(f"Step: {frame} | Time: {simulator.time:.2f} | Structure: {structure_metric:.3f}")
        
        if is_network and len(shape) == 2:
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
        elif is_network:
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
        else:  # Field simulator
            return [data_im, energy_network_obj, input_line, input_scatter, info_text]
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=steps, init_func=init,
                         interval=interval, blit=True)
    
    return anim
