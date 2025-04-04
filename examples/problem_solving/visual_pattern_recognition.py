"""Visual Pattern Recognition using the UNIFIED Consciousness Engine.

This example demonstrates the application of the UNIFIED Consciousness Engine to 
recognize visual patterns from the MNIST dataset of handwritten digits.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

# Torch/torchvision for dataset loading
import torch
from torchvision import datasets, transforms

# Import from UNIFIED Consciousness Engine
from unified.substrate.manifold import HolographicManifold
from unified.field.energy_field import EnergyField
from unified.field.data_field import DataField
from unified.field.field_simulator import FieldSimulator
from unified.mtu.mtu import MinimalThinkingUnit
from unified.mtu.mtu_network import MTUNetwork
from unified.trits.trit import Trit, TritState
from unified.trits.tryte import Tryte
from unified.visualization.mtu_visualizer import MTUVisualizer


def load_mnist_data(batch_size=64, train_limit=None, test_limit=None, download=True):
    """Load MNIST dataset for training and testing.
    
    Args:
        batch_size: Number of samples per batch
        train_limit: Maximum number of training samples to use (None for all)
        test_limit: Maximum number of test samples to use (None for all)
        download: Whether to download the dataset if not already downloaded
        
    Returns:
        train_loader: DataLoader with training data
        test_loader: DataLoader with test data
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('data', train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=download, transform=transform)
    
    # Limit dataset size if requested
    if train_limit is not None and train_limit < len(train_dataset):
        train_indices = list(range(train_limit))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    if test_limit is not None and test_limit < len(test_dataset):
        test_indices = list(range(test_limit))
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader


def create_visual_recognition_network(shape=(28, 28), mtu_density=0.2, hidden_layers=3):
    """Create an MTU network optimized for visual pattern recognition.
    
    Args:
        shape: Shape of the input field (matching MNIST image dimensions)
        mtu_density: Density of MTUs in the field
        hidden_layers: Number of hidden layers in the network
        
    Returns:
        network: Configured MTU network for visual recognition
    """
    # Create holographic manifold as substrate
    manifold = HolographicManifold(
        dimensions=2,
        curvature_function=lambda x: np.exp(-4 * np.sum(x**2, axis=-1))
    )
    
    # Create energy and data fields
    energy_field = EnergyField(shape=shape)
    energy_field._field = np.ones(shape) * 0.5  # Initialize with uniform energy
    
    data_field = DataField(shape=shape)
    
    # Create field simulator
    simulator = FieldSimulator(
        manifold=manifold,
        energy_field=energy_field,
        data_fields=[data_field],
        dt=0.1
    )
    
    # Create network with layered architecture - with higher connection density
    network = MTUNetwork(
        shape=shape,
        field_simulator=simulator,
        mtu_density=mtu_density,         # Higher MTU density
        connection_radius=5.0,           # Larger connection radius
        connection_density=0.7,          # Higher connection density
        connections_per_mtu=12,          # More connections per MTU
        enable_hebbian_learning=True,
        enable_pattern_memory=True,
        enable_adaptive_learning=True,
        structured_layers=hidden_layers + 2  # Input layer + hidden layers + output layer
    )
    
    # Ensure we have at least 10 output MTUs for digit classification
    create_specialized_output_layer(network, num_classes=10)
    
    # Add specialized MTUs for feature detection
    add_specialized_feature_detectors(network)
    
    print(f"Created network with {len(network._mtus)} MTUs and {network.total_connections} connections")
    print(f"Network has {network.num_layers} layers")
    
    return network


def create_specialized_output_layer(network, num_classes=10):
    """Create a specialized output layer with sufficient MTUs for classification.
    
    Args:
        network: The MTU network to modify
        num_classes: Number of classes to classify (digits 0-9)
    """
    # Get the index of the output layer
    output_layer = network.num_layers - 1
    
    # Get existing MTUs in the output layer
    output_mtus = [i for i, _ in enumerate(network._mtus) 
                  if network.get_mtu_layer(i) == output_layer]
    
    # If we have fewer than needed, add more
    num_to_add = max(0, num_classes - len(output_mtus))
    
    if num_to_add > 0:
        print(f"Adding {num_to_add} MTUs to output layer for classification")
        
        # Determine y-coordinate for output layer
        y_positions = [network._coordinates[i][1] for i in output_mtus]
        output_y = max(y_positions) if y_positions else network.shape[0] - 1
        
        # Evenly space the new MTUs along the x-axis
        x_spacing = network.shape[1] / (num_classes + 1)
        
        # Add specialized MTUs for each class
        for i in range(num_to_add):
            class_idx = len(output_mtus) + i
            x_pos = (class_idx + 1) * x_spacing
            
            # Create a new MTU with pattern memory enabled
            mtu = MinimalThinkingUnit(
                position=(x_pos, output_y),
                pattern_memory_enabled=True,
                attractor_dynamics_enabled=True
            )
            
            # Add to network
            mtu_idx = network.add_mtu(mtu)
            
            # Connect it to MTUs in the previous layer
            prev_layer = output_layer - 1
            prev_layer_mtus = [i for i, _ in enumerate(network._mtus) 
                             if network.get_mtu_layer(i) == prev_layer]
            
            # Connect to a subset of MTUs in the previous layer
            num_connections = min(5, len(prev_layer_mtus))
            for j in np.random.choice(prev_layer_mtus, num_connections, replace=False):
                network.connect_mtu(j, mtu_idx, weight=0.5)


def add_specialized_feature_detectors(network):
    """Add specialized MTUs for edge detection and feature extraction.
    
    Args:
        network: The MTU network to modify
    """
    # First layer specializations (edge detectors)
    edge_detector_types = [
        "horizontal",   # Detect horizontal lines
        "vertical",     # Detect vertical lines
        "diagonal_up",  # Detect diagonal lines (bottom-left to top-right)
        "diagonal_down" # Detect diagonal lines (top-left to bottom-right)
    ]
    
    # Get MTUs in the first hidden layer
    layer_1 = 1  # First hidden layer (after input layer)
    layer_1_mtus = [i for i, _ in enumerate(network._mtus) 
                   if network.get_mtu_layer(i) == layer_1]
    
    # Create specialized edge detector MTUs
    if len(layer_1_mtus) >= len(edge_detector_types):
        # Assign specializations to some existing MTUs
        specialized_mtus = layer_1_mtus[:len(edge_detector_types)]
        
        for mtu_idx, edge_type in zip(specialized_mtus, edge_detector_types):
            mtu = network._mtus[mtu_idx]
            
            # Create edge detector pattern
            pattern = create_edge_detector_pattern(network.shape, edge_type)
            
            # Initialize MTU with this pattern
            if hasattr(mtu, '_pattern_memory') and mtu._pattern_memory:
                mtu._pattern_memory.learn_pattern(pattern)
                
                # Also set the state field directly
                state_field = np.zeros(network.shape)
                for trit in pattern.trits:
                    x, y = trit.orientation
                    if 0 <= x < network.shape[1] and 0 <= y < network.shape[0]:
                        state_field[int(y), int(x)] = trit.value
                mtu._state_field = state_field
                
                print(f"Created {edge_type} edge detector MTU at index {mtu_idx}")


def create_edge_detector_pattern(shape, edge_type):
    """Create a pattern for detecting edges of a specific type.
    
    Args:
        shape: Shape of the field
        edge_type: Type of edge to detect
        
    Returns:
        pattern: Tryte pattern for edge detection
    """
    # Create field for the pattern
    field = np.zeros(shape)
    
    # Middle of the field
    mid_x, mid_y = shape[1] // 2, shape[0] // 2
    
    # Size of the edge detector
    size = min(shape[0], shape[1]) // 3
    
    # Create the pattern based on edge type
    if edge_type == "horizontal":
        # Horizontal line
        field[mid_y, mid_x-size//2:mid_x+size//2] = 1
    elif edge_type == "vertical":
        # Vertical line
        field[mid_y-size//2:mid_y+size//2, mid_x] = 1
    elif edge_type == "diagonal_up":
        # Diagonal line (bottom-left to top-right)
        for i in range(-size//2, size//2):
            y, x = mid_y - i, mid_x + i
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                field[y, x] = 1
    elif edge_type == "diagonal_down":
        # Diagonal line (top-left to bottom-right)
        for i in range(-size//2, size//2):
            y, x = mid_y + i, mid_x + i
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                field[y, x] = 1
    
    # Convert field to trits
    trits = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            if field[y, x] > 0:
                trit = Trit(state=TritState.POSITIVE)
                trit.orientation = (float(x), float(y))
                trits.append(trit)
    
    # Create pattern
    return Tryte(trits)


def image_to_tryte_patterns(image_batch, labels):
    """Convert a batch of MNIST images to tryte patterns for the MTU network.
    
    Args:
        image_batch: Batch of MNIST images (B, 1, 28, 28)
        labels: Corresponding labels
        
    Returns:
        patterns: List of (pattern, label) tuples
    """
    patterns = []
    
    for i in range(len(image_batch)):
        # Get the image and label
        image = image_batch[i][0].numpy()  # Shape: (28, 28)
        label = labels[i].item()
        
        # Normalize to -1 to 1 range for trits
        image = (image - 0.5) * 2
        
        # Convert to trits
        trit_matrix = []
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Threshold the pixel value to -1, 0, or 1
                pixel_value = image[y, x]
                if pixel_value > 0.3:
                    trit_state = TritState.POSITIVE  # +1
                elif pixel_value < -0.3:
                    trit_state = TritState.NEGATIVE  # -1
                else:
                    trit_state = TritState.NEUTRAL   # 0
                    
                # Create a trit with the state
                trit = Trit(state=trit_state)
                # Store position as orientation for our 2D image
                trit.orientation = (float(x), float(y))
                trit_matrix.append(trit)
        
        # Create a composite tryte pattern from all trits
        tryte = Tryte(trit_matrix)
        
        patterns.append((tryte, label))
    
    return patterns


def apply_data_augmentation(image):
    """Apply data augmentation to MNIST images.
    
    This helps the network generalize better by learning invariant features.
    
    Args:
        image: Input image tensor
        
    Returns:
        Augmented image tensor
    """
    # Convert to numpy for easier manipulation
    img = image.numpy()
    
    # Random rotation (up to 15 degrees)
    angle = np.random.uniform(-15, 15)
    rad = np.deg2rad(angle)
    cos_val, sin_val = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])
    
    # Get center of image
    center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
    
    # Create rotated image
    rotated = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Translate to origin, rotate, then translate back
            x, y = i - center_x, j - center_y
            new_x, new_y = np.dot(rotation_matrix, [x, y])
            new_x, new_y = int(new_x + center_x), int(new_y + center_y)
            
            # Check if the new coordinates are within bounds
            if 0 <= new_x < img.shape[0] and 0 <= new_y < img.shape[1]:
                rotated[i, j] = img[new_x, new_y]
    
    # Random translation (up to 2 pixels)
    shift_x, shift_y = np.random.randint(-2, 3), np.random.randint(-2, 3)
    translated = np.zeros_like(rotated)
    for i in range(rotated.shape[0]):
        for j in range(rotated.shape[1]):
            orig_x, orig_y = i - shift_x, j - shift_y
            if 0 <= orig_x < rotated.shape[0] and 0 <= orig_y < rotated.shape[1]:
                translated[i, j] = rotated[orig_x, orig_y]
    
    # Add slight noise
    noise = np.random.normal(0, 0.05, translated.shape)
    noisy = translated + noise
    noisy = np.clip(noisy, -1, 1)
    
    return torch.tensor(noisy).unsqueeze(0)


def train_visual_network(network, train_loader, epochs=1, samples_per_epoch=100):
    """Train the MTU network on MNIST images.
    
    Args:
        network: MTU network to train
        train_loader: DataLoader with training data
        epochs: Number of training epochs
        samples_per_epoch: Number of samples to use per epoch
        
    Returns:
        training_stats: Dictionary with training statistics
    """
    training_stats = {
        'accuracy_per_epoch': [],
        'loss_per_epoch': [],
        'connection_strength_changes': []
    }
    
    visualizer = MTUVisualizer()
    
    print(f"Training network for {epochs} epochs...")
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        initial_strengths = {}
        
        # Store initial connection strengths
        for mtu_idx in range(len(network._mtus)):
            if mtu_idx in network._connections:
                initial_strengths[mtu_idx] = {}
                for target in network._connections[mtu_idx]:
                    if (mtu_idx, target) in network._connection_strengths:
                        initial_strengths[mtu_idx][target] = network._connection_strengths[(mtu_idx, target)]
        
        # Get training data for this epoch
        data_iter = iter(train_loader)
        
        for batch_idx in range(samples_per_epoch):
            try:
                # Get a batch
                data, target = next(data_iter)
                
                # Apply data augmentation
                data = torch.cat([apply_data_augmentation(img) for img in data])
                
                # Convert to tryte patterns
                patterns = image_to_tryte_patterns(data, target)
                
                # Train on each pattern in the batch
                for pattern_idx, (pattern, label) in enumerate(patterns):
                    # Clear the network activity
                    network.reset()
                    
                    # Inject the pattern into each input layer MTU
                    # Find input layer MTUs (layer 0)
                    input_layer_mtus = [idx for idx in range(len(network._mtus)) 
                                       if network.get_mtu_layer(idx) == 0]
                    
                    # Inject the pattern at the appropriate position
                    for mtu_idx in input_layer_mtus:
                        network.inject_input(pattern, position=network._coordinates[mtu_idx])  # Inject at the MTU's position
                    
                    # Run the network for a fixed number of steps
                    recognition_steps = 10
                    outputs_by_layer = {}
                    for step in range(recognition_steps):
                        step_outputs = network.step()
                        
                        # Group outputs by layer
                        for mtu_idx, output in step_outputs:
                            mtu_layer = network.get_mtu_layer(mtu_idx)
                            if mtu_layer not in outputs_by_layer:
                                outputs_by_layer[mtu_layer] = []
                            outputs_by_layer[mtu_layer].append((mtu_idx, output))
                    
                    # Check the output layer activations
                    output_layer = network.num_layers - 1
                    output_mtus = outputs_by_layer.get(output_layer, [])
                    
                    # Simple readout mechanism: each output MTU is assigned to a digit
                    if output_mtus:
                        # Get the most active MTU in the output layer
                        most_active_mtu = max(output_mtus, key=lambda x: x[1].energy)
                        predicted_digit = network.get_mtu_index_in_layer(most_active_mtu[0], output_layer)
                        
                        # Check if the prediction matches the label
                        if predicted_digit == label:
                            correct += 1
                    
                    total += 1
                    
                    # Compute a simple loss based on the output layer activation
                    target_activation = 0
                    for mtu_idx, output in output_mtus:
                        mtu_digit = network.get_mtu_index_in_layer(mtu_idx, output_layer)
                        if mtu_digit == label:
                            target_activation += output.energy
                    
                    loss = 1.0 - (target_activation / (len(output_mtus) + 1e-6))
                    epoch_loss += loss
                    
                    # Provide feedback for Hebbian learning
                    network.hebbian_update()
                    
                    # Visualize every 10th pattern
                    if pattern_idx % 10 == 0:
                        # Visualize the input pattern
                        plt.figure(figsize=(6, 6))
                        img = np.zeros(network.shape)
                        for trit in pattern.trits:
                            x, y = trit.orientation
                            if 0 <= x < network.shape[1] and 0 <= y < network.shape[0]:
                                img[int(y), int(x)] = trit.value
                        
                        plt.imshow(img, cmap='gray')
                        plt.title(f"Input Pattern (Digit {label})")
                        plt.colorbar()
                        plt.savefig(f"mnist_input_epoch{epoch}_batch{batch_idx}.png")
                        plt.close()
                        
                        # Visualize network activity
                        fig = network.visualize()
                        plt.savefig(f"network_activity_epoch{epoch}_batch{batch_idx}.png")
                        plt.close()
            
            except StopIteration:
                break
        
        # Calculate epoch statistics
        epoch_accuracy = correct / total if total > 0 else 0
        epoch_loss = epoch_loss / total if total > 0 else float('inf')
        
        training_stats['accuracy_per_epoch'].append(epoch_accuracy)
        training_stats['loss_per_epoch'].append(epoch_loss)
        
        # Calculate connection strength changes
        strength_changes = {}
        for mtu_idx in initial_strengths:
            if mtu_idx in network._connections:
                changes = []
                for target in initial_strengths[mtu_idx]:
                    if (mtu_idx, target) in network._connection_strengths:
                        initial = initial_strengths[mtu_idx][target]
                        final = network._connection_strengths[(mtu_idx, target)]
                        changes.append(final - initial)
                if changes:
                    strength_changes[mtu_idx] = np.mean(changes)
        
        avg_strength_change = np.mean(list(strength_changes.values())) if strength_changes else 0
        training_stats['connection_strength_changes'].append(avg_strength_change)
        
        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.4f}, "
              f"Avg Connection Change: {avg_strength_change:.6f}")
        
        # Visualize network after each epoch
        fig = network.visualize()
        plt.savefig(f"mnist_network_epoch{epoch+1}.png")
        plt.close()
    
    return training_stats


def test_visual_network(network, test_loader, num_samples=100):
    """Test the trained network on MNIST test data.
    
    Args:
        network: Trained MTU network
        test_loader: DataLoader with test data
        num_samples: Number of test samples to evaluate
        
    Returns:
        test_stats: Dictionary with test statistics
    """
    test_stats = {
        'correct': 0,
        'total': 0,
        'confusion_matrix': np.zeros((10, 10), dtype=int),
        'examples': []
    }
    
    # Get test samples
    total = 0
    correct = 0
    
    # Get test data iterator
    data_iter = iter(test_loader)
    
    try:
        while total < num_samples:
            # Get a batch
            data, target = next(data_iter)
            
            # Convert to tryte patterns
            patterns = image_to_tryte_patterns(data, target)
            
            # Test on each pattern in the batch
            for pattern_idx, (pattern, label) in enumerate(patterns):
                if total >= num_samples:
                    break
                
                # Clear the network activity
                network.reset()
                
                # Inject the pattern to input layer MTUs
                input_layer_mtus = [idx for idx in range(len(network._mtus)) 
                                  if network.get_mtu_layer(idx) == 0]
                for mtu_idx in input_layer_mtus:
                    network.inject_input(pattern, position=network._coordinates[mtu_idx])
                
                # Run the network for a fixed number of steps
                recognition_steps = 10
                outputs_by_layer = {}
                
                for step in range(recognition_steps):
                    step_outputs = network.step()
                    
                    # Group outputs by layer
                    for mtu_idx, output in step_outputs:
                        mtu_layer = network.get_mtu_layer(mtu_idx)
                        if mtu_layer not in outputs_by_layer:
                            outputs_by_layer[mtu_layer] = []
                        outputs_by_layer[mtu_layer].append((mtu_idx, output))
                
                # Check the output layer activations
                output_layer = network.num_layers - 1
                output_mtus = outputs_by_layer.get(output_layer, [])
                
                # Determine if prediction was correct
                predicted_digit = -1
                
                if output_mtus:
                    # Get the most active MTU in the output layer
                    most_active_mtu = max(output_mtus, key=lambda x: x[1].energy)
                    predicted_digit = network.get_mtu_index_in_layer(most_active_mtu[0], output_layer)
                    
                    # Check if the prediction matches the label
                    if predicted_digit == label:
                        correct += 1
                
                # Update confusion matrix
                if 0 <= predicted_digit < 10:
                    test_stats['confusion_matrix'][label, predicted_digit] += 1
                
                # Save example
                test_stats['examples'].append((pattern, label, predicted_digit))
                
                total += 1
                
                # Visualize some test examples
                if pattern_idx % 10 == 0:
                    # Visualize the input pattern and network response
                    plt.figure(figsize=(6, 6))
                    img = np.zeros(network.shape)
                    for trit in pattern.trits:
                        x, y = trit.orientation
                        if 0 <= int(x) < network.shape[1] and 0 <= int(y) < network.shape[0]:
                            img[int(y), int(x)] = trit.value
                    
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Test: True={label}, Pred={predicted_digit}, "  
                              f"{'Correct' if predicted_digit == label else 'Incorrect'}")
                    plt.colorbar()
                    plt.savefig(f"mnist_test_sample{total}.png")
                    plt.close()
        
    except StopIteration:
        pass
    
    # Calculate overall accuracy
    test_stats['accuracy'] = correct / total if total > 0 else 0
    test_stats['correct'] = correct
    test_stats['total'] = total
    
    return test_stats


def visualize_training_results(training_stats):
    """Visualize the training results.
    
    Args:
        training_stats: Dictionary with training statistics
    """
    # Create a figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot accuracy
    epochs = range(1, len(training_stats['accuracy_per_epoch']) + 1)
    ax1.plot(epochs, training_stats['accuracy_per_epoch'])
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(epochs, training_stats['loss_per_epoch'])
    ax2.set_title('Training Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    # Plot connection strength changes
    ax3.plot(epochs, training_stats['connection_strength_changes'])
    ax3.set_title('Connection Strength Changes')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Average Change')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_training_results.png')
    plt.close()


def visualize_test_results(network, test_stats):
    """Visualize the test results.
    
    Args:
        network: The trained MTU network
        test_stats: Dictionary with test statistics
    """
    # Create confusion matrix visualization
    if 'confusion_matrix' in test_stats:
        plt.figure(figsize=(10, 8))
        cm = test_stats['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels
        classes = range(10)  # 0-9 digits
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("mnist_confusion_matrix.png")
        plt.close()
    
    # Create network activity visualization for a few test samples
    sample_count = min(5, len(test_stats.get('examples', [])))
    if 'examples' in test_stats and sample_count > 0:
        for i in range(sample_count):
            pattern, label, predicted = test_stats['examples'][i]
            
            # Display the sample
            plt.figure(figsize=(6, 6))
            img = np.zeros(network.shape)
            for trit in pattern.trits:
                x, y = trit.orientation
                if 0 <= int(x) < network.shape[1] and 0 <= int(y) < network.shape[0]:
                    img[int(y), int(x)] = trit.value
            
            plt.imshow(img, cmap='gray')
            plt.title(f"Test Sample: True={label}, Predicted={predicted}")
            plt.colorbar()
            plt.savefig(f"mnist_test_sample{i}.png")
            plt.close()


def document_performance_limitations(training_stats, test_stats, network, training_time, testing_time):
    """Document the performance and limitations of the network.
    
    Args:
        training_stats: Dictionary with training statistics
        test_stats: Dictionary with test statistics
        network: The trained MTU network
        training_time: Time taken for training
        testing_time: Time taken for testing
    """
    with open('mnist_performance_report.md', 'w') as f:
        f.write("# UNIFIED Consciousness Engine - MNIST Performance Report\n\n")
        
        f.write("## Network Architecture\n\n")
        f.write(f"- Network Shape: {network.shape}\n")
        f.write(f"- Number of MTUs: {len(network._mtus)}\n")
        f.write(f"- Number of Connections: {network.total_connections}\n")
        f.write(f"- Number of Layers: {network.num_layers}\n\n")
        
        f.write("## Training Performance\n\n")
        f.write(f"- Training Time: {training_time:.2f} seconds\n")
        f.write(f"- Final Training Accuracy: {training_stats['accuracy_per_epoch'][-1]:.4f}\n")
        f.write(f"- Final Training Loss: {training_stats['loss_per_epoch'][-1]:.4f}\n\n")
        
        f.write("## Testing Performance\n\n")
        f.write(f"- Testing Time: {testing_time:.2f} seconds\n")
        f.write(f"- Test Accuracy: {test_stats['accuracy']:.4f}\n\n")
        
        f.write("## Learning Dynamics\n\n")
        f.write(f"- Average Connection Strength Change: {training_stats['connection_strength_changes'][-1]:.6f}\n\n")
        
        f.write("## Limitations\n\n")
        f.write("1. **Limited Capacity**: The network has fewer parameters than traditional deep learning models, which may limit its capacity to learn complex patterns.\n")
        f.write("2. **Slower Convergence**: The learning process is slower compared to gradient-based methods like backpropagation.\n")
        f.write("3. **Image Preprocessing**: Converting images to trinary representations leads to information loss.\n")
        f.write("4. **Scalability**: The current implementation may face scalability challenges with larger networks.\n")
        f.write("5. **Readout Mechanism**: The simple readout mechanism may not fully capture the distributed representations.\n\n")
        
        f.write("## Advantages\n\n")
        f.write("1. **Emergent Learning**: The network demonstrates emergent pattern recognition capabilities without explicit supervision.\n")
        f.write("2. **Biological Plausibility**: The learning mechanism is more biologically plausible than backpropagation.\n")
        f.write("3. **Local Processing**: Learning is based on local information, which is more neuromorphic.\n")
        f.write("4. **Adaptability**: The network can adapt to new patterns without catastrophic forgetting.\n")


def run_mnist_demonstration(epochs=3, train_samples=300, test_samples=100):
    """Run the MNIST demonstration with enhanced UNIFIED Consciousness Engine parameters.
    
    This demonstration showcases the visual pattern recognition capabilities of the
    UNIFIED Consciousness Engine using the MNIST dataset with improved parameters.
    
    Args:
        epochs: Number of training epochs
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
    """
    print("UNIFIED Consciousness Engine - MNIST Demonstration")
    print("=" * 50 + "\n")
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=32, train_limit=train_samples, test_limit=test_samples)
    print(f"Loaded {train_samples} training samples and {test_samples} test samples")
    
    # Create MTU network with enhanced parameters
    print("\nCreating visual recognition network...")
    network = create_visual_recognition_network(
        shape=(28, 28),       # MNIST image size
        mtu_density=0.25,     # Increased density (0.2 -> 0.25)
        hidden_layers=3       # More hidden layers (2 -> 3)
    )
    
    # Enable adaptive learning with enhanced parameters
    network.enable_adaptive_learning = True
    if hasattr(network, '_hebbian_learning'):
        # Increase learning rate (0.01 -> 0.05)
        network._hebbian_learning.learning_rate = 0.05
        # Reduce decay rate (0.01 -> 0.001)
        network._hebbian_learning.decay_rate = 0.001
    
    # Train the network
    print("\nTraining network...")
    start_time = time.time()
    training_stats = train_visual_network(network, train_loader, epochs=epochs, samples_per_epoch=train_samples//epochs)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Visualize training results
    print("\nVisualizing training results...")
    visualize_training_results(training_stats)
    
    # Visualize learned features at each layer
    print("\nVisualizing learned features...")
    visualize_learned_features(network)
    
    # Get some test patterns for further analysis
    test_data_iter = iter(test_loader)
    test_data, test_labels = next(test_data_iter)
    test_patterns = image_to_tryte_patterns(test_data, test_labels)
    
    # Track attractor dynamics
    print("\nTracking attractor dynamics...")
    track_attractor_dynamics(network, test_patterns)
    
    # Test the network
    print("\nTesting network...")
    start_time = time.time()
    test_stats = test_visual_network(network, test_loader, num_samples=test_samples)
    testing_time = time.time() - start_time
    print(f"Testing completed in {testing_time:.2f} seconds")
    print(f"Test accuracy: {test_stats['accuracy']:.4f}")
    
    # Visualize test results
    print("\nVisualizing test results...")
    visualize_test_results(network, test_stats)
    
    # Compare with traditional neural networks
    print("\nComparing with traditional neural networks...")
    try:
        compare_with_neural_networks(network, test_patterns, test_labels.numpy())
    except Exception as e:
        print(f"Could not complete neural network comparison: {str(e)}")
    
    # Document performance and limitations
    print("\nDocumenting performance and limitations...")
    document_performance_limitations(training_stats, test_stats, network, training_time, testing_time)
    
    print("\nMNIST demonstration completed. Results saved to disk.")


def visualize_learned_features(network, filename_prefix="learned_features"):
    """Visualize the learned features at each layer of the network.
    
    This helps understand what patterns each layer of MTUs has learned to recognize.
    
    Args:
        network: Trained MTU network
        filename_prefix: Prefix for saved visualization files
    """
    # Group MTUs by layer
    mtus_by_layer = {}
    for i, mtu in enumerate(network._mtus):
        layer = network.get_mtu_layer(i)
        if layer not in mtus_by_layer:
            mtus_by_layer[layer] = []
        mtus_by_layer[layer].append((i, mtu))
    
    # For each layer, visualize the state fields of a sample of MTUs
    for layer, mtus in mtus_by_layer.items():
        # Take up to 16 MTUs from this layer
        sample_mtus = mtus[:min(16, len(mtus))]
        
        if len(sample_mtus) == 0:
            continue
            
        # Create a grid layout
        rows = int(np.ceil(np.sqrt(len(sample_mtus))))
        cols = int(np.ceil(len(sample_mtus) / rows))
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        fig.suptitle(f"Layer {layer} Learned Features", fontsize=16)
        
        # If only one MTU, axs is not subscriptable
        if len(sample_mtus) == 1:
            axs = np.array([[axs]])
        # If only one row, make axs 2D
        elif rows == 1:
            axs = np.array([axs])
        # If only one column, make axs 2D
        elif cols == 1:
            axs = np.array([[ax] for ax in axs])
            
        # Visualize each MTU's state field
        for i, (idx, mtu) in enumerate(sample_mtus):
            row = i // cols
            col = i % cols
            
            # Get the state field
            field = getattr(mtu, '_state_field', np.zeros(network.shape))
            
            # Plot the field
            im = axs[row, col].imshow(field, cmap='viridis', interpolation='nearest')
            fig.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
            axs[row, col].set_title(f"MTU {idx}")
            axs[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(sample_mtus), rows * cols):
            row = i // cols
            col = i % cols
            axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_layer_{layer}.png")
        plt.close()


def track_attractor_dynamics(network, patterns, num_steps=20, filename="attractor_dynamics.png"):
    """Track the formation of attractor dynamics in the network.
    
    This shows how the network settles into stable patterns over time.
    
    Args:
        network: Trained MTU network
        patterns: List of (pattern, label) tuples to test
        num_steps: Number of time steps to track
        filename: Output filename
    """
    # Choose a few representative patterns
    sample_patterns = patterns[:min(4, len(patterns))]
    
    # Set up the figure
    fig, axs = plt.subplots(len(sample_patterns), num_steps, 
                          figsize=(num_steps*1.5, len(sample_patterns)*2))
    fig.suptitle("Attractor Dynamics Over Time", fontsize=16)
    
    # Make axs 2D if only one pattern
    if len(sample_patterns) == 1:
        axs = np.array([axs])
    
    # For each pattern
    for p_idx, (pattern, label) in enumerate(sample_patterns):
        # Reset the network
        network.reset()
        
        # Create initial data field state
        # Convert pattern to field
        img = np.zeros(network.shape)
        for trit in pattern.trits:
            x, y = trit.orientation
            if 0 <= x < network.shape[1] and 0 <= y < network.shape[0]:
                img[int(y), int(x)] = trit.value
                
        # Initial state
        field_states = [img.copy()]
        
        # Inject the pattern to input layer MTUs
        input_layer_mtus = [idx for idx in range(len(network._mtus)) 
                           if network.get_mtu_layer(idx) == 0]
        for mtu_idx in input_layer_mtus:
            network.inject_input(pattern, position=network._coordinates[mtu_idx])
            
        # Run for num_steps steps and track data field
        for _ in range(num_steps - 1):  # -1 because we already have the initial state
            network.step()
            # Capture the data field state
            field_states.append(network.data_field.field.copy())
            
        # Visualize the progression
        for t in range(num_steps):
            axs[p_idx, t].imshow(field_states[t], cmap='viridis', interpolation='nearest')
            axs[p_idx, t].set_title(f"t={t}") if p_idx == 0 else None
            axs[p_idx, t].axis('off')
            
        # Label the row
        axs[p_idx, 0].set_ylabel(f"Digit {label}")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_with_neural_networks(network, patterns, test_labels, filename="comparison.png"):
    """Compare the UNIFIED network with traditional neural networks.
    
    Args:
        network: Trained MTU network
        patterns: List of test patterns
        test_labels: True labels for test patterns
        filename: Output filename
    """
    # Create a simple neural network model for comparison
    try:
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.metrics import accuracy_score
        
        # A simple MLP model
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
                
        # Extract the relevant data
        # Convert patterns to flat vectors
        inputs = []
        for pattern, _ in patterns:
            img = np.zeros(network.shape)
            for trit in pattern.trits:
                x, y = trit.orientation
                if 0 <= x < network.shape[1] and 0 <= y < network.shape[0]:
                    img[int(y), int(x)] = trit.value
            inputs.append(img.flatten())
            
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(test_labels, dtype=torch.long)
        
        # Train the MLP model
        model = SimpleNN(inputs.shape[1], 128, 10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Get predictions
        mlp_outputs = model(inputs)
        _, mlp_predictions = torch.max(mlp_outputs, 1)
        mlp_accuracy = accuracy_score(labels.numpy(), mlp_predictions.numpy())
        
        # Get UNIFIED predictions
        unified_predictions = []
        for pattern, _ in patterns:
            network.reset()
            
            # Inject pattern
            input_layer_mtus = [idx for idx in range(len(network._mtus)) 
                               if network.get_mtu_layer(idx) == 0]
            for mtu_idx in input_layer_mtus:
                network.inject_input(pattern, position=network._coordinates[mtu_idx])
                
            # Run steps
            outputs_by_layer = {}
            for step in range(10):
                step_outputs = network.step()
                for mtu_idx, output in step_outputs:
                    mtu_layer = network.get_mtu_layer(mtu_idx)
                    if mtu_layer not in outputs_by_layer:
                        outputs_by_layer[mtu_layer] = []
                    outputs_by_layer[mtu_layer].append((mtu_idx, output))
                    
            # Get output layer prediction
            output_layer = network.num_layers - 1
            output_mtus = outputs_by_layer.get(output_layer, [])
            
            if output_mtus:
                most_active_mtu = max(output_mtus, key=lambda x: x[1].energy)
                predicted_digit = network.get_mtu_index_in_layer(most_active_mtu[0], output_layer)
                unified_predictions.append(predicted_digit)
            else:
                unified_predictions.append(-1)  # No prediction
                
        unified_accuracy = accuracy_score(test_labels, unified_predictions)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(['MLP', 'UNIFIED'], [mlp_accuracy, unified_accuracy])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: MLP vs UNIFIED')
        plt.ylim(0, 1)
        plt.savefig(filename)
        plt.close()
        
        print(f"MLP Accuracy: {mlp_accuracy:.4f}")
        print(f"UNIFIED Accuracy: {unified_accuracy:.4f}")
        
        return mlp_accuracy, unified_accuracy
        
    except ImportError:
        print("Could not import PyTorch for comparison. Skipping comparison.")
        return 0, 0


def initialize_digit_templates(network):
    """Initialize the network with targeted templates for each digit class.
    
    This pre-initializes MTUs in the output layer to be sensitive to specific digit patterns,
    providing a better starting point for learning.
    
    Args:
        network: The MTU network to initialize
    """
    # Define simplified templates for each digit (0-9)
    digit_templates = {
        0: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]),
        1: np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0]
        ]),
        2: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1]
        ]),
        3: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]),
        4: np.array([
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0]
        ]),
        5: np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0]
        ]),
        6: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]),
        7: np.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        ]),
        8: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ]),
        9: np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ])
    }
    
    # Scale templates to match network's field size
    scaled_templates = {}
    for digit, template in digit_templates.items():
        # Use scipy zoom for better interpolation
        from scipy.ndimage import zoom
        scale_factor = (network.shape[0] / template.shape[0], network.shape[1] / template.shape[1])
        scaled_templates[digit] = zoom(template, scale_factor, order=1)
    
    # Get MTUs in the output layer
    output_layer = network.num_layers - 1
    output_mtus = [i for i, _ in enumerate(network._mtus) 
                   if network.get_mtu_layer(i) == output_layer]
    
    # If we have at least 10 output MTUs, initialize them with digit templates
    if len(output_mtus) >= 10:
        for digit, template in scaled_templates.items():
            if digit < len(output_mtus):
                mtu_idx = output_mtus[digit]
                mtu = network._mtus[mtu_idx]
                
                # Convert template to pattern
                trits = []
                for y in range(template.shape[0]):
                    for x in range(template.shape[1]):
                        value = template[y, x]
                        if value > 0.5:
                            # Create positive trit
                            trit = Trit(state=TritState.POSITIVE)
                            trit.orientation = (float(x), float(y))
                            trits.append(trit)
                        elif value < -0.5:
                            # Create negative trit
                            trit = Trit(state=TritState.NEGATIVE)
                            trit.orientation = (float(x), float(y))
                            trits.append(trit)
                
                # Create template pattern
                pattern = Tryte(trits)
                
                # Initialize MTU with this pattern
                if hasattr(mtu, '_pattern_memory') and mtu._pattern_memory:
                    # Store this as a known pattern
                    mtu._pattern_memory.learn_pattern(pattern)
                    mtu._known_patterns.append(pattern)
                    
                # Also set the state field based on the template
                state_field = np.zeros(network.shape)
                for y in range(template.shape[0]):
                    for x in range(template.shape[1]):
                        state_field[y, x] = template[y, x]                   
                mtu._state_field = state_field
                
                print(f"Initialized output MTU {mtu_idx} with template for digit {digit}")


def train_visual_network_with_curriculum(network, train_loader, epochs=5, samples_per_epoch=100):
    """Train the MTU network using curriculum learning for better pattern recognition.
    
    This starts with easier examples and gradually increases difficulty, with reinforcement
    for correct classifications.
    
    Args:
        network: MTU network to train
        train_loader: DataLoader with training data
        epochs: Number of training epochs
        samples_per_epoch: Number of samples to use per epoch
        
    Returns:
        training_stats: Dictionary with training statistics
    """
    training_stats = {
        'accuracy_per_epoch': [],
        'loss_per_epoch': [],
        'connection_strength_changes': [],
        'learning_rate_per_epoch': []
    }
    
    print(f"Training network with curriculum learning for {epochs} epochs...")
    
    # Set up digit difficulty ranking (based on visual complexity)
    # From easiest to hardest: 1, 7, 4, 0, 9, 3, 5, 2, 8, 6
    digit_difficulty = {1: 0, 7: 1, 4: 2, 0: 3, 9: 4, 3: 5, 5: 6, 2: 7, 8: 8, 6: 9}
    max_difficulty = max(digit_difficulty.values())
    
    for epoch in range(epochs):
        correct = 0
        total = 0
        epoch_loss = 0
        initial_strengths = {}
        
        # Determine current difficulty level (gradually increasing)
        current_max_difficulty = int((epoch + 1) / epochs * max_difficulty)
        allowed_digits = [d for d, diff in digit_difficulty.items() if diff <= current_max_difficulty]
        
        print(f"Epoch {epoch+1}/{epochs} - Working with digits: {allowed_digits}")
        
        # Store initial connection strengths
        for mtu_idx in range(len(network._mtus)):
            if mtu_idx in network._connections:
                initial_strengths[mtu_idx] = {}
                for target in network._connections[mtu_idx]:
                    if (mtu_idx, target) in network._connection_strengths:
                        initial_strengths[mtu_idx][target] = network._connection_strengths[(mtu_idx, target)]
        
        # Get training data for this epoch
        data_iter = iter(train_loader)
        
        # Training batch loop
        for batch_idx in range(samples_per_epoch // train_loader.batch_size + 1):
            try:
                # Get a batch
                data, target = next(data_iter)
                
                # Filter for allowed digits based on curriculum
                mask = torch.tensor([t.item() in allowed_digits for t in target])
                if not mask.any():
                    continue  # Skip this batch if no allowed digits
                    
                filtered_data = data[mask]
                filtered_target = target[mask]
                
                if len(filtered_data) == 0:
                    continue
                
                # Apply data augmentation
                augmented_data = torch.cat([apply_data_augmentation(img) for img in filtered_data])
                
                # Convert to tryte patterns
                patterns = image_to_tryte_patterns(augmented_data, filtered_target)
                
                # Train on each pattern in the batch
                for pattern_idx, (pattern, label) in enumerate(patterns):
                    # Clear the network activity
                    network.reset()
                    
                    # Inject the pattern to input layer MTUs
                    input_layer_mtus = [idx for idx in range(len(network._mtus)) 
                                       if network.get_mtu_layer(idx) == 0]
                    for mtu_idx in input_layer_mtus:
                        network.inject_input(pattern, position=network._coordinates[mtu_idx])
                    
                    # Run the network for a fixed number of steps
                    recognition_steps = 10
                    outputs_by_layer = {}
                    
                    for step in range(recognition_steps):
                        step_outputs = network.step()
                        
                        # Group outputs by layer
                        for mtu_idx, output in step_outputs:
                            mtu_layer = network.get_mtu_layer(mtu_idx)
                            if mtu_layer not in outputs_by_layer:
                                outputs_by_layer[mtu_layer] = []
                            outputs_by_layer[mtu_layer].append((mtu_idx, output))
                    
                    # Check the output layer activations
                    output_layer = network.num_layers - 1
                    output_mtus = outputs_by_layer.get(output_layer, [])
                    
                    # Determine if prediction was correct
                    prediction_correct = False
                    predicted_digit = -1
                    
                    if output_mtus:
                        # Get the most active MTU in the output layer
                        most_active_mtu = max(output_mtus, key=lambda x: x[1].energy)
                        predicted_digit = network.get_mtu_index_in_layer(most_active_mtu[0], output_layer)
                        
                        # Check if the prediction matches the label
                        if predicted_digit == label:
                            correct += 1
                            prediction_correct = True
                    
                    total += 1
                    
                    # Compute a simple loss based on the output layer activation
                    target_activation = 0
                    for mtu_idx, output in output_mtus:
                        mtu_digit = network.get_mtu_index_in_layer(mtu_idx, output_layer)
                        if mtu_digit == label:
                            target_activation += output.energy
                    
                    loss = 1.0 - (target_activation / (len(output_mtus) + 1e-6))
                    epoch_loss += loss
                    
                    # Apply reinforcement signal based on correctness
                    # This is a form of reward-based learning
                    if prediction_correct:
                        # Strong reinforcement for correct predictions
                        reinforcement = 0.9
                    else:
                        # Weaker reinforcement signal for incorrect predictions
                        reinforcement = 0.1
                        
                        # Apply corrective reinforcement by boosting the correct output MTU
                        output_layer_mtus = [i for i, _ in enumerate(network._mtus) 
                                           if network.get_mtu_layer(i) == output_layer]
                        
                        if label < len(output_layer_mtus):
                            correct_mtu_idx = output_layer_mtus[label]
                            
                            # Explicitly activate the correct MTU
                            correct_mtu = network._mtus[correct_mtu_idx]
                            correct_mtu._is_active = True
                            
                            # Also boost its state
                            if hasattr(correct_mtu, '_state_field'):
                                correct_mtu._state_field *= 1.5
                    
                    # Provide performance metric for adaptive learning
                    network.hebbian_update(performance_metric=reinforcement)
                    
                    # Visualize every 20th pattern
                    if pattern_idx % 20 == 0:
                        # Visualize the input pattern
                        plt.figure(figsize=(6, 6))
                        img = np.zeros(network.shape)
                        for trit in pattern.trits:
                            x, y = trit.orientation
                            if 0 <= x < network.shape[1] and 0 <= y < network.shape[0]:
                                img[int(y), int(x)] = trit.value
                        
                        plt.imshow(img, cmap='gray')
                        plt.title(f"Input Pattern (Digit {label}), Predicted: {predicted_digit}")
                        plt.colorbar()
                        plt.savefig(f"mnist_input_epoch{epoch}_batch{batch_idx}.png")
                        plt.close()
                        
                        # Visualize network activity
                        fig = network.visualize()
                        plt.savefig(f"network_activity_epoch{epoch}_batch{batch_idx}.png")
                        plt.close()
            
            except StopIteration:
                break
        
        # Calculate epoch statistics
        epoch_accuracy = correct / total if total > 0 else 0
        epoch_loss = epoch_loss / total if total > 0 else float('inf')
        
        training_stats['accuracy_per_epoch'].append(epoch_accuracy)
        training_stats['loss_per_epoch'].append(epoch_loss)
        
        # Get current learning rate
        current_lr = network._adaptive_learning.get_current_rate() if network._adaptive_learning else 0.01
        training_stats['learning_rate_per_epoch'].append(current_lr)
        
        # Calculate connection strength changes
        strength_changes = {}
        for mtu_idx in initial_strengths:
            if mtu_idx in network._connections:
                changes = []
                for target in initial_strengths[mtu_idx]:
                    if (mtu_idx, target) in network._connection_strengths:
                        initial = initial_strengths[mtu_idx][target]
                        final = network._connection_strengths[(mtu_idx, target)]
                        changes.append(final - initial)
                if changes:
                    strength_changes[mtu_idx] = np.mean(changes)
        
        avg_strength_change = np.mean(list(strength_changes.values())) if strength_changes else 0
        training_stats['connection_strength_changes'].append(avg_strength_change)
        
        print(f"Epoch {epoch+1}/{epochs} - Accuracy: {epoch_accuracy:.4f}, Loss: {epoch_loss:.4f}, "
              f"LR: {current_lr:.6f}, Avg Connection Change: {avg_strength_change:.6f}")
        
        # Visualize network after each epoch
        fig = network.visualize()
        plt.savefig(f"mnist_network_epoch{epoch+1}.png")
        plt.close()
    
    return training_stats


if __name__ == "__main__":
    # Run with enhanced but compatible parameters
    run_mnist_demonstration(epochs=3, train_samples=300, test_samples=100)
