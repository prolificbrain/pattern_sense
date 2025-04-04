"""ECG Anomaly Detection using UNIFIED Consciousness Engine.

This example demonstrates the application of the UNIFIED temporal pattern recognition
system to ECG data for arrhythmia detection and anomaly identification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from unified.mtu.temporal import TemporalPatternMemory, TimeSeriesPredictor
from unified.mtu.hierarchical import HierarchicalPatternNetwork

# Constants
DOWNLOAD_URL = "https://physionet.org/content/mitdb/1.0.0/"
WINDOW_SIZE = 64  # Reduced from 128 for faster processing
STEP_SIZE = 32
SAMPLE_COUNT = 10  # Reduced sample count for faster testing


def download_sample_data(target_dir="./data/ecg", sample_count=SAMPLE_COUNT):
    """Download sample ECG data.
    
    For a real application, download the MIT-BIH Arrhythmia Database from:
    https://physionet.org/content/mitdb/1.0.0/
    
    This function simulates the download with synthetic data.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Create placeholder data (in real app, download actual ECG data)
    print(f"In a real application, download ECG data from {DOWNLOAD_URL}")
    print(f"For this demonstration, creating synthetic ECG data in {target_dir}")
    
    # Generate synthetic ECG-like data
    for i in range(sample_count):
        # Create time axis (10 seconds at 250 Hz)
        t = np.linspace(0, 10, 2500)
        
        # Generate normal ECG-like signal
        if i % 3 != 0:  # Normal rhythm for 2/3 of samples
            # Base frequency (heart rate ~60-80 bpm)
            freq = 1.1 + 0.2 * np.random.rand()  # ~66-78 bpm
            
            # Generate PQRST-like complexes
            signal = np.zeros_like(t)
            
            # Add periodic complexes
            for j in range(int(10 * freq)):
                # Center of complex
                tc = j / freq
                
                # P wave (small positive deflection)
                p_wave = 0.2 * np.exp(-((t - tc + 0.2) ** 2) / 0.005)
                
                # QRS complex (large deflection)
                qrs = -0.3 * np.exp(-((t - tc + 0.05) ** 2) / 0.001) + \
                       1.0 * np.exp(-((t - tc) ** 2) / 0.0005) + \
                      -0.3 * np.exp(-((t - tc - 0.05) ** 2) / 0.001)
                
                # T wave (medium positive deflection)
                t_wave = 0.4 * np.exp(-((t - tc - 0.2) ** 2) / 0.01)
                
                # Combine components
                complex_wave = p_wave + qrs + t_wave
                
                # Add to signal where time matches
                mask = (t >= tc - 0.4) & (t <= tc + 0.4)
                signal[mask] += complex_wave[mask]
            
            label = "normal"
            
        else:  # Abnormal rhythm (arrhythmia)
            # Base frequency with irregularity
            freq = 1.1 + 0.2 * np.random.rand()  # Base rate
            
            # Generate PQRST-like complexes with irregularities
            signal = np.zeros_like(t)
            
            # Add periodic complexes with irregularities
            tc_prev = 0
            for j in range(int(10 * freq)):
                # Irregular timing
                if j > 0 and j % 3 == 0:  # Every third beat is irregular
                    tc = tc_prev + (1/freq) * (0.5 + 1.0 * np.random.rand())  # Premature or delayed
                else:
                    tc = j / freq
                
                tc_prev = tc
                
                # Skip if beyond time range
                if tc > 10:
                    continue
                
                # P wave (may be missing in some arrhythmias)
                if j % 4 != 0:  # P wave present most of the time
                    p_wave = 0.2 * np.exp(-((t - tc + 0.2) ** 2) / 0.005)
                else:
                    p_wave = 0  # Missing P wave
                
                # QRS complex (may be wider or taller in some arrhythmias)
                if j % 5 == 0:  # Wider QRS occasionally
                    qrs = -0.3 * np.exp(-((t - tc + 0.05) ** 2) / 0.002) + \
                           1.2 * np.exp(-((t - tc) ** 2) / 0.001) + \
                          -0.3 * np.exp(-((t - tc - 0.05) ** 2) / 0.002)
                else:
                    qrs = -0.3 * np.exp(-((t - tc + 0.05) ** 2) / 0.001) + \
                           1.0 * np.exp(-((t - tc) ** 2) / 0.0005) + \
                          -0.3 * np.exp(-((t - tc - 0.05) ** 2) / 0.001)
                
                # T wave
                t_wave = 0.4 * np.exp(-((t - tc - 0.2) ** 2) / 0.01)
                
                # Combine components
                complex_wave = p_wave + qrs + t_wave
                
                # Add to signal where time matches
                mask = (t >= tc - 0.4) & (t <= tc + 0.4)
                signal[mask] += complex_wave[mask]
            
            label = "abnormal"
        
        # Add noise
        signal += 0.05 * np.random.randn(len(t))
        
        # Save signal and label
        np.savez(os.path.join(target_dir, f"ecg_{i:03d}_{label}.npz"),
                signal=signal, label=label)
    
    return target_dir


def load_ecg_data(data_dir):
    """Load ECG signals and labels from directory."""
    signals = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".npz"):
            # Extract label from filename
            label = "abnormal" if "abnormal" in filename else "normal"
            
            # Load signal
            data = np.load(os.path.join(data_dir, filename))
            signal = data['signal']
            
            signals.append(signal)
            labels.append(1 if label == "abnormal" else 0)
    
    return signals, np.array(labels)


def extract_windows(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """Extract overlapping windows from signal."""
    windows = []
    for i in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[i:i+window_size])
    return windows


def train_temporal_model(signals, labels):
    """Train temporal pattern recognition model on ECG data."""
    # Initialize temporal memory
    temporal_memory = TemporalPatternMemory(max_patterns=5000, max_sequences=500)
    
    # Track indices for normal and abnormal patterns
    normal_indices = set()
    abnormal_indices = set()
    
    # Process each signal
    for signal, label in zip(signals, labels):
        # Extract windows
        windows = extract_windows(signal)
        
        # Process each window
        for window in windows:
            # Reshape window to 2D array (required for pattern memory)
            pattern = window.reshape(-1, 1)
            
            # Observe pattern
            pattern_idx, _ = temporal_memory.observe_pattern(pattern)
            
            # Track indices by label
            if label == 1:  # Abnormal
                abnormal_indices.add(pattern_idx)
            else:  # Normal
                normal_indices.add(pattern_idx)
    
    return temporal_memory, normal_indices, abnormal_indices


def train_hierarchical_model(signals, labels):
    """Train hierarchical pattern recognition model on ECG data."""
    # Initialize hierarchical network
    hierarchical_network = HierarchicalPatternNetwork(
        input_dimensions=(WINDOW_SIZE, 1),
        max_levels=2,  # Reduced from 3 to 2 levels
        patterns_per_level=500  # Reduced pattern count
    )
    
    # Track indices for normal and abnormal patterns at each level
    normal_indices = {level: set() for level in range(2)}
    abnormal_indices = {level: set() for level in range(2)}
    
    # Process only a subset of windows for faster training
    for signal, label in zip(signals, labels):
        # Extract fewer windows
        windows = extract_windows(signal, window_size=WINDOW_SIZE, step_size=STEP_SIZE*2)
        
        # Process only first few windows
        for window in windows[:5]:  # Limit to 5 windows per signal
            pattern = window.reshape(WINDOW_SIZE, 1)
            level_indices = hierarchical_network.learn_pattern(pattern, max_level=1)  # Only learn up to level 1
            
            # Track indices by label
            for level, idx in level_indices.items():
                if label == 1:  # Abnormal
                    abnormal_indices[level].add(idx)
                else:  # Normal
                    normal_indices[level].add(idx)
    
    return hierarchical_network, normal_indices, abnormal_indices


def train_time_series_predictor(signals, labels):
    """Train time series predictor on ECG data."""
    # Initialize predictor
    predictor = TimeSeriesPredictor(
        window_size=WINDOW_SIZE,
        prediction_horizon=WINDOW_SIZE // 2,
        step_size=STEP_SIZE
    )
    
    # Concatenate all normal signals for training
    normal_signals = [signal for signal, label in zip(signals, labels) if label == 0]
    if normal_signals:
        # Concatenate with small gaps
        training_data = np.concatenate(normal_signals)
        
        # Train predictor
        predictor.train(training_data)
    
    return predictor


def detect_anomalies_temporal(temporal_memory, test_signals, normal_indices, abnormal_indices):
    """Detect anomalies in test signals using temporal pattern memory."""
    results = []
    
    for signal in test_signals:
        # Extract windows
        windows = extract_windows(signal)
        
        # Process each window
        window_results = []
        for window in windows:
            # Reshape window
            pattern = window.reshape(-1, 1)
            
            # Recognize pattern
            pattern_idx, sequence_matches = temporal_memory.observe_pattern(pattern)
            
            # Check if pattern is abnormal
            if pattern_idx in abnormal_indices:
                is_abnormal = True
                confidence = 1.0
            elif pattern_idx in normal_indices:
                is_abnormal = False
                confidence = 1.0
            elif sequence_matches:
                # Use sequence matching for unknown patterns
                abnormal_sim = 0
                normal_sim = 0
                
                for seq_idx, sim in sequence_matches:
                    # Check patterns in sequence
                    seq = temporal_memory._sequence_memory[seq_idx]
                    abnormal_count = sum(1 for p in seq if p in abnormal_indices)
                    normal_count = sum(1 for p in seq if p in normal_indices)
                    
                    if abnormal_count > normal_count:
                        abnormal_sim += sim
                    else:
                        normal_sim += sim
                
                is_abnormal = abnormal_sim > normal_sim
                confidence = abs(abnormal_sim - normal_sim) / (abnormal_sim + normal_sim + 1e-10)
            else:
                # No matches, default to normal
                is_abnormal = False
                confidence = 0.5
            
            window_results.append((is_abnormal, confidence))
        
        # Aggregate window results
        abnormal_windows = sum(1 for is_abnormal, _ in window_results if is_abnormal)
        total_windows = len(window_results)
        abnormal_ratio = abnormal_windows / total_windows if total_windows > 0 else 0
        
        # Signal is abnormal if enough windows are abnormal
        is_signal_abnormal = abnormal_ratio > 0.3
        confidence = abs(abnormal_ratio - 0.3) / 0.7  # Scale to 0-1
        
        results.append({
            'prediction': 1 if is_signal_abnormal else 0,
            'confidence': confidence,
            'abnormal_ratio': abnormal_ratio
        })
    
    return results


def detect_anomalies_hierarchical(hierarchical_network, test_signals, normal_indices, abnormal_indices):
    """Detect anomalies in test signals using hierarchical pattern network."""
    results = []
    
    for signal in test_signals:
        # Extract windows
        windows = extract_windows(signal)
        
        # Process each window
        window_results = []
        for window in windows:
            # Reshape window
            pattern = window.reshape(WINDOW_SIZE, 1)
            
            # Recognize pattern hierarchically
            hierarchy_results = hierarchical_network.recognize_pattern(pattern)
            
            # Check results at each level
            level_abnormal = [False] * len(hierarchy_results)
            level_confidence = [0.0] * len(hierarchy_results)
            
            for level, matches in hierarchy_results.items():
                if not matches:
                    continue
                
                # Check if best match is abnormal
                best_match_idx = matches[0][0]
                best_match_sim = matches[0][1]
                
                if best_match_idx in abnormal_indices[level]:
                    level_abnormal[level] = True
                    level_confidence[level] = best_match_sim
                elif best_match_idx in normal_indices[level]:
                    level_abnormal[level] = False
                    level_confidence[level] = best_match_sim
            
            # Higher levels have more weight
            weighted_abnormal = sum(level * is_abnormal for level, is_abnormal in enumerate(level_abnormal))
            weighted_confidence = sum(level * conf for level, conf in enumerate(level_confidence))
            total_weight = sum(level for level in range(len(level_abnormal)))
            
            # Window is abnormal if weighted sum is high enough
            is_window_abnormal = weighted_abnormal > total_weight / 2
            confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            
            window_results.append((is_window_abnormal, confidence))
        
        # Aggregate window results
        abnormal_windows = sum(1 for is_abnormal, _ in window_results if is_abnormal)
        total_windows = len(window_results)
        abnormal_ratio = abnormal_windows / total_windows if total_windows > 0 else 0
        
        # Signal is abnormal if enough windows are abnormal
        is_signal_abnormal = abnormal_ratio > 0.3
        confidence = abs(abnormal_ratio - 0.3) / 0.7  # Scale to 0-1
        
        results.append({
            'prediction': 1 if is_signal_abnormal else 0,
            'confidence': confidence,
            'abnormal_ratio': abnormal_ratio
        })
    
    return results


def detect_anomalies_prediction(predictor, test_signals):
    """Detect anomalies in test signals using prediction-based approach."""
    results = []
    
    for signal in test_signals:
        # Extract windows with enough space for prediction
        windows = extract_windows(signal, window_size=WINDOW_SIZE, step_size=WINDOW_SIZE)
        
        # Process each window
        window_errors = []
        for i, window in enumerate(windows[:-1]):  # Skip last window
            # Predict next window
            prediction = predictor.predict(window)
            
            # Get actual next window
            next_window = signal[WINDOW_SIZE*(i+1):WINDOW_SIZE*(i+2)]
            if len(next_window) < len(prediction):
                continue
            
            # Calculate prediction error
            error = np.mean((next_window[:len(prediction)] - prediction) ** 2)
            window_errors.append(error)
        
        if not window_errors:
            # No predictions made
            results.append({
                'prediction': 0,  # Default to normal
                'confidence': 0.5,
                'mean_error': 0.0
            })
            continue
        
        # Calculate mean error
        mean_error = np.mean(window_errors)
        
        # Signal is abnormal if mean error is high
        # Threshold determined empirically
        error_threshold = 0.2
        is_signal_abnormal = mean_error > error_threshold
        confidence = min(1.0, mean_error / (2 * error_threshold))  # Scale to 0-1
        
        results.append({
            'prediction': 1 if is_signal_abnormal else 0,
            'confidence': confidence,
            'mean_error': mean_error
        })
    
    return results


def evaluate_performance(results, true_labels):
    """Evaluate anomaly detection performance."""
    predictions = [r['prediction'] for r in results]
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def visualize_results(signals, results, true_labels, num_samples=2):
    """Visualize anomaly detection results."""
    # Select random samples to visualize
    indices = np.random.choice(len(signals), min(num_samples, len(signals)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get signal and results
        signal = signals[idx]
        true_label = "Abnormal" if true_labels[idx] == 1 else "Normal"
        pred = results[idx]['prediction']
        conf = results[idx]['confidence']
        pred_label = "Abnormal" if pred == 1 else "Normal"
        
        # Plot signal
        t = np.arange(len(signal)) / 250  # Assuming 250 Hz sampling rate
        axes[i].plot(t, signal)
        
        # Add colored background based on prediction
        if pred == 1:  # Abnormal prediction
            axes[i].axvspan(0, t[-1], alpha=0.2, color='red')
        else:  # Normal prediction
            axes[i].axvspan(0, t[-1], alpha=0.2, color='green')
        
        # Add labels
        axes[i].set_title(f"True: {true_label}, Predicted: {pred_label} (Confidence: {conf:.2f})")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
    
    plt.tight_layout()
    plt.savefig('ecg_results.png')
    plt.close()


def run_ecg_anomaly_detection():
    """Run the complete ECG anomaly detection demo."""
    print("UNIFIED Consciousness Engine - ECG Anomaly Detection")
    print("===================================================\n")
    
    # Step 1: Prepare data
    print("Step 1: Preparing ECG data...")
    data_dir = download_sample_data(sample_count=SAMPLE_COUNT)
    signals, labels = load_ecg_data(data_dir)
    
    # Downsample signals for faster processing
    signals = [signal[::2] for signal in signals]  # Take every other sample
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.3, random_state=42)
    
    print(f"Loaded {len(signals)} signals ({sum(labels)} abnormal, {len(labels)-sum(labels)} normal)")
    print(f"Training set: {len(X_train)} signals, Test set: {len(X_test)} signals\n")
    
    # Step 2: Train models
    print("Step 2: Training pattern recognition models...")
    
    # Train temporal model
    print("Training temporal pattern memory...")
    temporal_memory, normal_indices_t, abnormal_indices_t = train_temporal_model(X_train, y_train)
    
    # Train hierarchical model
    print("Training hierarchical pattern network...")
    hierarchical_network, normal_indices_h, abnormal_indices_h = train_hierarchical_model(X_train, y_train)
    
    # Train time series predictor
    print("Training time series predictor...")
    predictor = train_time_series_predictor(X_train, y_train)
    
    print("\nTraining complete.")
    print(f"Temporal memory: {len(temporal_memory._pattern_memory._patterns)} patterns learned")
    print(f"Hierarchical network: {sum(len(level._patterns) for level in hierarchical_network.levels)} patterns learned across {len(hierarchical_network.levels)} levels")
    
    # Step 3: Detect anomalies
    print("\nStep 3: Detecting anomalies in test signals...")
    
    # Detect using temporal memory
    print("Using temporal pattern memory...")
    results_temporal = detect_anomalies_temporal(temporal_memory, X_test, normal_indices_t, abnormal_indices_t)
    
    # Detect using hierarchical network
    print("Using hierarchical pattern network...")
    results_hierarchical = detect_anomalies_hierarchical(hierarchical_network, X_test, normal_indices_h, abnormal_indices_h)
    
    # Skip prediction-based approach for faster execution
    print("Skipping prediction-based approach for faster execution...")
    results_prediction = [{'prediction': 0, 'confidence': 0.5} for _ in X_test]
    
    # Step 4: Evaluate performance
    print("\nStep 4: Evaluating performance...")
    
    metrics_temporal = evaluate_performance(results_temporal, y_test)
    metrics_hierarchical = evaluate_performance(results_hierarchical, y_test)
    metrics_prediction = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}  # Placeholder
    
    print("\nTemporal Pattern Memory:")
    print(f"Accuracy: {metrics_temporal['accuracy']:.2f}")
    print(f"Precision: {metrics_temporal['precision']:.2f}")
    print(f"Recall: {metrics_temporal['recall']:.2f}")
    print(f"F1 Score: {metrics_temporal['f1_score']:.2f}")
    
    print("\nHierarchical Pattern Network:")
    print(f"Accuracy: {metrics_hierarchical['accuracy']:.2f}")
    print(f"Precision: {metrics_hierarchical['precision']:.2f}")
    print(f"Recall: {metrics_hierarchical['recall']:.2f}")
    print(f"F1 Score: {metrics_hierarchical['f1_score']:.2f}")
    
    # Step 5: Visualize results
    print("\nStep 5: Visualizing results...")
    visualize_results(X_test, results_hierarchical, y_test, num_samples=2)  # Reduced from 4 to 2 samples
    print("Results visualization saved to 'ecg_results.png'")
    
    return {
        'temporal': metrics_temporal,
        'hierarchical': metrics_hierarchical,
        'prediction': metrics_prediction
    }


if __name__ == "__main__":
    run_ecg_anomaly_detection()
