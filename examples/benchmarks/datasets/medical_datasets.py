"""Medical dataset loaders for UNIFIED Consciousness Engine benchmarks.

This module provides functions to load and preprocess real-world medical
datasets for benchmarking pattern recognition algorithms.
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io


def load_dataset(dataset_name, data_dir="./data"):
    """Load and preprocess a medical dataset.
    
    Args:
        dataset_name: Name of dataset to load
        data_dir: Directory to store downloaded data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Load dataset based on name
    if dataset_name == 'breast_cancer':
        return load_breast_cancer_dataset()
    elif dataset_name == 'diabetes':
        return load_diabetes_dataset()
    elif dataset_name == 'ecg':
        return load_ecg_dataset(data_dir)
    elif dataset_name == 'skin_lesion':
        return load_skin_lesion_dataset(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_breast_cancer_dataset():
    """Load breast cancer dataset from scikit-learn.
    
    Returns:
        Tuple of (X, y, metadata)
    """
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for pattern recognition (2D patterns)
    X_reshaped = []
    for x in X:
        # Reshape to 2D grid (approximately square)
        size = int(np.ceil(np.sqrt(len(x))))
        x_pad = np.zeros(size * size)
        x_pad[:len(x)] = x
        X_reshaped.append(x_pad.reshape(size, size))
    
    # Create metadata
    metadata = {
        'name': 'Breast Cancer Wisconsin',
        'description': 'Diagnostic breast cancer dataset',
        'n_samples': len(X),
        'n_features': data.data.shape[1],
        'classes': data.target_names.tolist(),
        'feature_names': data.feature_names.tolist()
    }
    
    return np.array(X_reshaped), y, metadata


def load_diabetes_dataset():
    """Load diabetes dataset from scikit-learn.
    
    Returns:
        Tuple of (X, y, metadata)
    """
    # Load dataset
    data = load_diabetes()
    X = data.data
    
    # Convert regression target to binary classification
    # (diabetes progression above/below median)
    median = np.median(data.target)
    y = (data.target > median).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Reshape for pattern recognition (2D patterns)
    X_reshaped = []
    for x in X:
        # Reshape to 2D grid (approximately square)
        size = int(np.ceil(np.sqrt(len(x))))
        x_pad = np.zeros(size * size)
        x_pad[:len(x)] = x
        X_reshaped.append(x_pad.reshape(size, size))
    
    # Create metadata
    feature_names = data.feature_names
    if hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    
    metadata = {
        'name': 'Diabetes',
        'description': 'Diabetes progression dataset (binarized)',
        'n_samples': len(X),
        'n_features': data.data.shape[1],
        'classes': ['Normal', 'Diabetes'],
        'feature_names': feature_names
    }
    
    return np.array(X_reshaped), y, metadata


def load_ecg_dataset(data_dir):
    """Load ECG dataset (MIT-BIH Arrhythmia).
    
    If the dataset is not available locally, it will generate synthetic data.
    
    Args:
        data_dir: Directory to store data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    ecg_dir = os.path.join(data_dir, 'ecg')
    os.makedirs(ecg_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(ecg_dir, 'mitbih_train.csv')):
        try:
            # Try to download MIT-BIH dataset
            print("Downloading MIT-BIH Arrhythmia dataset...")
            url = "https://storage.googleapis.com/mitbih_database/mitbih_train.csv"
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(ecg_dir, 'mitbih_train.csv'), 'wb') as f:
                    f.write(response.content)
                print("Download complete.")
            else:
                print("Failed to download dataset. Generating synthetic data instead.")
                return _generate_synthetic_ecg(ecg_dir)
        except Exception as e:
            print(f"Error downloading dataset: {e}. Generating synthetic data instead.")
            return _generate_synthetic_ecg(ecg_dir)
    
    # Load dataset
    try:
        df = pd.read_csv(os.path.join(ecg_dir, 'mitbih_train.csv'), header=None)
        
        # Last column is the class label
        X = df.iloc[:, :-1].values
        y_multi = df.iloc[:, -1].values
        
        # Convert to binary classification (normal vs abnormal)
        # Class 0 is normal, others are abnormal
        y = (y_multi > 0).astype(int)
        
        # Reshape for pattern recognition (2D patterns)
        X_reshaped = []
        for x in X:
            # Reshape to 2D grid (approximately square)
            size = int(np.ceil(np.sqrt(len(x))))
            x_pad = np.zeros(size * size)
            x_pad[:len(x)] = x
            X_reshaped.append(x_pad.reshape(size, size))
        
        # Create metadata
        metadata = {
            'name': 'MIT-BIH Arrhythmia',
            'description': 'ECG heartbeat categorization dataset',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'classes': ['Normal', 'Abnormal'],
            'original_classes': ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
        }
        
        return np.array(X_reshaped), y, metadata
    
    except Exception as e:
        print(f"Error loading dataset: {e}. Generating synthetic data instead.")
        return _generate_synthetic_ecg(ecg_dir)


def _generate_synthetic_ecg(ecg_dir):
    """Generate synthetic ECG data for benchmarking.
    
    Args:
        ecg_dir: Directory to store data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    print("Generating synthetic ECG data...")
    
    # Parameters
    n_samples = 1000
    sequence_length = 187  # Typical ECG sequence length
    
    # Generate data
    X = []
    y = []
    
    for i in range(n_samples):
        # Generate time axis
        t = np.linspace(0, 10, sequence_length)
        
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
            
            y.append(0)  # Normal
            
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
            
            y.append(1)  # Abnormal
        
        # Add noise
        signal += 0.05 * np.random.randn(len(t))
        
        # Reshape to 2D grid
        size = int(np.ceil(np.sqrt(len(signal))))
        signal_pad = np.zeros(size * size)
        signal_pad[:len(signal)] = signal
        X.append(signal_pad.reshape(size, size))
    
    # Create metadata
    metadata = {
        'name': 'Synthetic ECG',
        'description': 'Synthetic ECG data for benchmarking',
        'n_samples': n_samples,
        'n_features': sequence_length,
        'classes': ['Normal', 'Abnormal'],
        'synthetic': True
    }
    
    return np.array(X), np.array(y), metadata


def load_skin_lesion_dataset(data_dir):
    """Load skin lesion dataset (HAM10000).
    
    If the dataset is not available locally, it will generate synthetic data.
    
    Args:
        data_dir: Directory to store data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    skin_dir = os.path.join(data_dir, 'skin_lesion')
    os.makedirs(skin_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(os.path.join(skin_dir, 'HAM10000_metadata.csv')):
        try:
            # Try to download a small subset of HAM10000 dataset
            print("Downloading HAM10000 metadata...")
            metadata_url = "https://raw.githubusercontent.com/ptschandl/HAM10000_dataset/master/HAM10000_metadata.csv"
            response = requests.get(metadata_url)
            if response.status_code == 200:
                with open(os.path.join(skin_dir, 'HAM10000_metadata.csv'), 'wb') as f:
                    f.write(response.content)
                print("Metadata download complete.")
                
                # For the actual images, we would need to download from Harvard Dataverse
                # which requires authentication. For this benchmark, we'll use synthetic data.
                print("Using synthetic skin lesion data for benchmarking.")
                return _generate_synthetic_skin_lesion(skin_dir)
            else:
                print("Failed to download dataset. Generating synthetic data instead.")
                return _generate_synthetic_skin_lesion(skin_dir)
        except Exception as e:
            print(f"Error downloading dataset: {e}. Generating synthetic data instead.")
            return _generate_synthetic_skin_lesion(skin_dir)
    
    # If we have metadata but not images, use synthetic data
    if not os.path.exists(os.path.join(skin_dir, 'HAM10000_images')):
        print("No image data found. Using synthetic skin lesion data for benchmarking.")
        return _generate_synthetic_skin_lesion(skin_dir)
    
    # Load dataset (this would be implemented for real data)
    # For now, return synthetic data
    return _generate_synthetic_skin_lesion(skin_dir)


def _generate_synthetic_skin_lesion(skin_dir):
    """Generate synthetic skin lesion data for benchmarking.
    
    Args:
        skin_dir: Directory to store data
        
    Returns:
        Tuple of (X, y, metadata)
    """
    print("Generating synthetic skin lesion data...")
    
    # Parameters
    n_samples = 1000
    image_size = 28  # Small images for faster processing
    
    # Generate data
    X = []
    y = []
    
    for i in range(n_samples):
        # Generate synthetic skin lesion image
        image = np.zeros((image_size, image_size))
        
        # Create circular lesion
        center_x, center_y = image_size // 2, image_size // 2
        radius = np.random.randint(5, 10)
        
        # Add random offset to center
        center_x += np.random.randint(-3, 4)
        center_y += np.random.randint(-3, 4)
        
        # Create base lesion
        for x in range(image_size):
            for y in range(image_size):
                dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if dist < radius:
                    # Base intensity (darker in center)
                    intensity = 0.7 - 0.5 * (1 - dist / radius)
                    image[x, y] = intensity
        
        # Determine if malignant (1) or benign (0)
        is_malignant = (i % 3 == 0)  # 1/3 of samples are malignant
        
        if is_malignant:
            # Add irregular border for malignant lesions
            for angle in range(0, 360, 30):
                angle_rad = np.radians(angle)
                # Random protrusion
                protrusion = np.random.randint(2, 5)
                x_offset = int(np.cos(angle_rad) * (radius + protrusion))
                y_offset = int(np.sin(angle_rad) * (radius + protrusion))
                
                # Add protrusion
                x_pos = center_x + x_offset
                y_pos = center_y + y_offset
                
                if 0 <= x_pos < image_size and 0 <= y_pos < image_size:
                    # Draw small circle at protrusion point
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if dx**2 + dy**2 <= 4:  # Small circle
                                nx, ny = x_pos + dx, y_pos + dy
                                if 0 <= nx < image_size and 0 <= ny < image_size:
                                    image[nx, ny] = 0.6
            
            # Add asymmetry
            for x in range(image_size):
                for y in range(image_size):
                    if image[x, y] > 0 and x > center_x:
                        image[x, y] *= 1.3  # Darker on one side
            
            y.append(1)  # Malignant
        else:
            # Benign lesions have more regular borders
            y.append(0)  # Benign
        
        # Add noise
        image += 0.05 * np.random.randn(image_size, image_size)
        
        # Clip values to [0, 1]
        image = np.clip(image, 0, 1)
        
        X.append(image)
    
    # Create metadata
    metadata = {
        'name': 'Synthetic Skin Lesion',
        'description': 'Synthetic skin lesion data for benchmarking',
        'n_samples': n_samples,
        'image_size': image_size,
        'classes': ['Benign', 'Malignant'],
        'synthetic': True
    }
    
    return np.array(X), np.array(y), metadata
