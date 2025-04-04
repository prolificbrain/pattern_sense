"""Medical imaging pattern recognition using UNIFIED Consciousness Engine.

This example demonstrates the application of the UNIFIED pattern recognition system
to medical imaging data, specifically chest X-rays for anomaly detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.feature import hog
from unified.mtu.learning import PatternMemory
from unified.trits.tryte import Tryte
from unified.trits.trit import Trit, TritState

# Constants
IMAGE_SIZE = (64, 64)  # Resize images for faster processing
DOWNLOAD_URL = "https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345"


def download_sample_data(target_dir="./data/chest_xray", sample_count=10):
    """Download sample chest X-ray data.
    
    For a real application, download the NIH Chest X-ray dataset from:
    https://nihcc.app.box.com/v/ChestXray-NIHCC
    
    This function simulates the download with placeholder data.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Create placeholder data (in real app, download actual images)
    print(f"In a real application, download {sample_count} images from {DOWNLOAD_URL}")
    print(f"For this demonstration, creating synthetic X-ray data in {target_dir}")
    
    # Generate synthetic data for demonstration
    for i in range(sample_count):
        # Create synthetic X-ray-like image (64x64 grayscale)
        img = np.zeros(IMAGE_SIZE)
        
        # Add basic lung-like structures
        center_x, center_y = IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2
        for x in range(IMAGE_SIZE[0]):
            for y in range(IMAGE_SIZE[1]):
                # Create oval lung fields
                dist_left = np.sqrt((x - center_x + 15)**2 + (y - center_y)**2)
                dist_right = np.sqrt((x - center_x - 15)**2 + (y - center_y)**2)
                
                if dist_left < 20 or dist_right < 20:
                    img[x, y] = 0.8  # Lung fields are darker
                else:
                    img[x, y] = 0.2  # Background is lighter
        
        # Add random noise
        img += np.random.normal(0, 0.05, IMAGE_SIZE)
        
        # Add anomaly to some images (simulating pathology)
        if i % 3 == 0:  # Every third image has an anomaly
            # Add a small bright spot (nodule)
            spot_x = np.random.randint(center_x - 15, center_x + 15)
            spot_y = np.random.randint(center_y - 15, center_y + 15)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if 0 <= spot_x + dx < IMAGE_SIZE[0] and 0 <= spot_y + dy < IMAGE_SIZE[1]:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < 3:
                            img[spot_x + dx, spot_y + dy] = 1.0  # Bright spot
            
            label = "abnormal"
        else:
            label = "normal"
        
        # Clip values to [0, 1]
        img = np.clip(img, 0, 1)
        
        # Save image and label
        plt.imsave(os.path.join(target_dir, f"xray_{i:03d}_{label}.png"), 
                  img, cmap='gray')
    
    return target_dir


def load_xray_data(data_dir):
    """Load X-ray images and labels from directory."""
    images = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            # Extract label from filename
            label = "abnormal" if "abnormal" in filename else "normal"
            
            # Load and preprocess image
            img_path = os.path.join(data_dir, filename)
            img = np.array(Image.open(img_path).convert('L')) / 255.0
            
            images.append(img)
            labels.append(1 if label == "abnormal" else 0)
    
    return np.array(images), np.array(labels)


def xray_to_tryte(image):
    """Convert X-ray image to Tryte format."""
    height, width = image.shape
    trits = []
    
    # Create trits for significant pixels
    for y in range(height):
        for x in range(width):
            # Only create trits for significant pixels
            if image[y, x] > 0.3:
                # Determine state based on intensity
                if image[y, x] > 0.7:
                    state = TritState.POSITIVE  # Bright spots (potential anomalies)
                elif image[y, x] < 0.3:
                    state = TritState.NEGATIVE  # Dark areas (lung fields)
                else:
                    state = TritState.NEUTRAL  # Mid-range intensities
                
                # Create trit with position and energy proportional to intensity
                trits.append(Trit(
                    state=state,
                    energy=float(image[y, x]),
                    orientation={'x': float(x), 'y': float(y)}
                ))
    
    return Tryte(trits=trits, height=height, width=width)


def train_xray_recognition(images, labels):
    """Train the UNIFIED system on X-ray images."""
    # Initialize pattern memory
    pattern_mem = PatternMemory(max_patterns=1000, sparse_threshold=0.2)
    
    # Track indices for normal and abnormal patterns
    normal_indices = []
    abnormal_indices = []
    
    # Learn patterns
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert to tryte
        tryte = xray_to_tryte(img)
        
        # Learn pattern
        idx = pattern_mem.learn_pattern(tryte)
        
        # Track indices by label
        if label == 1:  # Abnormal
            abnormal_indices.append(idx)
        else:  # Normal
            normal_indices.append(idx)
    
    return pattern_mem, normal_indices, abnormal_indices


def detect_anomalies(pattern_mem, test_images, normal_indices, abnormal_indices):
    """Detect anomalies in test images using pattern similarity."""
    results = []
    
    for img in test_images:
        # Convert to tryte
        tryte = xray_to_tryte(img)
        
        # Get top matches
        matches = pattern_mem.recognize_pattern(tryte, top_k=3)
        
        # Calculate similarity to normal and abnormal patterns
        normal_sim = 0
        abnormal_sim = 0
        match_count = 0
        
        for idx, sim in matches:
            if idx in normal_indices:
                normal_sim += sim
                match_count += 1
            elif idx in abnormal_indices:
                abnormal_sim += sim
                match_count += 1
        
        # Normalize similarities
        if match_count > 0:
            normal_sim /= match_count
            abnormal_sim /= match_count
        
        # Classify based on similarity
        is_abnormal = abnormal_sim > normal_sim
        confidence = abs(abnormal_sim - normal_sim)
        
        results.append({
            'prediction': 1 if is_abnormal else 0,
            'confidence': confidence,
            'normal_sim': normal_sim,
            'abnormal_sim': abnormal_sim
        })
    
    return results


def evaluate_performance(results, true_labels):
    """Evaluate anomaly detection performance."""
    predictions = [r['prediction'] for r in results]
    
    # Calculate metrics
    true_pos = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
    false_pos = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
    true_neg = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
    false_neg = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
    
    # Calculate performance metrics
    accuracy = (true_pos + true_neg) / len(true_labels) if len(true_labels) > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'true_positive': true_pos,
            'false_positive': false_pos,
            'true_negative': true_neg,
            'false_negative': false_neg
        }
    }


def visualize_results(images, results, true_labels, num_samples=4):
    """Visualize anomaly detection results."""
    # Select random samples to visualize
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(images[idx], cmap='gray')
        true_label = "Abnormal" if true_labels[idx] == 1 else "Normal"
        axes[0, i].set_title(f"True: {true_label}")
        axes[0, i].axis('off')
        
        # Prediction visualization
        pred = results[idx]['prediction']
        conf = results[idx]['confidence']
        pred_label = "Abnormal" if pred == 1 else "Normal"
        
        # Create heatmap based on confidence
        heatmap = np.zeros_like(images[idx])
        if pred == 1:  # Abnormal prediction
            # Highlight potential anomalies
            heatmap = np.where(images[idx] > 0.7, conf, 0)
        
        axes[1, i].imshow(images[idx], cmap='gray')
        axes[1, i].imshow(heatmap, cmap='hot', alpha=0.5)
        axes[1, i].set_title(f"Pred: {pred_label} (Conf: {conf:.2f})")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('xray_results.png')
    plt.close()


def run_medical_imaging_demo():
    """Run the complete medical imaging demo."""
    print("UNIFIED Consciousness Engine - Medical Imaging Pattern Recognition")
    print("=================================================================\n")
    
    # Step 1: Prepare data
    print("Step 1: Preparing X-ray data...")
    data_dir = download_sample_data(sample_count=30)
    images, labels = load_xray_data(data_dir)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42)
    
    print(f"Loaded {len(images)} images ({sum(labels)} abnormal, {len(labels)-sum(labels)} normal)")
    print(f"Training set: {len(X_train)} images, Test set: {len(X_test)} images\n")
    
    # Step 2: Train pattern recognition
    print("Step 2: Training UNIFIED pattern recognition system...")
    pattern_mem, normal_indices, abnormal_indices = train_xray_recognition(X_train, y_train)
    print(f"Learned {len(pattern_mem._patterns)} patterns")
    print(f"Normal patterns: {len(normal_indices)}, Abnormal patterns: {len(abnormal_indices)}\n")
    
    # Step 3: Detect anomalies
    print("Step 3: Detecting anomalies in test images...")
    results = detect_anomalies(pattern_mem, X_test, normal_indices, abnormal_indices)
    
    # Step 4: Evaluate performance
    print("Step 4: Evaluating performance...")
    metrics = evaluate_performance(results, y_test)
    
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1_score']:.2f}\n")
    
    # Step 5: Visualize results
    print("Step 5: Visualizing results...")
    visualize_results(X_test, results, y_test)
    print("Results visualization saved to 'xray_results.png'\n")
    
    return metrics


if __name__ == "__main__":
    run_medical_imaging_demo()
