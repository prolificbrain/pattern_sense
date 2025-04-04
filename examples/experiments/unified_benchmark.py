"""Comprehensive benchmark for UNIFIED visual pattern recognition"""
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from unified.mtu.learning import PatternMemory
from unified.trits.tryte import Tryte
from unified.trits.trit import Trit, TritState


def mnist_to_tryte(image):
    """Convert MNIST image (28x28 array) to Tryte format"""
    trits = []
    for y in range(28):
        for x in range(28):
            if image[y,x] > 0:  # Only store non-zero pixels
                trits.append(Trit(
                    state=TritState.POSITIVE if image[y,x] > 128 else TritState.NEUTRAL,
                    energy=float(image[y,x])/255.0,
                    orientation={'x': float(x), 'y': float(y)}
                ))
    return Tryte(trits=trits, height=28, width=28)


def run_mnist_benchmark():
    """Compare performance on MNIST dataset"""
    print("Starting MNIST benchmark...")
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('../data', train=False, transform=transform)
    
    print("Data loaded, training UNIFIED system...")
    # UNIFIED System Test - limited to 1000 samples for speed
    unified_mem = PatternMemory(max_patterns=10000)
    start_time = time.time()
    
    # Train on subset
    sample_count = 1000  # Limit for faster testing
    for i, (img, _) in enumerate(DataLoader(train_set, batch_size=1)):
        if i >= sample_count:
            break
        img = img.squeeze().numpy()
        tryte = mnist_to_tryte(img)
        unified_mem.learn_pattern(tryte)
        if i % 100 == 0:
            print(f"Processed {i} samples")
    
    unified_train_time = time.time() - start_time
    print(f"UNIFIED training complete in {unified_train_time:.2f} seconds")
    
    # Traditional ML Comparison
    print("Training SVM classifier...")
    X_train = train_set.data[:sample_count].numpy().reshape(-1, 784)
    y_train = train_set.targets[:sample_count].numpy()
    
    svm = SVC(kernel='rbf')
    svm_start = time.time()
    svm.fit(X_train, y_train)
    svm_time = time.time() - svm_start
    print(f"SVM training complete in {svm_time:.2f} seconds")
    
    # Visualization
    print("Generating visualizations...")
    fig, ax = plt.subplots(2, 5, figsize=(15,6))
    for i in range(5):
        pattern = unified_mem.get_attractor_pattern(i)
        if pattern is not None:
            ax[0,i].imshow(pattern, cmap='gray')
        ax[1,i].imshow(train_set[i][0].squeeze(), cmap='gray')
        ax[0,i].set_title(f"UNIFIED Pattern {i}")
        ax[1,i].set_title(f"Original Image {i}")
    
    plt.tight_layout()
    plt.suptitle('UNIFIED vs Original Patterns', y=1.02)
    plt.savefig('pattern_comparison.png')
    print("Visualization saved to pattern_comparison.png")
    
    return {
        'unified_train_time': unified_train_time,
        'svm_train_time': svm_time,
        'unified_pattern_count': len(unified_mem._patterns),
        'svm_support_vectors': len(svm.support_vectors_),
        'sample_count': sample_count
    }


if __name__ == "__main__":
    print("UNIFIED Consciousness Engine - Pattern Recognition Benchmark")
    results = run_mnist_benchmark()
    print(f"\nBenchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
