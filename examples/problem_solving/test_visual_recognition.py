"""Test script for visual pattern recognition with MNIST data"""
import numpy as np
from torchvision import datasets
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
                    orientation={'x': float(x), 'y': float(y)}  # Use dict for named access
                ))
    return Tryte(trits=trits, height=28, width=28)

def test_pattern_memory():
    """Test pattern memory with MNIST samples"""
    # Load MNIST data
    mnist = datasets.MNIST('../data', train=True, download=True)
    images = mnist.data.numpy()
    
    # Initialize pattern memory
    pattern_mem = PatternMemory(max_patterns=1000)
    
    # Test learning 100 samples
    for i in range(100):
        tryte = mnist_to_tryte(images[i])
        idx = pattern_mem.learn_pattern(tryte)
        print(f"Learned pattern {i} at index {idx}")
    
    # Verify recall
    test_idx = np.random.randint(0, 100)
    recalled = pattern_mem.get_attractor_pattern(test_idx)
    print(f"Recalled pattern {test_idx} with shape {recalled.shape}")

if __name__ == "__main__":
    test_pattern_memory()
