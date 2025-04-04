# PatternSense - Quick Start Guide

## Installation

### Prerequisites

- Python 3.12 or higher
- uv (Python package manager)

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/prolificbrain/PatternSense.git
cd PatternSense

# Create and activate a virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Basic Usage

### Pattern Recognition

```python
from patternsense import PatternMemory
import numpy as np

# Create a pattern memory
pattern_memory = PatternMemory(max_patterns=1000)

# Learn some patterns
pattern1 = np.random.randn(10, 10)
pattern2 = np.random.randn(10, 10)

idx1 = pattern_memory.learn_pattern(pattern1)
idx2 = pattern_memory.learn_pattern(pattern2)

# Recognize a pattern
test_pattern = pattern1 + 0.1 * np.random.randn(10, 10)  # Noisy version
matches = pattern_memory.recognize_pattern(test_pattern, top_k=3)

for idx, similarity in matches:
    print(f"Pattern {idx} matched with similarity {similarity:.4f}")
```

### GPU Acceleration

```python
from patternsense import AcceleratedPatternMemory
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create accelerated pattern memory
accel_memory = AcceleratedPatternMemory(
    max_patterns=10000,
    use_gpu=(device.type != 'cpu'),
    batch_size=64
)

# Process patterns in batches
batch_patterns = [torch.randn(10, 10) for _ in range(100)]
indices = accel_memory.learn_patterns_batch(batch_patterns)
```

### Hierarchical Pattern Recognition

```python
from patternsense import HierarchicalPatternNetwork

# Create hierarchical network
hierarchical_network = HierarchicalPatternNetwork(
    input_dimensions=(10, 10),
    max_levels=3,
    patterns_per_level=1000
)

# Learn patterns hierarchically
patterns = [np.random.randn(10, 10) for _ in range(10)]
for pattern in patterns:
    level_indices = hierarchical_network.learn_pattern(pattern)
    print(f"Pattern stored at indices: {level_indices}")
```

### Temporal Pattern Recognition

```python
from patternsense import TemporalPatternMemory

# Create temporal pattern memory
temporal_memory = TemporalPatternMemory(
    max_patterns=1000,
    window_size=5
)

# Observe a sequence of patterns
sequence = [np.random.randn(10, 10) for _ in range(20)]
for pattern in sequence:
    idx, matches = temporal_memory.observe_pattern(pattern)
    print(f"Pattern stored at index {idx}")
    if matches:
        print(f"Matched with: {matches}")
```

### Anomaly Detection

```python
from patternsense import AnomalyScorer, PatternMemory, PatternClusteringEngine

# Create pattern memory and clustering engine
pattern_memory = PatternMemory(max_patterns=1000)
clustering_engine = PatternClusteringEngine(
    pattern_memory=pattern_memory,
    clustering_method='kmeans',
    n_clusters=3
)

# Create anomaly detector
anomaly_detector = AnomalyScorer(
    methods=['reconstruction', 'clustering', 'statistical'],
    pattern_memory=pattern_memory,
    clustering_engine=clustering_engine
)

# Generate normal and abnormal patterns
normal_patterns = [np.random.randn(10, 10) for _ in range(100)]
abnormal_patterns = [np.random.randn(10, 10) * 2 for _ in range(20)]

# Train anomaly detector
anomaly_detector.train(normal_patterns, abnormal_patterns)

# Detect anomalies
test_patterns = [np.random.randn(10, 10) for _ in range(10)] + [np.random.randn(10, 10) * 2 for _ in range(5)]
for pattern in test_patterns:
    is_anomaly, score, method_scores = anomaly_detector.is_anomaly(pattern)
    if is_anomaly:
        print(f"Anomaly detected with score {score:.4f}")
        print(f"Method scores: {method_scores}")
```

## Running Examples

### ECG Anomaly Detection

```bash
python examples/applications/ecg_anomaly_detection.py
```

### Advanced Pattern Recognition Demo

```bash
python examples/advanced_pattern_recognition_demo.py
```

### Benchmarking

```bash
python examples/benchmarks/comprehensive_benchmark_report.py
```

## Next Steps

- Check out the [full documentation](https://github.com/prolificbrain/PatternSense/docs)
- Explore the [examples directory](https://github.com/prolificbrain/PatternSense/examples)
- Read the [benchmarking results](https://github.com/prolificbrain/PatternSense/examples/benchmarks)
