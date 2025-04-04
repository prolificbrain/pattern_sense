# PatternSense - Pattern Recognition Benchmarks

## Overview

This directory contains benchmarking tools and results for PatternSense's pattern recognition capabilities. The benchmarks evaluate the performance of various pattern recognition algorithms against traditional machine learning methods across multiple datasets.

## Features Benchmarked

1. **GPU Acceleration** - Leveraging Apple Metal (MPS) for accelerated pattern operations
2. **Unsupervised Clustering** - Automatic pattern discovery without labeled data
3. **Hierarchical Pattern Recognition** - Multi-level pattern abstraction
4. **Advanced Anomaly Detection** - Multi-method approach to identifying outliers
5. **Parallel Processing** - Multi-threaded batch operations for improved performance

## Benchmark Results

The benchmark results demonstrate that PatternSense performs competitively with traditional machine learning methods, and in some cases outperforms them, particularly on the diabetes dataset where `PatternMemory` achieved the highest accuracy (79.70%) compared to SVM (75.19%) and RandomForest (75.94%).

### Breast Cancer Dataset

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| SVM | 97.66% | 98.41% | 97.87% | 98.13% | 99.75% |
| RandomForest | 94.74% | 95.35% | 96.35% | 95.85% | 99.20% |
| PatternMemory | 95.32% | 96.15% | 96.45% | 96.30% | 99.30% |
| HierarchicalPatternNetwork | 94.74% | 95.31% | 96.30% | 95.81% | 98.53% |

### Diabetes Dataset

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| SVM | 75.19% | 75.00% | 76.12% | 75.56% | 84.40% |
| RandomForest | 75.94% | 75.38% | 78.31% | 76.81% | 84.34% |
| PatternMemory | 79.70% | 79.70% | 79.70% | 79.70% | 85.07% |
| HierarchicalPatternNetwork | 74.44% | 73.85% | 73.85% | 73.85% | 82.36% |

## Key Findings

1. **Pattern Memory Performance**: The basic `PatternMemory` implementation shows competitive performance with traditional ML methods, particularly on the diabetes dataset where it outperforms both SVM and RandomForest.

2. **Hierarchical Pattern Recognition**: The `HierarchicalPatternNetwork` demonstrates good performance, especially considering it builds abstractions across multiple levels.

3. **Anomaly Detection**: The multi-method approach to anomaly detection achieved high accuracy (91.81%) and AUC (97.55%) on the breast cancer dataset.

4. **Unsupervised Clustering**: The clustering engine successfully identified natural clusters in the data that strongly correlate with the class labels.

## Running the Benchmarks

To run the comprehensive benchmark suite:

```bash
python comprehensive_benchmark_report.py
```

For a focused benchmark on a single dataset:

```bash
python run_focused_benchmark.py
```

## Advanced Pattern Recognition Demo

To see all the enhanced features in action, run the advanced pattern recognition demo:

```bash
python ../advanced_pattern_recognition_demo.py
```

This demo showcases:
- GPU-accelerated pattern operations
- Unsupervised clustering for pattern discovery
- Advanced anomaly detection with multiple scoring methods
- Hierarchical pattern recognition

## Datasets

The benchmarks use the following datasets:

1. **Breast Cancer Wisconsin** - Diagnostic breast cancer dataset with 569 samples
2. **Diabetes** - Diabetes progression dataset (binarized) with 442 samples

## Future Work

1. **Expanded Dataset Testing** - Test on more complex datasets like ECG and skin lesion images
2. **Hyperparameter Optimization** - Fine-tune parameters for each algorithm
3. **Ensemble Methods** - Combine multiple pattern recognition approaches
4. **Real-time Processing** - Optimize for real-time pattern recognition in streaming data
