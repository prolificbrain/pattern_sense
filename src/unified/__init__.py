"""UNIFIED Consciousness Engine - Advanced Pattern Recognition Framework.

The UNIFIED Consciousness Engine is a comprehensive framework for pattern recognition,
hierarchical abstraction, and cognitive processing using trinary logic and field dynamics.
"""

__version__ = "0.2.0"

# Import main components for easier access
from unified.mtu.learning import PatternMemory
from unified.mtu.accelerated import AcceleratedPatternMemory
from unified.mtu.hierarchical import HierarchicalPatternNetwork
from unified.mtu.temporal import TemporalPatternMemory, TimeSeriesPredictor
from unified.mtu.clustering import PatternClusteringEngine
from unified.mtu.anomaly import AnomalyScorer

# Define package structure
__all__ = [
    # Core pattern memory components
    'PatternMemory',
    'AcceleratedPatternMemory',
    'HierarchicalPatternNetwork',
    'TemporalPatternMemory',
    'TimeSeriesPredictor',
    
    # Advanced analytics
    'PatternClusteringEngine',
    'AnomalyScorer',
]
