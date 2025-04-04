"""PatternSense - Advanced Pattern Recognition Framework.

PatternSense is a comprehensive framework for pattern recognition,
hierarchical abstraction, and cognitive processing using trinary logic and field dynamics.
"""

__version__ = "0.2.0"

# Import main components for easier access
from patternsense.mtu.learning import PatternMemory
from patternsense.mtu.accelerated import AcceleratedPatternMemory
from patternsense.mtu.hierarchical import HierarchicalPatternNetwork
from patternsense.mtu.temporal import TemporalPatternMemory, TimeSeriesPredictor
from patternsense.mtu.clustering import PatternClusteringEngine
from patternsense.mtu.anomaly import AnomalyScorer

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
