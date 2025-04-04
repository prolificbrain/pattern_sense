"""Minimal Thinking Unit (MTU) module for the UNIFIED Consciousness Engine.

This module contains the core pattern recognition and processing components
of the UNIFIED Consciousness Engine, including basic pattern memory, hierarchical
pattern networks, temporal pattern memory, and advanced analytics.
"""

# Legacy components
from .mtu import MinimalThinkingUnit
from .mtu_network import MTUNetwork

# Core pattern memory components
from .learning import PatternMemory
from .accelerated import AcceleratedPatternMemory
from .hierarchical import HierarchicalPatternNetwork
from .temporal import TemporalPatternMemory, TimeSeriesPredictor

# Advanced analytics
from .clustering import PatternClusteringEngine
from .anomaly import AnomalyScorer

__all__ = [
    # Legacy components
    "MinimalThinkingUnit",
    "MTUNetwork",
    
    # Core pattern memory components
    "PatternMemory",
    "AcceleratedPatternMemory",
    "HierarchicalPatternNetwork",
    "TemporalPatternMemory",
    "TimeSeriesPredictor",
    
    # Advanced analytics
    "PatternClusteringEngine",
    "AnomalyScorer",
]
