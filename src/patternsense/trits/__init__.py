"""Trit-based logic module for the UNIFIED Consciousness Engine."""

from .trit import Trit, TritState
from .tryte import Tryte
from .trit_patterns import create_pulse_pattern, create_wave_pattern

__all__ = [
    "Trit",
    "TritState",
    "Tryte",
    "create_pulse_pattern",
    "create_wave_pattern",
]
