"""
Hierarchical Air Hockey Multi-Agent Reinforcement Learning
A sophisticated 2v2 air hockey environment with hierarchical agent control.
"""

__version__ = "0.1.0"
__author__ = "Hierarchical Air Hockey Team"

from .environments import HierarchicalAirHockeyEnv
from .agents import HierarchicalAgent, HighLevelAgent, LowLevelAgent
from .training import HierarchicalTrainer

__all__ = [
    "HierarchicalAirHockeyEnv",
    "HierarchicalAgent", 
    "HighLevelAgent",
    "LowLevelAgent",
    "HierarchicalTrainer"
]
