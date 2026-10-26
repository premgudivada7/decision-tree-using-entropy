"""gaintree — grow a classification tree by maximising information gain (NumPy only)."""

from .info import entropy_of_counts, gain_of_split, shannon_entropy, surprise
from .classifier import GainTreeClassifier, TreeNode
from .render import draw, to_dot, to_text
from .play_tennis import load_play_tennis

__version__ = "0.1.0"

__all__ = [
    "GainTreeClassifier", "TreeNode",
    "surprise", "shannon_entropy", "entropy_of_counts", "gain_of_split",
    "to_text", "to_dot", "draw", "load_play_tennis",
]
