"""
structure Subpackage

Collection of data structures to store the structure of particle phases.
"""

# Collect submodules into unified module namespace.

from .core import POSITION_TOLERANCE
from .crystals import CrystalBase, CrystalMotif, buildCrystal
from .lattice import Lattice
from .particles import *
from .symmetry import SpaceGroup

