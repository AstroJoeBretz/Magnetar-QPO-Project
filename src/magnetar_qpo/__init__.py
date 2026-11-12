"""Magnetar QPO model: tangled-field interior coupled to a magnetosphere.

Eigenfrequencies of the star+magnetosphere shear-wave problem and the residue-theorem
solution of the initial value problem (Bretz, van Eysden & Link, MNRAS).
"""

from .params import ModelParams, PhysicalScales

__all__ = ["ModelParams", "PhysicalScales"]
__version__ = "0.1.0"
