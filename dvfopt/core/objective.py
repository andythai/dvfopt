"""Objective functions for SLSQP optimisation."""

import numpy as np


def objectiveEuc(phi, phi_init):
    """L2 norm objective function."""
    return np.linalg.norm(phi - phi_init)
