"""
This module contains a collection of functions for calculating characteristics
of optical filters and thin film stacks.
"""

# import external packages
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

def admittance(inc_medium:ArrayLike, theta:float) -> Tuple:
    """
    Calculates optical admittance of an incident medium.

    Parameters
    -------------
    inc_med: Iterable[complex], complex refractive indices of incident medium
    theta: float, angle of incidence of radiation in radians

    Returns
    --------------
    (Tuple)
    (s-polarized admittance of the incident medium, p-polarized admittance of the incident medium,
    """

    # calculate complex dialectric constants (square the values)
    dialectrics = [m**2 for m in inc_medium]

    # Calculate S and P admittances of the incident media
    admit_s_inc = np.sqrt(dialectrics - dialectrics * np.sin(theta)**2)
    admit_p_inc = dialectrics / admit_s_inc

    return admit_s_inc, admit_p_inc



