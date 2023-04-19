"""
This module contains the Medium class
representing an optical medium.

Wiki:
In optics, an optical medium is material through which light and other
electromagnetic waves propagate. It is a form of transmission medium.
The permittivity and permeability of the medium define how electromagnetic
waves propagate in it.
"""

from typing import Iterable, Dict
from numpy.typing import NDArray
import numpy as np

class OpticalMedium():
    """
    Abstract representation of an optical medium.

    Attributes
    ----------
        wavelengths: Iterable[float], 1-D array of wavelengths in nanometers
        ref_index: Iterable[complex],1-D array of complex refractive indices
        name: str, the name of the medium (e.g. air, water, glass)
    """

    def __init__(
            self,
            wavelengths: Iterable[float],
            ref_index: Iterable[complex],
            name: str
    ) -> None:
        """
        Initializes the OpticalMedium class.

        Parameters
        ----------
        wavelengths: Iterable[float], 1-D array of wavelengths in nanometers
        ref_index: Iterable[complex], 1-D array of complex refractive indices
        name: str, the name of the medium (e.g. air, water, glass)

        Raises
        ----------
        ValueError if wavelengths and ref_index shapes do not match.
        """

        if not len(wavelengths) == len(ref_index):
            raise ValueError("length of wavelengths and ref_index must be equal")

        self.wavelengths = [float(x) for x in wavelengths]
        self.ref_index = [complex(y) for y in ref_index]
        self.name = name

    def admittance(self, theta:float) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of the incident medium.

        Parameters
        -------------
        theta: float, angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the incident medium,
            'p': p-polarized admittance of the incident medium }
        """

        # calculate complex dialectric constants (square the values)
        dialectrics = [m**2 for m in self.ref_index]

        # Calculate S and P admittances of the incident media
        admit_s_inc = np.sqrt(dialectrics - dialectrics * np.sin(theta)**2)
        admit_p_inc = dialectrics / admit_s_inc

        return {'s': admit_s_inc, 'p': admit_p_inc}
