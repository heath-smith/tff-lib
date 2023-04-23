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

    Properties
    ----------
        thickness: float, Read-Write, layer thickness in nanometers (>= 0)

    Attributes
    ----------
        wavelengths: Iterable[float], 1-D array of wavelengths in nanometers
        ref_index: Iterable[complex], 1-D array of complex refractive indices
    """

    def __init__(
            self,
            wavelengths: Iterable[float],
            ref_index: Iterable[complex],
            thickness: float = 'inf'
    ) -> None:
        """
        Initializes the OpticalMedium class.

        Parameters
        ----------
        wavelengths: Iterable[float], 1-D array of wavelengths in nanometers
        ref_index: Iterable[complex], 1-D array of complex refractive indices
        thickness: float, medium thickness in nanometers

        Raises
        ----------
        ValueError if wavelengths and ref_index shapes do not match or thickness
            less than 0.
        """

        if not len(wavelengths) == len(ref_index):
            raise ValueError("length of wavelengths and ref_index must be equal")

        self.wavelengths = np.array([float(x) for x in wavelengths])
        self.ref_index = np.array([complex(y) for y in ref_index])

        # uses setter to initialize
        self.thickness = float(thickness)

    @property
    def thickness(self) -> float:
        """
        float, thickness in nanometers, must be greater than zero
        """
        return self._thickness

    @thickness.setter
    def thickness(self, new_thickness:float):
        if new_thickness <= 0:
            raise ValueError("thickness must be greater than 0")
        self._thickness = float(new_thickness)

    def admittance(
            self,
            inc_medium: 'OpticalMedium',
            theta: float
    ) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of the incident medium.

        Parameters
        -------------
        theta: float, angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the medium-incident interface,
            'p': p-polarized admittance of the medium-incident interface }
        """

        # calculate complex dialectric constants (square the values)
        # for both the current medium and the incident medium
        dialectrics = self.ref_index**2
        inc_dialectrics = inc_medium.ref_index**2

        # Calculate S and P admittances
        admit_s = np.sqrt(dialectrics - inc_dialectrics * np.sin(theta)**2)
        admit_p = dialectrics / admit_s

        return {'s': admit_s, 'p': admit_p}
