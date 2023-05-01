"""
This module contains a wrapper class for the OpticalMedium
C-Extension class.

Wiki:
In optics, an optical medium is material through which light and other
electromagnetic waves propagate. It is a form of transmission medium.
The permittivity and permeability of the medium define how electromagnetic
waves propagate in it.
"""

from typing import Iterable, Dict
from numpy.typing import NDArray
import medium

class OpticalMedium(medium.OpticalMedium):
    """
    Python wrapper for OpticalMedium C-extension class.

    Properties
    ----------
        thick: float, thickness in nanometers. -1 if unspecified, or must
            be greater than zero.
        ntype: int, one of -1, 0, or 1. Use 1 for a high index material,
            0 for a low index material, or -1 if not specified.

    Attributes
    ----------
        waves: Iterable[float], 1-D array of wavelengths in nanometers
        nref: Iterable[complex], 1-D array of complex refractive indices


    Raises
    ----------
        ValueError
            if ntype not in (1, 0, -1), thick not -1 or > 0, len(waves) != len(nref),
             waves or nref not 1-D.
    """

    def __init__(
            self,
            waves: Iterable[float],
            nref: Iterable[complex],
            **kwargs
    ) -> None:
        """
        Initializes the OpticalMedium class.

        args
        ----------
        wavelengths: Iterable[float], 1-D array of wavelengths in nanometers
        nref: Iterable[complex], 1-D array of complex refractive indices

        kwargs
        ----------
        thick: float, medium thickness in nanometers. -1 if unspecified, or must
            be greater than zero. (default -1)
        ntype: int, one of -1, 0, or 1. Use 1 for a high index material,
            0 for a low index material, or -1 if not specified. (default -1)

        Raises
        ----------
        ValueError
            if ntype not in (1, 0, -1), thick not -1 or > 0, len(waves) != len(nref),
             waves or nref not 1-D.
        """
        super().__init__(waves, nref, **kwargs)


    @property
    def thick(self) -> float:
        """
        float, thickness in nanometers, can be -1 if unspecified or
        must be greater than zero
        """
        return self._thick

    @thick.setter
    def thick(self, new_thick:float):
        self._thick = float(new_thick)

    @property
    def ntype(self) -> int:
        """
        int, index type, one of 1, 0, or -1. 1 denotes a high index
        material and 0 is for a low index. -1 if unspecified.
        """
        return self._ntype

    @ntype.setter
    def ntype(self, new_ntype:float):
        self._ntype = float(new_ntype)


    def absorption_coefficients(self, n_reflect: int = 4) -> NDArray:
        """
        Calculate the absorption coefficients for n_ref reflections.

        Parameters
        ----------
        n_reflect: int, the number of reflections (default 4)

        Returns
        ----------
        NDArray, absorption coefficients as a function of wavelength
        """
        return super().absorption_coeffs(n_reflect)

    def nref_effective(self, theta: float) -> NDArray:
        """
        Calculates the effective refractive index through the
        medium. Requires a non-zero thickness.

        Parameters
        -----------
        theta: float, angle of incidence of radiation in radians

        Returns
        ----------
        NDArray, Effective substrate refractive indices as a function
            of wavelength
        """
        return super().nref_effective(theta)

    def admittance(self, incident: 'OpticalMedium', theta: float) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of the incident->medium interface.

        Parameters
        -------------
        incident: OpticalMedium, incident medium of the radiation
        theta: float, angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the incident->medium interface,
            'p': p-polarized admittance of the incident->medium interface }
        """
        return super().admittance(incident, theta)

    def path_length(self, incident: 'OpticalMedium', theta: float) -> NDArray:
        """
        Calculates the estimated optical path length through the medium
        given incident medium and incident angle.

        Parameters
        -------------
        inc_medium: OpticalMedium, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        ------------
        NDArray, path length through the medium as a function of wavelength

        Raises
        ----------
        ValueError, if medium thickness < 0
        """

        # calculate the denominator
        den = np.sqrt(1 - (np.abs(incident.nref)**2 * (np.sin(theta)**2) / self.effective_index(theta)**2))

        # return the path length
        return self.thickness / den
