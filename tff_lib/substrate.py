"""
This module contains the Substrate class.
"""

from typing import Dict, Iterable
from numpy.typing import NDArray
import numpy as np
from .medium import OpticalMedium

class Substrate(OpticalMedium):
    """
    Child class of OpticalMedium, represents an optical substrate
    which can be used in the construction of a thin-film optical
    filter. Inherits all public attributes, properties, and methods
    of OpticalMedium.

    See Also
    ----------
    >>> tff_lib.OpticalMedium(
            wavelengths: Iterable[float],
            ref_index: Iterable[complex],
            thickness: float
        ) -> None
    """

    def __init__(
            self,
            wavelengths: Iterable[float],
            ref_index: Iterable[complex],
            thickness: float
    ) -> None:
        """
        Initialize the class attributes. Raises ValueError if wavelengths
        and ref_index not equal length or thickness less than zero.

        Parameters
        ----------
        wavelengths: Iterable[float], 1-D wavelength values
        ref_index: Iterable[complex] 1-D complex refractive indices
        thickness: float, substrate thickness in nanometers
        """

        super().__init__(wavelengths, ref_index, thickness)

    def absorption_coefficients(self, n_ref: int = 4) -> NDArray:
        """
        Calculate the absorption coefficient for n_ref reflections.

        Parameters
        ----------
        n_ref: int, the number of reflections (default 4)

        Returns
        ----------
        NDArray
        """
        return (n_ref * np.pi * np.imag(self.ref_index)) / self.wavelengths

    def effective_index(self, theta: float) -> NDArray:
        """
        Calculates the effective substrate refractive index for abs. substrates.

        Parameters
        -----------
        theta: float, angle of incidence of radiation in radians

        Returns
        ----------
        (NDArray) Effective substrate refractive index.
        """

        # calculate the effective substrate refractive index
        inside1 = ((np.imag(self.ref_index)**2 + np.real(self.ref_index)**2)**2
                    + 2 * (np.imag(self.ref_index) - np.real(self.ref_index))
                    * (np.imag(self.ref_index) + np.real(self.ref_index))
                    * np.sin(theta)**2 + np.sin(theta)**4)
        inside2 = 0.5 * (-np.imag(self.ref_index)**2
                        + np.real(self.ref_index)**2
                        + np.sin(theta)**2
                        + np.sqrt(inside1))

        return np.sqrt(inside2)

    def path_length(self, inc_medium: OpticalMedium, theta: float) -> NDArray:
        """
        Calculates the estimated optical path length through substrate
        given incident medium and incident angle.

        Parameters
        -------------
        inc_medium: OpticalMedium, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        ------------
        (ArrayLike) Length of the path through the substrate.
        """

        # calculate the denominator
        den = np.sqrt(1 - (np.abs(inc_medium.ref_index)**2
                           * (np.sin(theta)**2)
                           / self.effective_index(theta)**2))

        # return the path length
        return self.thickness / den

    def fresnel_coefficients(
            self,
            inc_medium: OpticalMedium,
            theta: float
    ) -> Dict[str, NDArray]:
        """
        Calculates the fresnel amplitudes & intensities of the bare substrate.

        Parameters
        -----------
        inc_medium: OpticalMedium, complex refractive index on incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray]
        {
            'Ts' : s-polarized Fresnel Transmission Intensity,
            'Tp' : p-polarized Fresnel Transmission Intensity,
            'Rs' : s-polarized Fresnel Reflection Intensity,
            'Rp' : p-polarized Fresnel Reflection Intensity,
            'rs' : s-polarized Fresnel Reflection Amplitude,
            'rp' : p-polarized Fresnel Reflection Amplitude
        }
        """

        if not self.ref_index.shape == inc_medium.ref_index.shape:
            raise ValueError(
                f'shaped not equal: {self.ref_index.shape} != {inc_medium.ref_index.shape}.')

        # Calculation of the Fresnel Amplitude Coefficients
        rs_num = (inc_medium.ref_index * np.cos(theta)
                  - np.sqrt(self.ref_index**2- inc_medium.ref_index**2 * np.sin(theta)**2))
        rs_den = (inc_medium.ref_index * np.cos(theta)
                  + np.sqrt(self.ref_index**2 - inc_medium.ref_index**2 * np.sin(theta)**2))
        r_s = rs_num / rs_den

        rp_num = (self.ref_index**2
                  * np.cos(theta)
                  - inc_medium.ref_index
                  * np.sqrt(self.ref_index**2 - inc_medium.ref_index**2 * np.sin(theta)**2))
        rp_den = (self.ref_index**2
                  * np.cos(theta)
                  + inc_medium.ref_index
                  * np.sqrt(self.ref_index**2 - inc_medium.ref_index**2 * np.sin(theta)**2))
        r_p = -rp_num / rp_den

        # Calculation of Fresnel Intensities for bare substrate interface
        big_r_s = np.abs(r_s)**2
        big_r_p = np.abs(r_p)**2

        return {
            'Ts': (1 - big_r_s),     # S-polarized transmission
            'Tp': (1 - big_r_p),     # P-polarized transmission
            'Rs': big_r_s,           # S-polarized reflection
            'Rp': big_r_p,           # P-polarized reflection
            'rs': r_s,               # S-polarized fresnel Amplitude Coefficient
            'rp': r_p                # P-polarized fresnel Amplitude Coefficient
        }

    def admittance(
            self,
            inc_medium: OpticalMedium,
            theta: float
    ) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of substrate and incident
        medium interface.

        args
        -------------
        inc_medium: OpticalMedium, complex refractive index of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the medium-incident interface,
            'p': p-polarized admittance of the medium-incident interface }

        Raises
        ----------
        ValueError if inc_medium shape does not match
        substrate.ref_index shape or  theta <= 0.
        """

        if not np.shape(self.ref_index) == np.shape(inc_medium.ref_index):
            raise ValueError("inc_medium.ref_index shape must match substrate.ref_index")

        # calculate complex dialectric constants (square the values)
        # for both the substrate and the incident medium
        sub_dialectrics = self.ref_index**2
        med_dialectrics = inc_medium.ref_index**2

        # Calculate S and P admittances
        admit_s = np.sqrt(sub_dialectrics - med_dialectrics * np.sin(theta)**2)
        admit_p = sub_dialectrics / admit_s

        return {'s': admit_s, 'p': admit_p}

    def eff_admittance(
            self,
            inc_medium: OpticalMedium,
            theta: float
    ) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of substrate and incident
        medium interface using the effective refractive index
        of the substrate.

        args
        -----------
        inc_medium: OpticalMedium, complex refractive index of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray] {
            's': s-polarized admittance of the medium-incident interface,
            'p': p-polarized admittance of the medium-incident interface }

        Raises
        ----------
        ValueError if inc_medium shape does not match
        substrate.ref_index shape or  theta <= 0.
        """

        if not np.shape(self.ref_index) == np.shape(inc_medium.ref_index):
            raise ValueError("inc_medium.ref_index shape must match substrate.ref_index")

        # calculate complex dialectric constants (square the values)
        # for both the substrate and the incident medium
        sub_dialectrics = self.effective_index(theta)**2
        med_dialectrics = inc_medium.ref_index**2

        # Calculate S and P admittances
        admit_s = np.sqrt(sub_dialectrics - med_dialectrics * np.sin(theta)**2)
        admit_p = sub_dialectrics / admit_s

        return {'s': admit_s, 'p': admit_p}
