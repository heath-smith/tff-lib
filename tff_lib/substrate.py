"""
This module contains the Substrate class.
"""

from typing import Dict, Iterable
from numpy.typing import NDArray
import numpy as np
from .medium import OpticalMedium

class Substrate():
    """
    Abstract object representing a physical substrate.

    Attributes
    -----------
        thickness: float, thickness in mm
        wavelengths: Iterable[float], 1-D wavelength values
        ref_index: Iterable[complex], 1-D complex refractive indices
    """

    def __init__(
            self,
            thickness: float,
            wavelengths: Iterable[float],
            ref_index: Iterable[complex]
    ) -> None:
        """
        Initialize the class attributes. Raises ValueError if wavelengths
        and ref_index not equal length or thickness less than zero.

        Parameters
        ----------
        thickness: float, thickness in mm
        wavelengths: Iterable[float], 1-D wavelength values
        ref_index: Iterable[complex] 1-D complex refractive indices
        """

        if not len(wavelengths) == len(ref_index):
            raise ValueError("wavelengths must be same length as ref_index")
        if thickness <= 0.0:
            raise ValueError("thickness must be greater than 0")

        self.thickness = float(thickness)
        self.wavelengths = np.array([float(x) for x in wavelengths])
        self.ref_index = np.array([complex(y) for y in ref_index])

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

        # calculate the numerator and denominator
        num = self.thickness * (10**6)
        den = np.sqrt(1 - (np.abs(inc_medium.ref_index)**2
                           * (np.sin(theta)**2)
                           / self.effective_index(theta)**2))

        # return the path length
        return num / den

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
            theta: float,
            use_eff_idx: bool = False
    ) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of substrate and incident
        medium interface.

        args
        -------------
        inc_medium: OpticalMedium, complex refractive index of incident medium
        theta: float, angle of incidence of radiation in radians

        kwargs
        ----------
        use_eff_idx: bool, use the effective refractive index (default False)

        Returns
        --------------
        (Tuple)
        (s-polarized admittance of the substrate medium,
        p-polarized admittance of the substrate medium)

        Raises ValueError if inc_medium shape does not match
        substrate.ref_index shape or  theta <= 0.
        """

        if not np.shape(self.ref_index) == np.shape(inc_medium.ref_index):
            raise ValueError("inc_medium.ref_index shape must match substrate.ref_index")

        # calculate complex dialectric constants (square the values)
        # for both the substrate and the incident medium
        if use_eff_idx:
            sub_dialectrics = self.effective_index(theta)**2
        else:
            sub_dialectrics = self.ref_index**2

        med_dialectrics = inc_medium.ref_index**2

        admit_s_sub = np.sqrt(sub_dialectrics - med_dialectrics * np.sin(theta)**2)
        admit_p_sub = sub_dialectrics / admit_s_sub

        return {'s': admit_s_sub, 'p': admit_p_sub}
