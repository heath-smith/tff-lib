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

    def __init__(self, waves: Iterable[float], nref: Iterable[complex], **kwargs) -> None:
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

    def admittance(self, inc: 'OpticalMedium', theta: float) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of the incident->medium interface.

        Parameters
        -------------
        inc: OpticalMedium, incident medium of the radiation
        theta: float, angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the incident->medium interface,
            'p': p-polarized admittance of the incident->medium interface }
        """
        return super().admittance(inc, theta)

    def admit_effective(self, inc: 'OpticalMedium', theta: float) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of substrate and incident
        medium interface using the effective refractive index
        of the substrate.

        args
        -----------
        inc: OpticalMedium, complex refractive index of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray] {
            's': s-polarized admittance of the medium-incident interface,
            'p': p-polarized admittance of the medium-incident interface }

        Raises
        ----------
        ValueError, if inc_medium shape does not match
            substrate.ref_index shape or  theta <= 0.
        """
        return super().admit_effective(inc, theta)


    def path_length(self, inc: 'OpticalMedium', theta: float) -> NDArray:
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

        return super().path_length(inc, theta)


    def fresnel_coefficients(
            self,
            inc_medium: OpticalMedium,
            theta: float
    ) -> Dict[str, NDArray]:
        """
        Calculates the fresnel amplitudes & intensities of the medium.

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

