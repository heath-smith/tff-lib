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
    Python wrapper class for OpticalMedium C-extension module.

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


    Methods
    ----------
    >>> absorption_coeffs(self, n_reflect: int = 4) -> NDArray
    >>> nref_eff(self, theta: float|Iterable[float]) -> NDArray
    >>> admittance(self, inc: OpticalMedium, theta: float|Iterable[float]) -> Dict[str, NDArray]
    >>> admittance_eff(self, inc: OpticalMedium, theta: float|Iterable[float]) -> Dict[str, NDArray]
    >>> path_length(self, inc: OpticalMedium, theta: float|Iterable[float]) -> NDArray
    >>> fresnel_coeffs(self, inc: OpticalMedium, theta: float|Iterable[float]) -> Dict[str, NDArray]
    """

    def __init__(self, waves: Iterable[float], nref: Iterable[complex], **kwargs) -> None:
        """
        Initializes the OpticalMedium class.

        args
        ----------
        waves: Iterable[float], 1-D array of wavelengths in nanometers
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

    def absorption_coeffs(self) -> NDArray:
        """
        Calculate the absorption coefficients for n_ref reflections.

        Returns
        ----------
        NDArray, absorption coefficients as a function of wavelength
        """
        return super().absorption_coeffs()

    def nref_eff(self, theta: float|Iterable[float]) -> NDArray:
        """
        Calculates the effective refractive index through the
        medium. Requires a non-zero thickness.

        Parameters
        -----------
        theta: float|Iterable[float], angle of incidence of radiation in radians

        Returns
        ----------
        NDArray, Effective substrate refractive indices as a function
            of wavelength
        """
        return super().nref_eff(theta)

    def admittance(self, inc: 'OpticalMedium', theta: float|Iterable[float]) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of the incident->medium interface.

        Parameters
        -------------
        inc: OpticalMedium, incident medium of the radiation
        theta: float|Iterable[float], angle of incidence of radiation in radians

        Returns
        --------------
        Dict[str, NDArray] {
            's': s-polarized admittance of the incident->medium interface,
            'p': p-polarized admittance of the incident->medium interface }
        """
        return super().admittance(inc, theta)

    def admittance_eff(self, inc: 'OpticalMedium', theta: float|Iterable[float]) -> Dict[str, NDArray]:
        """
        Calculates optical admittance of substrate and incident
        medium interface using the effective refractive index
        of the substrate.

        args
        -----------
        inc: OpticalMedium, complex refractive index of incident medium
        theta: float|Iterable[float], angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray] {
            's': s-polarized admittance of the medium-incident interface,
            'p': p-polarized admittance of the medium-incident interface }

        Raises
        ----------
        ValueError, if inc shape does not match
            substrate.nref shape or  theta <= 0.
        """
        return super().admittance_eff(inc, theta)

    def path_length(self, inc: 'OpticalMedium', theta: float|Iterable[float]) -> NDArray:
        """
        Calculates the estimated optical path length through the medium
        given incident medium and incident angle.

        Parameters
        -------------
        inc: OpticalMedium, refractive indices of incident medium
        theta: float|Iterable[float], angle of incidence of radiation in radians

        Returns
        ------------
        NDArray, path length through the medium as a function of wavelength

        Raises
        ----------
        ValueError, if medium thickness < 0
        """

        return super().path_length(inc, theta)

    def fresnel_coeffs(self, inc: 'OpticalMedium', theta: float|Iterable[float]) -> Dict[str, NDArray]:
        """
        Calculates the fresnel amplitudes & intensities of the medium.

        Parameters
        -----------
        inc: OpticalMedium, complex refractive index on incident medium
        theta: float|Iterable[float], angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray]
        {
            'Ts' : s-polarized Fresnel Transmission Intensity,
            'Tp' : p-polarized Fresnel Transmission Intensity,
            'Rs' : s-polarized Fresnel Reflection Intensity,
            'Rp' : p-polarized Fresnel Reflection Intensity,
            'Fs' : s-polarized Fresnel Reflection Amplitude,
            'Fp' : p-polarized Fresnel Reflection Amplitude
        }
        """
        return super().fresnel_coeffs(inc, theta)
