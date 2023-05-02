"""
This module contains the ThinFilmFilter class.
"""

from typing import Dict
from numpy.typing import NDArray
import numpy as np
from .substrate import Substrate
from .films import FilmStack
from .medium import OpticalMedium

class ThinFilmFilter():
    """
    Abstract representation of a thin-film
    optical filter. A filter consists of 3 parts,
    the substrate, a thin film stack, and the incident
    medium.

    Attributes
    ----------
        substrate: Substrate, the substrate of the thin film filter
        film_stack: FilmStack, optical thin film stack
        incident: OpticalMedium,
    """

    def __init__(
            self,
            substrate: Substrate,
            film_stack: FilmStack,
            incident: OpticalMedium
    ) -> None:
        """
        Initializes the ThinFilmFilter class.

        Parameters
        ----------
        substrate: Substrate, the substrate of the filter
        film_stack: FilmStack, optical thin film stack
        incident: OpticalMedium, the optical medium
        """

        self.substrate = substrate
        self.film_stack = film_stack
        self.incident = incident

    def fresnel_coefficients(self, theta: float, reflection: str) -> Dict[str, NDArray]:
        """
        Calculates the fresnel amplitudes & intensities of filter given the
        substrate admittance and incident medium admittance.

        Parameters
        ------------
        admit_sub: Tuple, substrate admittance (s-polarized, p-polarized)
        char_matrix: Tuple, incident medium admittance (s-polarized, p-polarized)
        reflection: str, one of 'medium' or 'substrate'

        Returns
        -------------
        Dict[str, NDArray] {
            'Ts' : s-polarized Fresnel Transmission Intensity,
            'Tp' : p-polarized Fresnel Transmission Intensity,
            'Rs' : s-polarized Fresnel Reflection Intensity,
            'Rp' : p-polarized Fresnel Reflection Intensity,
            'ts' : s-polarized Fresnel Transmission Amplitude,
            'tp' : p-polarized Fresnel Transmission Amplitude,
            'rs' : s-polarized Fresnel Reflection Amplitude,
            'rp' : p-polarized Fresnel Reflection Amplitude}
        """

        if reflection == 'medium':
            admit_sub = self.substrate.admittance(self.incident, theta)
            admit_inc = self.incident.admittance(self.incident, theta)
            char_matrix = self.film_stack.char_matrix(self.incident, theta)
        elif reflection == 'substrate':
            theta_inverse = np.arcsin(
                self.incident.ref_index / self.substrate.ref_index * np.sin(theta))
            admit_sub = self.substrate.eff_admittance(self.incident, theta_inverse)
            admit_inc = self.incident.admittance(self.incident, theta_inverse)
            char_matrix = self.film_stack.char_matrix(self.incident, theta_inverse)
        else:
            raise ValueError("reflection must be one of 'medium' or 'substrate'")

        # calculate admittance of the incident interface
        admit_inc_int = {
            's': ((char_matrix['S21'] - admit_sub['s'] * char_matrix['S22'])
                    / (char_matrix['S11'] - admit_sub['s'] * char_matrix['S12'])),
            'p': ((char_matrix['P21'] + admit_sub['p'] * char_matrix['P22'])
                    / (char_matrix['P11'] + admit_sub['p'] * char_matrix['P12']))}

        # Calculation of the Fresnel Amplitude Coefficients
        fresnel = {
            'rs': (admit_inc['s'] + admit_inc_int['s']) / (admit_inc['s'] - admit_inc_int['s']),
            'rp': (admit_inc['p'] - admit_inc_int['p']) / (admit_inc['p'] + admit_inc_int['p'])}
        fresnel['ts'] = (
            (1 + fresnel['rs']) / (char_matrix['S11'] - char_matrix['S12']  * admit_sub['s']))
        fresnel['tp'] = (
            (1 + fresnel['rp']) / (char_matrix['P11'] + char_matrix['P12'] * admit_sub['p']))

        # Calculation of the Fresnel Amplitude Intensities
        fresnel['Rs'] = np.abs(fresnel['rs'])**2
        fresnel['Rp'] = np.abs(fresnel['rp'])**2
        fresnel['Ts'] = np.real(admit_sub['s'] / admit_inc['s']) * np.abs(fresnel['ts'])**2
        fresnel['Tp'] = np.real(admit_sub['p'] / admit_inc['p']) * np.abs(fresnel['tp'])**2

        return fresnel

    def filter_spectrum(self, theta: float) -> Dict[str, NDArray]:
        """
        Calculates the transmission and reflection spectra of the
        thin-film interference filter.

        Parameters
        ------------
        substrate: Substrate, the substrate for the thin film stack
        inc_medium: ArrayLike, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        -----------
        Dict[str, NDArray] {
            'T' : average transmission spectrum over wavelength range ([Tp + Ts] / 2),
            'Ts' : s-polarized transmission spectrum wavelength range,
            'Tp' : p-polarized transmission spectrum over wavelength range,
            'R' : average reflection spectrum over wavelength range,
            'Rs' : s-polarized reflection spectrum over wavelength range,
            'Rp' : p-polarized reflection spectrum over wavelength range }
        """

        # Calculate the path length through the substrate
        sub_p_len = self.substrate.path_length(self.incident, theta)

        # Fresnel coefficients of incident medium / substrate interface
        sub_fresnel = self.substrate.fresnel_coefficients(self.incident, theta)

        # reflection originating from incident medium
        inc_med_ref = self.fresnel_coefficients(theta, 'medium')

        # reflection originating from substrate
        sub_ref = self.fresnel_coefficients(theta, 'substrate')

        # calculate the absorption coefficients for multiple reflections
        alpha = self.substrate.absorption_coefficients(n_ref=4)

        ## might be better to have a function from here down, which
        ## takes in the results from the previous method calls
        ## instead of calling all of these from C++

        # calculate filter reflection
        spec = {'Rs': (
            inc_med_ref['Rs']
            + ((inc_med_ref['Ts']**2) * sub_fresnel['Rs'] * np.exp(-2 * alpha * sub_p_len))
            / (1 - (sub_ref['Rs'] * sub_fresnel['Rs'] * np.exp(-2 * alpha * sub_p_len)))
        )}
        spec['Rp'] = (
            inc_med_ref['Rp']
            + ((inc_med_ref['Tp']**2)  * sub_fresnel['Rp'] * np.exp(-2 * alpha * sub_p_len))
            / (1 - (sub_ref['Rp']  * sub_fresnel['Rp'] * np.exp(-2 * alpha * sub_p_len)))
        )
        spec['R'] = (spec['Rs'] + spec['Rp']) / 2

        # calculate filter transmission
        spec['Ts'] = (
            (inc_med_ref['Ts'] * sub_fresnel['Ts'] * np.exp(-alpha * sub_p_len))
            / (1 - (sub_ref['Rs'] * sub_fresnel['Rs'] * np.exp(-2 * alpha * sub_p_len)))
        )
        spec['Tp'] = (
            (inc_med_ref['Tp'] * sub_fresnel['Tp'] * np.exp(-alpha * sub_p_len))
            / (1 - (sub_ref['Rp'] * sub_fresnel['Rp'] * np.exp(-2 * alpha * sub_p_len)))
        )
        spec['T'] = (spec['Ts'] + spec['Tp']) / 2

        return spec
