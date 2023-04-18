"""
This module contains the ThinFilmFilter class.
"""

from typing import Dict
from numpy.typing import ArrayLike, NDArray
import numpy as np

class ThinFilmFilter():
    """
    Abstract representation of a thin-film
    optical filter.
    """

    substrate = None
    film_stack = None
    incident_medium = None


    def admittance(self, inc_medium:ArrayLike, theta:float) -> Tuple:
        """
        Calculate admittances of thin film filter.

        Parameters
        -------------
        inc_medium: ArrayLike, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        ----------
        (Tuple) (s-polarized admittance of the film stack,
        p-polarized admittance of the film stack,
        phase upon reflection for each film)


        References
        ----------
        https://www.svc.org/DigitalLibrary/documents/2008_Summer_AMacleod.pdf
        """

        dialec_med = [m**2 for m in inc_medium]
        dialec_films = [film**2 for film in self.get_matrix()]

        # Calculate admittances & phase factors for each layer
        admit_s = np.ones((self.num_layers, len(self.layers[0].wavelengths)))
        admit_p = np.ones((self.num_layers, len(self.layers[0].wavelengths)))
        delta = np.ones((self.num_layers, len(self.layers[0].wavelengths)))

        # iterate each layer in thin film stack
        for i, lyr in enumerate(self.layers):
            admit_s[i, :] = np.sqrt(dialec_films[i, :] - dialec_med * np.sin(theta)**2)
            admit_p[i, :] = dialec_films[i, :] / admit_s[i, :]
            delta[i, :] = (2 * np.pi * lyr.thickness * np.sqrt(dialec_films[i, :] - dialec_med * np.sin(theta)**2)) / self.layers[0].wavelengths

        # Flip layer-based arrays ns_film, np_film, delta
        # since the last layer is the top layer
        admit_s = np.flipud(admit_s)
        admit_p = np.flipud(admit_p)
        delta = np.flipud(delta)

        return admit_s, admit_p, delta


    def incident_reflection(theta):
        """
        Computes the reflection originating from the incident medium.
        """

        theta_inv = np.arcsin(med / sub * np.sin(theta))
        layers_inv = [(str(v[0]), float(v[1])) for v in np.flipud(layers)]
        sub_adm = self.admittance(layers_inv, waves, sub_n_eff, med, np.flipud(films), theta_inv)
        sub_char = self.characteristic_matrix(sub_adm['ns_film'], sub_adm['np_film'], np.flipud(sub_adm['delta']))
        sub_ref = self.fresnel_coefficients(sub_adm, sub_char)

        return sub_ref

    def substrate_reflection(theta):
        """
        Computes the reflection originating from the substrate.
        """

        med_adm = self.admittance(layers, waves, sub, med, films, theta)
        med_char = self.characteristic_matrix(med_adm['ns_film'], med_adm['np_film'], med_adm['delta'])
        inc_med_ref = self.fresnel_coefficients(med_adm, med_char) # ---> admit_sub, admit_inc_med

        return inc_med_ref

    def filter_spectrum(
            self, substrate:Substrate, inc_medium:ArrayLike, theta:float) -> Dict[str, NDArray]:
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

        # calculate effective substrate refractive index
        sub_n_eff = substrate.effective_index(theta)

        # Calculate the path length through the substrate
        sub_p_len = substrate.path_length(inc_medium, theta)

        # Fresnel coefficients of incident medium / substrate interface
        sub_fresnel = substrate.fresnel_coefficients(inc_medium, theta)

        # reflection originating from incident medium
        inc_med_ref = self.incident_reflection()

        # reflection originating from substrate
        sub_ref = self.substrate_reflection()

        # calculate the absorption coefficient for multiple reflections
        alpha = (4 * np.pi * np.imag(substrate.ref_index)) / substrate.wavelengths

        # calculate filter reflection
        spec = {'Rs': (
            inc_med_ref['Rs'] + ((inc_med_ref['Ts']**2) * sub_fresnel['Rs'] * np.exp(-2 * alpha * sub_p_len))
            / (1 - (sub_ref['Rs'] * sub_fresnel['Rs'] * np.exp(-2 * alpha * sub_p_len)))
        )}
        spec['Rp'] = (
            inc_med_ref['Rp']  + ((inc_med_ref['Tp']**2)  * sub_fresnel['Rp'] * np.exp(-2 * alpha * sub_p_len))
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

