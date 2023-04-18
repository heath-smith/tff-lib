"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
from typing import SupportsIndex, Iterable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike

class ThinFilm():
    """
    Abstract object representing a thin optical film.
    """

    material = ''       # str 'H' or 'L'
    thickness = 0       # layer thickness, nanometers
    wavelengths = []    # wavelength array for material
    ref_index = []      # refractive indices as fx of wavelengths

    def __init__(
        self,
        material:str,
        thickness:float,
        wavelengths:Iterable[float],
        ref_index:Iterable[complex]
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        material: str, 'H' or 'L' (case-insensitive)
        thickness: float, layer thickness in nanometers, must be greater than zero
        wavelengths: Iterable[float], wavelength array
        ref_index: Iterable[complex], refractive indices as f(x) of wavelength

        Raises
        ----------
        ValueError, if len(wavelenghts) != len(ref_index) or material not in
            ('H', 'L')
        """

        # validate inputs
        if str(material).upper() not in ('H', 'L'):
            raise ValueError(f"material must be one of 'H' or 'L'. received {material}")
        if not len(wavelengths) == len(ref_index):
            raise ValueError("len(wavelengths) and len(ref_index) must be equal.")
        if float(thickness) <= 0:
            raise ValueError(f"thickness must be greater than zero. received {thickness}")

        self.material = str(material).upper()
        self.thickness = float(thickness)
        self.wavelengths = [float(x) for x in wavelengths]
        self.ref_index = [complex(y) for y in ref_index]

    def __add__(self, film:'ThinFilm'):
        """
        Adds thickness values of two thin films. Must have matching
        wavelengths, refractive indices, and material attributes.
        """

        if not self.wavelengths == film.wavelengths:
            raise ValueError("Wavelength values must be equivalent to add thin films.")
        if not self.ref_index == film.ref_index:
            raise ValueError("Refractive indices must be equivalent to add thin films.")
        if not self.material == film.material:
            raise ValueError("'material' values must be equivalent to add thin films.")

        self.thickness = self.thickness + film.thickness    # update self.thickness

        return type(self)(self.material, self.thickness, self.wavelengths, self.ref_index)

    def __sub__(self, film:'ThinFilm'):
        """
        Subtracts thickness values of two thin films. Must have matching
        wavelengths, refractive indices, and material attributes.
        """

        if not self.wavelengths == film.wavelengths:
            raise ValueError("Wavelength values must be equivalent to add thin films.")
        if not self.ref_index == film.ref_index:
            raise ValueError("Refractive indices must be equivalent to add thin films.")
        if not self.material == film.material:
            raise ValueError("'material' values must be equivalent to add thin films.")

        self.thickness = self.thickness - film.thickness    # update self.thickness

        return type(self)(self.material, self.thickness, self.wavelengths, self.ref_index)


class FilmStack():
    """
    Abstract object representing a stack of optical
    thin films.
    """

    layers = []             # list of thin film layers
    total_thick = 0         # total thickness in nm
    num_layers = 0          # number of layers
    max_total_thick = 0     # maximum total thickness (nanometers)
    max_layers = 0          # max limit for num layers
    first_lyr_min_thick = 0 # first lyr min thickness (nanometers)
    min_thick = 0           # all other layers min thickness (nanometers)

    def __init__(self, layers: Iterable[ThinFilm], **kwargs) -> None:
        """
        Initialize class and set attributes

        args
        ----------
        layers: Iterable, ThinFilms to be used in the film stack

        kwargs
        ----------
        max_total_thick: float, max total thickness in nanometers (default 20_000)
        max_layers: int, maximum number of ThinFilm layers (default 20)
        first_lyr_min_thick: float, min thickness of first layer in nanometers (default 500)
        min_thick: float, min thickness of remaining layers in nanometers (default 10)

        See Also
        ----------
        films.ThinFilm()
        """

        max_total_thick = kwargs.get('max_total_thick', 20_000)
        max_layers = kwargs.get('max_layers', 25)
        first_lyr_min_thick = kwargs.get('first_lyr_min_thick', 500)
        min_thick = kwargs.get('min_thick', 10)

        # validate inputs
        if len(layers) < 1:
            raise ValueError("layers must have at least 1 value.")
        if max_total_thick <= 0:
            raise ValueError("max_total_thick must be greater than 0.")
        if first_lyr_min_thick <= 0:
            raise ValueError("first_lyr_min_thick must be greater than 0.")
        if min_thick <= 0:
            raise ValueError("min_thick must be greater than 0.")
        if max_layers <= 0:
            raise ValueError("max_layers must be greater than 0.")

        self.layers = layers
        self.max_total_thick = float(max_total_thick)
        self.max_layers = int(max_layers)
        self.first_lyr_min_thick = float(first_lyr_min_thick)
        self.min_thick = min_thick

        # update num_layers, total_thick
        self.get_total_thick()
        self.num_layers = len(self.layers)

    def insert_layer(self, layer: ThinFilm, index: SupportsIndex) -> None:
        """
        Insert a new thin film layer before index.
        Updates num_layers and total_thick attributes.
        """
        self.layers.insert(index, layer)
        self.num_layers = len(self.layers)
        self.get_total_thick()

    def append_layer(self, layer: ThinFilm) -> None:
        """
        Appends a new thin film layer to end of stack.
        Updates num_layers and total_thick attributes.
        """
        self.layers.append(layer)
        self.num_layers = len(self.layers)
        self.get_total_thick()

    def remove_layer(self, index: SupportsIndex = -1) -> ThinFilm:
        """
        Remove and return thin film layer at index (default last).
        Updates num_layers and total_thick attributes. If popped layer
        is not first or last layer, combines layers index + 1 and index - 1.
        """

        # if index not first/last value in layers
        if 0 < index < self.num_layers - 1:

            # add surrounding layers
            new_lyr = self.layers[index - 1] + self.layers[index + 1]
            # update layer at (index - 1)
            self.layers[index - 1] = new_lyr
            # pop layers at (index + 1)
            self.layers.pop(index + 1)

        # pop layer at index
        lyr = self.layers.pop(index)

        # update num_layers and total_thickness
        self.num_layers = len(self.layers)
        self.get_total_thick()

        return lyr

    def get_layer(self, index: SupportsIndex) -> ThinFilm:
        """
        Return layer at index. Does not alter layer stack.
        """
        return self.layers[index]

    def get_total_thick(self) -> float:
        """
        Calculates total thickness of the stack in nanometers.
        """

        # reset total_thick
        self.total_thick = 0

        # iterate films in layers
        for lyr in self.layers:
            self.total_thick += lyr.thickness

        return self.total_thick

    def get_matrix(self) -> NDArray:
        """
        Returns a matrix of thin film refractive indices.

        Returns
        ----------
        numpy.ArrayLike, (N x M) matrix with thin film refractive indices of
            shape (num_layers x len(film.ref_index))
        """

        matrix = np.zeros((len(self.layers), len(self.layers[0].ref_index)))

        for i, lyr in enumerate(self.layers):
            matrix[i, :] = lyr

        return matrix

    def get_reverse_matrix(self) -> NDArray:
        """
        Returns thin film matrix in reverse order.

        See Also
        ----------
        >>> FilmStack.get_matrix()
        """
        return np.flipud(self.get_matrix())

    def get_stack(self) -> Iterable[ThinFilm]:
        """
        Returns a copy of the thin film stack.
        """
        return self.layers

    def get_reverse_stack(self) -> Iterable[ThinFilm]:
        """
        Return a copy of the stack in reverse order. Does not mutate
        the calling instance.
        """
        return list(reversed(self.layers))

    def admittance(self, inc_medium:ArrayLike, theta:float) -> Tuple:
        """
        Calculate admittances of thin film stack.

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

    def characteristic_matrix(self, ns_film:ArrayLike, np_film:ArrayLike, delta:ArrayLike) -> Dict:
        """
        Calculates the characteristic matrix for a thin film stack.

        Parameters
        -----------
        ns_film = s-polarized admittance of the film stack layers.\n
        np_film = p-polarized admittance of the film stack layers.\n
        delta = phase upon reflection for each film.

        Returns
        ------------
        (Dict) characteristic matrix\n
        { S11 (array): matrix entry\n
        S12 (array): matrix entry\n
        S21 (array): matrix entry\n
        S22 (array): matrix entry\n
        P11 (array): matrix entry\n
        P12 (array): matrix entry\n
        P21 (array): matrix entry\n
        P22 (array): matrix entry }

        Raises
        --------------
        ValueError if dimensions of ns_film, np_film, and delta do not match.
        TypeError if array types are not float.
        TypeError if arrays are not numpy.ndarray type.

        Examples
        ------------
        >>> char_matrix = c_mat(nsFilm, npFilm, delta)
        """

        # validate the input array shapes
        if not ns_film.shape == np_film.shape:
            raise ValueError(f'shape mismatch -----> {ns_film.shape} != {np_film.shape}.')

        # Calculation of the characteristic matrix elements
        # shape of 'delta' is (N-layers X len(wavelength range))
        elements = {
            's11': np.cos(delta),
            's22': np.cos(delta),
            'p11': np.cos(delta),
            'p22': np.cos(delta),
            's12': (1j / ns_film) * np.sin(delta),
            'p12': (-1j / np_film) * np.sin(delta),
            's21': (1j * ns_film) * np.sin(delta),
            'p21': (-1j * np_film) * np.sin(delta)
        }

        # Initialize the characteristic matrices
        matrices = {
            'S11': np.ones(np.shape(elements['s11'])[1]).astype(np.complex128),
            'S12': np.zeros(np.shape(elements['s11'])[1]).astype(np.complex128),
            'S21': np.zeros(np.shape(elements['s11'])[1]).astype(np.complex128),
            'S22': np.ones(np.shape(elements['s11'])[1]).astype(np.complex128),
            'P11': np.ones(np.shape(elements['p11'])[1]).astype(np.complex128),
            'P12': np.zeros(np.shape(elements['p11'])[1]).astype(np.complex128),
            'P21': np.zeros(np.shape(elements['p11'])[1]).astype(np.complex128),
            'P22': np.ones(np.shape(elements['p11'])[1]).astype(np.complex128)
        }

        for i in range(np.shape(elements['s11'])[0]):
            _matrices = matrices
            matrices['S11'] = ((_matrices['S11'] * elements['s11'][i, :])
                                + (_matrices['S12'] * elements['s21'][i, :]))
            matrices['S12'] = ((_matrices['S11'] * elements['s12'][i, :])
                                + (_matrices['S12'] * elements['s22'][i, :]))
            matrices['S21'] = ((_matrices['S21'] * elements['s11'][i, :])
                                + (_matrices['S22'] * elements['s21'][i, :]))
            matrices['S22'] = ((_matrices['S21'] * elements['s12'][i, :])
                                + (_matrices['S22'] * elements['s22'][i, :]))

            matrices['P11'] = ((_matrices['P11'] * elements['p11'][i, :])
                                + (_matrices['P12'] * elements['p21'][i, :]))
            matrices['P12'] = ((_matrices['P11'] * elements['p12'][i, :])
                                + (_matrices['P12'] * elements['p22'][i, :]))
            matrices['P21'] = ((_matrices['P21'] * elements['p11'][i, :])
                                + (_matrices['P22'] * elements['p21'][i, :]))
            matrices['P22'] = ((_matrices['P21'] * elements['p12'][i, :])
                                + (_matrices['P22'] * elements['p22'][i, :]))

        return matrices

    def fresnel_coefficients(self, admit_sub:Tuple, admit_inc_med:Tuple) -> Dict[str, NDArray]:
        """
        Calculates the fresnel amplitudes & intensities of film given the
        substrate admittance and incident medium admittance.

        Parameters
        ------------
        admit_sub: Tuple, substrate admittance (s-polarized, p-polarized)
        char_matrix: Tuple, incident medium admittance (s-polarized, p-polarized)

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
        char_matrix = self.characteristic_matrix()
        admit_s_sub, admit_p_sub = admit_sub
        admit_s_inc, admit_p_inc = admit_inc_med

        # calculate admittance of the incident interface
        admit_s_inc_int = ((char_matrix['S21'] - admit_s_sub * char_matrix['S22'])
                            / (char_matrix['S11'] - admit_s_sub * char_matrix['S12']))
        admit_p_inc_int = ((char_matrix['P21'] + admit_p_sub * char_matrix['P22'])
                            / (char_matrix['P11'] + admit_p_sub * char_matrix['P12']))

        # Calculation of the Fresnel Amplitude Coefficients
        fresnel = {
            'rs': (admit_s_inc + admit_s_inc_int) / (admit_s_inc - admit_s_inc_int),
            'rp': (admit_p_inc - admit_p_inc_int) / (admit_p_inc + admit_p_inc_int)}
        fresnel['ts'] = (
            (1 + fresnel['rs']) / (char_matrix['S11'] - char_matrix['S12']  * admit_s_sub))
        fresnel['tp'] = (
            (1 + fresnel['rp']) / (char_matrix['P11'] + char_matrix['P12'] * admit_p_sub))

        # Calculation of the Fresnel Amplitude Intensities
        fresnel['Rs'] = np.abs(fresnel['rs'])**2
        fresnel['Rp'] = np.abs(fresnel['rp'])**2
        fresnel['Ts'] = np.real(admit_s_sub / admit_s_inc) * np.abs(fresnel['ts'])**2
        fresnel['Tp'] = np.real(admit_p_sub / admit_p_inc) * np.abs(fresnel['tp'])**2

        return fresnel

    def get_spectrum(self,
        waves:ArrayLike, sub:ArrayLike, med:ArrayLike, films:ArrayLike, layers:list,
        sub_thick:int|float, theta:int|float, units:str='rad') -> Dict:
        """
        Calculates the transmission and reflection spectra of a
        thin-film interference filter.

        Parameters
        ------------
        waves =
        sub =
        med =
        high =
        low =
        films =
        layers =
        sub_thick =
        theta =
        units =

        Returns
        -----------
        (Dict) transmission and reflection value arrays\n
        { 'T' : average transmission spectrum over wavelength range ([Tp + Ts] / 2),\n
        'Ts' : s-polarized transmission spectrum wavelength range,\n
        'Tp' : p-polarized transmission spectrum over wavelength range,\n
        'R' : average reflection spectrum over wavelength range,\n
        'Rs' : s-polarized reflection spectrum over wavelength range,\n
        'Rp' : p-polarized reflection spectrum over wavelength range }

        Raises
        -------------
        ValueError, TypeError

        See Also
        -------------
        utils.sub_n_eff()
        utils.path_len()
        utils.film_matrix()
        """

        # convert incident angle from degrees to radians
        theta = theta * (np.pi / 180) if units == 'deg' else theta

        # calculate effective substrate refractive index
        sn_eff = utils.effective_index(sub, theta)

        # Calculate the path length through the substrate
        p_len = utils.path_length(sub_thick, med, sn_eff, theta)

        # Fresnel Amplitudes & Intensities
        # incident medium / substrate interface
        fr_bare = fresnel_bare(sub, med, theta)

        # reflection originates from incident medium
        med_adm = admit_delta(layers, waves, sub, med, films, theta)
        med_char = char_matrix(med_adm['ns_film'], med_adm['np_film'], med_adm['delta'])
        med_ref = fresnel_film(med_adm, med_char)

        # reflection originates from substrate
        theta_inv = np.arcsin(med / sub * np.sin(theta))
        layers_inv = [(str(v[0]), float(v[1])) for v in np.flipud(layers)]
        sub_adm = admit_delta(layers_inv, waves, sn_eff, med, np.flipud(films), theta_inv)
        sub_char = char_matrix(sub_adm['ns_film'], sub_adm['np_film'], np.flipud(sub_adm['delta']))
        sub_ref = fresnel_film(sub_adm, sub_char)

        # calculate the absorption coefficient for multiple reflections
        alpha = (4 * np.pi * np.imag(sub)) / waves

        # calculate filter reflection
        spec = {'Rs': (
            med_ref['Rs'] + ((med_ref['Ts']**2) * fr_bare['Rs'] * np.exp(-2 * alpha * p_len))
            / (1 - (sub_ref['Rs'] * fr_bare['Rs'] * np.exp(-2 * alpha * p_len)))
        )}
        spec['Rp'] = (
            med_ref['Rp']  + ((med_ref['Tp']**2)  * fr_bare['Rp'] * np.exp(-2 * alpha * p_len))
            / (1 - (sub_ref['Rp']  * fr_bare['Rp'] * np.exp(-2 * alpha * p_len)))
        )
        spec['R'] = (spec['Rs'] + spec['Rp']) / 2

        # calculate filter transmission
        spec['Ts'] = (
            (med_ref['Ts'] * fr_bare['Ts'] * np.exp(-alpha * p_len))
            / (1 - (sub_ref['Rs'] * fr_bare['Rs'] * np.exp(-2 * alpha * p_len)))
        )
        spec['Tp'] = (
            (med_ref['Tp'] * fr_bare['Tp'] * np.exp(-alpha * p_len))
            / (1 - (sub_ref['Rp'] * fr_bare['Rp'] * np.exp(-2 * alpha * p_len)))
        )
        spec['T'] = (spec['Ts'] + spec['Tp']) / 2

        return spec
