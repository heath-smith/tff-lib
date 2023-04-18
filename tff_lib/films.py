"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
from typing import SupportsIndex, Iterable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from .substrate import Substrate

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

    def characteristic_matrix(
            self, ns_film:ArrayLike, np_film:ArrayLike, delta:ArrayLike) -> Dict[str, NDArray]:
        """
        Calculates the characteristic matrix for a thin film stack.

        Parameters
        -----------
        ns_film: ArrayLike, s-polarized admittance of the film stack layers.
        np_film: ArrayLike, p-polarized admittance of the film stack layers.
        delta: ArrayLike, phase upon reflection for each film.

        Returns
        ------------
        Dict[str, NDArray] {
            'S11': matrix entry,
            'S12': matrix entry,
            'S21': matrix entry,
            'S22': matrix entry,
            'P11': matrix entry,
            'P12': matrix entry,
            'P21': matrix entry,
            'P22': matrix entry }
        """



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
