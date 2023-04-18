"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
from typing import SupportsIndex, Iterable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
from .substrate import Substrate

class WritePropertyError(Exception):
    pass

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

    # non-public attributes
    _stack = []             # list of thin film layers

    # public attributes
    max_total_thick = 0     # maximum total thickness (nanometers)
    max_layers = 0          # max limit for num layers
    first_lyr_min_thick = 0 # first lyr min thickness (nanometers)
    min_thick = 0           # all other layers min thickness (nanometers)

    def __init__(self, films: Iterable[ThinFilm], **kwargs) -> None:
        """
        Initialize class and set attributes

        args
        ----------
        films: Iterable[ThinFilm], thin film layer to be used in the film stack

        kwargs
        ----------
        max_total_thick: float, max total thickness in nanometers (default 20_000)
        max_layers: int, maximum number of ThinFilm layers (default 20)
        first_lyr_min_thick: float, min thickness of first layer in nanometers (default 500)
        min_thick: float, min thickness of remaining layers in nanometers (default 10)

        See Also
        ----------
        >>> class ThinFilm()
        """

        max_total_thick = kwargs.get('max_total_thick', 20_000)
        max_layers = kwargs.get('max_layers', 25)
        first_lyr_min_thick = kwargs.get('first_lyr_min_thick', 500)
        min_thick = kwargs.get('min_thick', 10)

        # validate inputs
        if len(films) < 1:
            raise ValueError("films must have at least 1 value.")
        if max_total_thick <= 0:
            raise ValueError("max_total_thick must be greater than 0.")
        if first_lyr_min_thick <= 0:
            raise ValueError("first_lyr_min_thick must be greater than 0.")
        if min_thick <= 0:
            raise ValueError("min_thick must be greater than 0.")
        if max_layers <= 0:
            raise ValueError("max_layers must be greater than 0.")

        # set properties (managed-attributes)
        self._stack = films
        self._total_thick = self.total_thick
        self._num_layers = len(self._layers)

        # set attributes (un-managed)
        self.max_total_thick = float(max_total_thick)
        self.max_layers = int(max_layers)
        self.first_lyr_min_thick = float(first_lyr_min_thick)
        self.min_thick = float(min_thick)

    @property
    def total_thick(self):
        """
        Property/Managed Attribute - float, total thickness of the
        stack in nanometers. (Read Only)
        """
        return np.sum([lyr.thickness for lyr in self._stack])

    @total_thick.setter
    def total_thick(self, val):
        raise WritePropertyError("total_thick is a read-only property")

    @property
    def num_layers(self) -> int:
        """
        Property/Managed Attribute - int, number of thin film layers
        in stack. (Read Only)
        """
        return len(self._stack)

    @num_layers.setter
    def num_layers(self, val):
        raise WritePropertyError("num_layers is a read-only property")

    @property
    def layers(self) -> Iterable[float]:
        """
        Property - Iterable[float], the layer thickness values.
        """
        return [lyr.thickness for lyr in self._stack]

    @layers.setter
    def layers(self, lyrs:Iterable[float]):
        if not len(self._stack) == len(lyrs):
            raise ValueError("lyrs must have same length as 'stack' property")
        for i, film in enumerate(self._stack):
            film.thickness = lyrs[i]

    @property
    def matrix(self) -> NDArray:
        """
        Property - numpy.NDArray, a matrix of each film's refractive indices.
        """
        matrix = np.zeros((len(self._stack), len(self._stack[0].ref_index)))

        for i, lyr in enumerate(self._stack):
            matrix[i, :] = lyr

        return matrix

    @property
    def stack(self) -> Iterable[ThinFilm]:
        """
        Property - Iterable[ThinFilm], the stack of thin films.
        """
        return self._stack

    @stack.setter
    def stack(self, films:Iterable[ThinFilm]):
        if len(films) < 1:
            raise ValueError("film stack must contain at least 1 layer")
        self._stack = films

    def insert_layer(self, layer: ThinFilm, index: SupportsIndex) -> None:
        """
        Insert a new thin film layer before index.
        Updates num_layers and total_thick attributes.
        """
        self._stack.insert(index, layer)

    def append_layer(self, layer: ThinFilm) -> None:
        """
        Appends a new thin film layer to end of stack.
        Updates num_layers and total_thick attributes.
        """
        self._stack.append(layer)

    def remove_layer(self, index: SupportsIndex = -1) -> ThinFilm:
        """
        Remove and return thin film layer at index (default last).
        Updates num_layers and total_thick attributes. If popped layer
        is not first or last layer, combines layers index + 1 and index - 1.
        """

        # if index not first/last value in layers
        if 0 < index < self.num_layers - 1:

            # add surrounding layers
            new_lyr = self._stack[index - 1] + self._stack[index + 1]
            # update layer at (index - 1)
            self._stack[index - 1] = new_lyr
            # pop layers at (index + 1)
            self._stack.pop(index + 1)

        # pop layer at index
        lyr = self._stack.pop(index)

        return lyr

    def get_layer(self, index: SupportsIndex) -> ThinFilm:
        """
        Return layer at index. Does not alter layer stack.
        """
        return self._stack[index]

    def admittance(self, inc_medium:ArrayLike, theta:float) -> Dict[str, NDArray]:
        """
        Calculate admittances of thin film filter.

        Parameters
        -------------
        inc_medium: ArrayLike, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

        Returns
        ----------
        Dict[str, NDArray] {
            's': s-polarized admittance of the film stack,
            'p': p-polarized admittance of the film stack,
            'delta': phase upon reflection for each film}

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

        return {'s': admit_s, 'p': admit_p, 'delta': delta}

    def characteristic_matrix(self, inc_medium:ArrayLike, theta:float) -> Dict[str, NDArray]:
        """
        Calculates the characteristic matrix for a thin film stack.

        Parameters
        -----------
        inc_medium: ArrayLike, refractive indices of incident medium
        theta: float, angle of incidence of radiation in radians

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

        admit = self.admittance(inc_medium, theta)

        # Calculation of the characteristic matrix elements
        # shape of 'delta' is (N-layers X len(wavelength range))
        elements = {
            's11': np.cos(admit['delta']),
            's22': np.cos(admit['delta']),
            'p11': np.cos(admit['delta']),
            'p22': np.cos(admit['delta']),
            's12': (1j / admit['s']) * np.sin(admit['delta']),
            'p12': (-1j / admit['p']) * np.sin(admit['delta']),
            's21': (1j * admit['s']) * np.sin(admit['delta']),
            'p21': (-1j * admit['p']) * np.sin(admit['delta'])
        }

        # Initialize the characteristic matrices
        matrices = {
            'S11': np.ones(np.shape(elements['s11'])[1]),
            'S12': np.zeros(np.shape(elements['s11'])[1]),
            'S21': np.zeros(np.shape(elements['s11'])[1]),
            'S22': np.ones(np.shape(elements['s11'])[1]),
            'P11': np.ones(np.shape(elements['p11'])[1]),
            'P12': np.zeros(np.shape(elements['p11'])[1]),
            'P21': np.zeros(np.shape(elements['p11'])[1]),
            'P22': np.ones(np.shape(elements['p11'])[1])
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
