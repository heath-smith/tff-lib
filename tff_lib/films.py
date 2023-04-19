"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
import random
from typing import SupportsIndex, Iterable, Dict
import numpy as np
from numpy.typing import NDArray
from .medium import OpticalMedium

class WritePropertyError(Exception):
    """
    Custom property write error class.
    """

class ThinFilm():
    """
    Abstract object representing a thin optical film.

    Properties
    ----------
        thickness: float, Read-Write, layer thickness in nanometers

    Attributes
    ----------
        material: str, 'H' or 'L'
        wavelengths: Iterable[float], 1-D wavelength array for material
        ref_index: Iterable[complex], 1-D refractive indices as fx of wavelengths
    """

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
        wavelengths: Iterable[float], 1-D wavelength array for material
        ref_index: Iterable[complex], 1-D refractive indices as f(x) of wavelength

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

        # set thickness using setter
        self.thickness = float(thickness)

        # set attributes
        self.material = str(material).upper()
        self.wavelengths = [float(x) for x in wavelengths]
        self.ref_index = [complex(y) for y in ref_index]

    @property
    def thickness(self) -> float:
        """
        float, layer thickness in nanometers, must be greater than zero
        """
        return self._thickness

    @thickness.setter
    def thickness(self, new_thickness:float):
        if new_thickness <= 0:
            raise ValueError("thickness must be greater than 0")
        self._thickness = new_thickness

    def split_film(self):
        """
        Split the thin film layer into two layers with height
        1/2 of the calling instance. Reduces calling instance
        thickness to 1/2 its original value.

        Returns
        ----------
        ThinFilm() object with thickness value half of calling instance
        thickness attribute.
        """

        new_thickness = self.thickness * 0.5
        self.thickness = new_thickness

        return ThinFilm(
            self.material, new_thickness, self.wavelengths, self.ref_index)

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
        Subtracts thickness values of two thin films with matching
        attributes. Returns absolute difference between thickness
        values.
        """

        if not self.wavelengths == film.wavelengths:
            raise ValueError("Wavelength values must be equivalent to add thin films.")
        if not self.ref_index == film.ref_index:
            raise ValueError("Refractive indices must be equivalent to add thin films.")
        if not self.material == film.material:
            raise ValueError("'material' values must be equivalent to add thin films.")

        self.thickness = abs(self.thickness - film.thickness)    # update self.thickness

        return type(self)(self.material, self.thickness, self.wavelengths, self.ref_index)


class FilmStack():
    """
    Abstract object representing a stack of optical
    thin films.

    Properties
    ----------
        stack: Iterable[ThinFilm], Read-Write, list of thin film layers
        total_thick: float, ReadOnly, total thickness of stack
        num_layers: float, Read-Only, number of layers in stack
        layers: Iterable[float], Read-Write, layer thickness values
        matrix: NDArray, Read-Only, matrix of refractive indices for each layer

    Attributes
    ----------
        max_total_thick: float, maximum total thickness (nanometers) (default 20_000)
        max_layers: int, max limit number of ThinFilm layers (default 25)
        min_layers: int, minimum number of ThinFilm layers (default 5)
        first_lyr_min_thick: float, first lyr min thickness (nanometers) (default 500)
        min_thick: float, all other layers min thickness (nanometers) (default 10)

    See Also
    ----------
    >>> tff_lib.ThinFilm()
    """

    def __init__(self, films: Iterable[ThinFilm], **kwargs) -> None:
        """
        Initialize class and set attributes.

        args
        ----------
        films: Iterable[ThinFilm], thin film layer to be used in the film stack

        kwargs
        ----------
        max_total_thick: float, max total thickness in nanometers (default 20_000)
        max_layers: int, maximum number of ThinFilm layers (default 25)
        min_layers: int, minimum number of ThinFilm layers (default 5)
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
        min_layers = kwargs.get('min_layers', 5)

        # validate inputs
        if max_total_thick <= 0:
            raise ValueError("max_total_thick must be greater than 0.")
        if first_lyr_min_thick <= 0:
            raise ValueError("first_lyr_min_thick must be greater than 0.")
        if min_thick <= 0:
            raise ValueError("min_thick must be greater than 0.")
        if max_layers <= 0:
            raise ValueError("max_layers must be greater than 0.")
        if min_layers <= 0:
            raise ValueError("min_layers must be greater than 0.")
        if min_layers >= max_layers:
            raise ValueError("min_layers must be less than max_layers")

        # set properties (managed-attributes) using setter
        self.stack = films

        # set attributes (un-managed)
        self.max_total_thick = float(max_total_thick)
        self.max_layers = int(max_layers)
        self.first_lyr_min_thick = float(first_lyr_min_thick)
        self.min_thick = float(min_thick)
        self.min_layers = int(min_layers)

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
        Property - Iterable[float], 1-D layer thickness values.
        """
        return [lyr.thickness for lyr in self._stack]

    @layers.setter
    def layers(self, lyrs:Iterable[float]):
        if not len(self._stack) == len(lyrs):
            raise ValueError("lyrs must be same length as stack")
        for i, film in enumerate(self._stack):
            film.thickness = lyrs[i]

    @property
    def matrix(self) -> NDArray:
        """
        Property - numpy.NDArray, a matrix of each film's refractive indices.
        """
        matrix = np.zeros(
            (len(self._stack), len(self._stack[0].ref_index))).astype(np.complex128)

        for i, lyr in enumerate(self._stack):
            matrix[i, :] = lyr.ref_index

        return matrix

    @matrix.setter
    def matrix(self, val):
        raise WritePropertyError("matrix is a read-only property")

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

        # because films is mutable, python will re-use the 'films' object.
        # to avoid erroneous behavior, create a new object for _stack so
        # films will be re-evaluated each time it is changed
        # ref ---> https://python-guide.readthedocs.io/en/latest/writing/gotchas/
        self._stack = [f for f in films]

    def insert_layer(self, layer: ThinFilm, index: SupportsIndex) -> None:
        """
        Insert a new thin film layer before index.
        Updates num_layers and total_thick attributes.
        """
        ## ---> This one actually needs some more thought because layers
        ## must alternate between 'H' and 'L' material.. so layers cannot
        ## just be inserted freely...

        #self._stack.insert(index, layer)
        raise NotImplementedError

    def append_layer(self, layer: ThinFilm) -> None:
        """
        Appends a new thin film layer to end of stack.
        Updates num_layers and total_thick attributes.
        """
        if self._stack[-1].material == layer.material:
            raise ValueError("film stacks must have alternating materials")

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

    def admittance(self, inc_medium:OpticalMedium, theta:float) -> Dict[str, NDArray]:
        """
        Calculate admittances of thin film filter.

        Parameters
        -------------
        inc_medium: OpticalMedium, refractive indices of incident medium
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

        dialec_med = [m**2 for m in inc_medium.ref_index]
        dialec_films = [film**2 for film in self.matrix]

        # Calculate admittances & phase factors for each layer
        admit_s = np.ones((self.num_layers, len(self.layers[0].wavelengths)))
        admit_p = np.ones((self.num_layers, len(self.layers[0].wavelengths)))
        delta = np.ones((self.num_layers, len(self.layers[0].wavelengths)))

        # iterate each layer in thin film stack
        for i, lyr in enumerate(self.layers):
            admit_s[i, :] = np.sqrt(dialec_films[i, :] - dialec_med * np.sin(theta)**2)
            admit_p[i, :] = dialec_films[i, :] / admit_s[i, :]
            delta[i, :] = (
                2 * np.pi * lyr.thickness * np.sqrt(
                dialec_films[i, :] - dialec_med * np.sin(theta)**2)) / self.layers[0].wavelengths

        # Flip layer-based arrays ns_film, np_film, delta
        # since the last layer is the top layer
        admit_s = np.flipud(admit_s)
        admit_p = np.flipud(admit_p)
        delta = np.flipud(delta)

        return {'s': admit_s, 'p': admit_p, 'delta': delta}

    def char_matrix(self, inc_medium:OpticalMedium, theta:float) -> Dict[str, NDArray]:
        """
        Calculates the characteristic matrix for a thin film stack.

        Parameters
        -----------
        inc_medium: OpticalMedium, refractive indices of incident medium
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


class RandomFilmStack(FilmStack):
    """
    A randomized film stack. Inherits public attributes, properties,
     and methods from FilmStack().

    See Also
    ----------
    >>> tff_lib.FilmStack()
    """

    def __init__(
            self,
            wavelengths: Iterable[float],
            high_mat: Iterable[complex],
            low_mat: Iterable[complex],
            **kwargs
    ) -> None:
        """
        Generates a randomized thin film stack object.

        args
        ----------
        wavelengths: Iterable[float], 1-D wavelength array for materials
        high_mat: Iterable[complex], 1-D refractive indices of high index material
        low_mat: Iterable[complex], 1-D refractive indices of low index material

        kwargs
        ----------
        max_total_thick: float, max total thickness in nanometers (default 20_000)
        max_layers: int, maximum number of ThinFilm layers (default 20)
        min_layers: int, minimum number of ThinFilm layers (default 5)
        first_lyr_min_thick: float, min thickness of first layer in nanometers (default 500)
        min_thick: float, min thickness of remaining layers in nanometers (default 10)

        See Also
        ----------
        >>> tff_lib.FilmStack()
        """

        max_layers = kwargs.get('max_layers', 20)
        min_layers = kwargs.get('min_layers', 5)
        total_thick = kwargs.get('max_total_thick', 20_000)

        if not len(high_mat) == len(wavelengths):
            raise ValueError("high_mat length must match wavelengths")
        if not len(low_mat) == len(wavelengths):
            raise ValueError("low_mat length must match wavelengths")
        if not len(low_mat) == len(high_mat):
            raise ValueError("high_mat length must match low_mat length")

        # generate a random number of layers between min_layers - max_layers
        rand_layers = min_layers + round((max_layers - min_layers) * random.uniform(0.0, 1.0))
        scale_factor = 2 * total_thick / rand_layers

        # random film stack
        rand_stack = []

        # generate thin film layers
        for i in range(rand_layers):
            rand_stack.append(
                ThinFilm(
                    "H" if i % 2 == 0 else "L",
                    scale_factor * random.uniform(0.0, 1.0),
                    wavelengths,
                    high_mat if i % 2 == 0 else low_mat
                )
            )

        # pass kwargs to parent __init__()
        super().__init__(rand_stack, **kwargs)
