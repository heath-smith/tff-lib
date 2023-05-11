"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
import copy
from typing import SupportsIndex, Iterable, Dict
import numpy as np
from numpy.typing import NDArray
from .medium import OpticalMedium

class WritePropertyError(Exception):
    """
    Custom property write error class.
    """

class ThinFilm(OpticalMedium):
    """
    ThinFilm is a sub-class of medium.OpticalMedium, representing an optical
    thin-film used to in thin-film filter construction. Inherits all public
    methods, attributes, and properties of OpticalMedium, with additional
    methods to perform common operations on optical thin-films.

    Methods
    ----------
    >>> __add__(self, film: ThinFilm) -> ThinFilm
    >>> __sub__(self, film: ThinFilm) -> ThinFilm
    >>> split_film(self, ratio: float = 0.5) -> ThinFilm

    See Also
    ----------
    >>> class OpticalMedium(
            waves: Iterable[float],
            nref: Iterable[complex],
            **kwargs: Any
        )
    """

    def __init__(self, waves: Iterable[float], nref: Iterable[complex], **kwargs) -> None:
        """
        Initializes the ThinFilm class.

        args
        ----------
        waves: Iterable[float], 1-D array of wavelengths in nanometers
        nref: Iterable[complex], 1-D array of complex refractive indices

        kwargs
        ----------
        thick: float, medium thickness in nanometers. -1 if unspecified, or must
            be greater than zero. (default -1)
        ntype: int, refractive index material type (default -1)

        Raises
        ----------
        ValueError
            if thick not -1 or > 0, len(waves) != len(nref),
             waves or nref not 1-D.

        See Also
        ----------
        >>> class OpticalMedium(
                waves: Iterable[float],
                nref: Iterable[complex],
                **kwargs: Any
            )
        """
        super().__init__(waves, nref, **kwargs)

    def __add__(self, film: 'ThinFilm') -> 'ThinFilm':
        """
        Adds thickness values of two thin films. Must have matching
        wavelengths, refractive indices, and material attributes.
        """

        if not all(self.waves == film.waves):
            raise ValueError("Wavelength values must be equivalent to add thin films.")
        if not all(self.nref == film.nref):
            raise ValueError("Refractive indices must be equivalent to add thin films.")
        if not self.ntype == film.ntype:
            raise ValueError("'material' values must be equivalent to add thin films.")

        self.thick = self.thick + film.thick    # update self.thick

        return type(self)(self.waves, self.nref, thick=self.thick, ntype=self.ntype)

    def __sub__(self, film: 'ThinFilm') -> 'ThinFilm':
        """
        Subtracts thickness values of two thin films with matching
        attributes. Returns absolute difference between thickness
        values.
        """

        if not all(self.waves == film.waves):
            raise ValueError("Wavelength values must be equivalent to add thin films.")
        if not all(self.nref == film.nref):
            raise ValueError("Refractive indices must be equivalent to add thin films.")
        if not self.ntype == film.ntype:
            raise ValueError("'material' values must be equivalent to add thin films.")

        self.thick = abs(self.thick - film.thick)    # update self.thick

        return type(self)(self.waves, self.nref, thick=self.thick, ntype=self.ntype)

    def split_film(self, ratio: float = 0.5) -> 'ThinFilm':
        """
        Split the thin film layer into two layers with heights set
        by the ratio. Changes calling instance to the the thickness
        set by multiplying the current thickness * ratio.

        args
        ----------
        ratio: float, ratio used for partitioning thickness, must be between (0, 1)

        Returns
        ----------
        ThinFilm() object with thickness specified by thickness * (1 - ratio)

        Raises
        ----------
        ValueError, if ratio not between 0 and 1
        """

        if not 0 < ratio < 1:
            raise ValueError("ratio must be between 0 and 1")
        out_thick = self.thick * (1 - ratio)
        self.thick = self.thick * ratio

        return ThinFilm(
            self.waves, self.nref, thick=out_thick, ntype=self.ntype)


class FilmStack():
    """
    The FilmStack class is an abstraction of a physical thin-film stack
    containing ThinFilm layers with finite thicknesses.

    Properties
    ----------
        stack: Iterable[ThinFilm], Read-Write, list of thin film layers
        total_thick: float, ReadOnly, total thickness of stack
        num_layers: int, Read-Only, number of layers in stack
        layers: Iterable[float], Read-Write, layer thickness values
        matrix: NDArray, Read-Only, matrix of refractive indices for each layer

    Attributes
    ----------
        max_total_thick: float, maximum total thickness (nanometers) (default 20_000)
        max_layers: int, max limit number of ThinFilm layers (default 25)
        min_layers: int, minimum number of ThinFilm layers (default 5)
        first_lyr_min_thick: float, first lyr min thickness (nanometers) (default 500)
        min_thick: float, all other layers min thickness (nanometers) (default 10)

    Methods
    ----------
    >>> insert_layer(self, lyr: ThinFilm, index: SupportsIndex) -> None
    >>> insert_split_layer(self, lyr: ThinFilm, index: SupportsIndex, ratio: float = 0.5) -> None
    >>> append_layer(self, lyr: ThinFilm) -> None
    >>> remove_layer(self, index: SupportsIndex = -1) -> ThinFilm
    >>> get_layer(self, index: SupportsIndex) -> ThinFilm
    >>> admittance(self, inc: OpticalMedium, theta: float) -> Dict[str, NDArray]
    >>> char_matrix(self, inc: OpticalMedium, theta: float) -> Dict[str, NDArray]

    See Also
    ----------
    >>> class ThinFilm(
                waves: Iterable[float],
                nref: Iterable[complex],
                **kwargs
        )
    >>> class RandomFilmStack(
                waves: Iterable[float],
                high: Iterable[complex],
                low: Iterable[complex],
                **kwargs
        )
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
        max_thick: float, max thickness of layers in nanometers (default 2500)

        See Also
        ----------
        >>> class ThinFilm(
                    waves: Iterable[float],
                    nref: Iterable[complex],
                    **kwargs
            ) -> None
        """

        max_total_thick = kwargs.get('max_total_thick', 20_000)
        max_layers = kwargs.get('max_layers', 25)
        first_lyr_min_thick = kwargs.get('first_lyr_min_thick', 500)
        min_thick = kwargs.get('min_thick', 10)
        max_thick = kwargs.get('max_thick', 2500)
        min_layers = kwargs.get('min_layers', 5)

        # validate inputs
        if max_total_thick <= 0:
            raise ValueError("max_total_thick must be greater than 0.")
        if first_lyr_min_thick <= 0:
            raise ValueError("first_lyr_min_thick must be greater than 0.")
        if min_thick <= 0:
            raise ValueError("min_thick must be greater than 0.")
        if max_thick <= 0:
            raise ValueError("max_thick must be greater than 0.")
        if max_thick <= min_thick:
            raise ValueError("max_thick must be greater than min_thick.")
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

    def __str__(self):
        """
        Prints the readable form of FilmStack.
        """

        return f"""[{", ".join([
            str((i, lyr.thick, lyr.ntype)) for i, lyr in enumerate(self.stack)])}]"""

    @property
    def total_thick(self):
        """
        Property/Managed Attribute - float, total thickness of the
        stack in nanometers. (Read Only)
        """
        return np.sum([lyr.thick for lyr in self._stack])

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
        return [lyr.thick for lyr in self._stack]

    @layers.setter
    def layers(self, lyrs:Iterable[float]):
        if not len(self._stack) == len(lyrs):
            raise ValueError("lyrs must be same length as stack")
        for i, film in enumerate(self._stack):
            film.thick = lyrs[i]

    @property
    def matrix(self) -> NDArray:
        """
        Property - numpy.NDArray, a matrix of each film's refractive indices.
        """
        matrix = np.zeros(
            (len(self._stack), len(self._stack[0].nref))).astype(np.complex128)

        for i, lyr in enumerate(self._stack):
            matrix[i, :] = lyr.nref

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

        # verify there is at least 1 film
        if len(films) < 1:
            raise ValueError("film stack must contain at least 1 layer")

        # validate the wavelengths of each film
        for film in films:
            if not all(film.waves == films[0].waves):
                raise ValueError("All films must have same wavelength values.")

        # because films is mutable, python will re-use the 'films' object.
        # to avoid erroneous behavior, create a new object (ie: deep copy)
        # for _stack so films will be re-evaluated each time setter is called
        # ref ---> https://python-guide.readthedocs.io/en/latest/writing/gotchas/
        self._stack = copy.deepcopy(films)

    def insert_layer(self, lyr: ThinFilm, index: SupportsIndex) -> None:
        """
        Inserts a new thin-film before index.

        args
        ----------
        lyr: ThinFilm, thin-film object to be inserted
        index: SupportsIndex, position to insert thin-film
        """

        # copy stack
        new_stack = list(self._stack)

        # insert into stack
        new_stack.insert(index, lyr)

        # update property
        self.stack = new_stack

    def insert_split_layer(
            self,
            lyr: ThinFilm,
            index: SupportsIndex,
            ratio: float = 0.5
    ) -> None:
        """
        Inserts a new thin-film layer into the stack by partitioning the layer
        specified at index according to the ratio.

        args
        ----------
        lyr: ThinFilm, the thin-film object to be inserted
        index: SupportsIndex, the position in the stack to insert lyr

        kwargs
        ----------
        ratio: float, the ratio to partition the layer at index
        """

        # create a new stack with lyrs[0 : i-1]
        new_stack = self._stack[:index]

        # partition the layer specified at index using ratio
        lyr_a = self.get_layer(index)
        lyr_b = lyr_a.split_film(ratio)

        # insert the first partition, new layer, second partition
        new_stack.append(lyr_a)
        new_stack.append(lyr)
        new_stack.append(lyr_b)

        # insert remaining layers
        new_stack += self._stack[index + 1:]

        # update the stack with setter
        self.stack = new_stack

    def append_layer(self, lyr: ThinFilm) -> None:
        """
        Appends a new thin film layer to end of stack.

        args
        ----------
        lyr: ThinFilm, the thin-film object to be appended to stack
        """
        self._stack.append(lyr)

    def remove_layer(self, index: SupportsIndex = -1) -> ThinFilm:
        """
        Remove and return thin film layer at index (default last).
        If popped layer is not first or last layer, combines layers
        index + 1 and index - 1.

        args
        ----------
        index: SupportsIndex, index of layer to remove (default -1)

        Returns
        ----------
        ThinFilm, the thin-film object that was removed from the stack
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

        args
        ----------
        index: SupportsIndex, index of layer to be returned

        Returns
        ----------
        ThinFilm, the thin-film layer at index
        """
        return self._stack[index]

    def admittance(self, inc: OpticalMedium, theta: float) -> Dict[str, NDArray]:
        """
        Calculate admittances of the film stack.

        Parameters
        -------------
        inc: OpticalMedium, refractive indices of incident medium
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

        # calculate complex dialectric constants
        dialec_med = inc.nref**2
        dialec_films = self.matrix**2

        # allocate memory to store results
        admit_s = np.ones_like(self.matrix)
        admit_p = np.ones_like(self.matrix)
        delta = np.ones_like(self.matrix)

        # Calculate admittances & phase factors for each layer
        for i, lyr in enumerate(self.stack):
            admit_s[i, :] = np.sqrt(dialec_films[i, :] - dialec_med * np.sin(theta)**2)
            admit_p[i, :] = dialec_films[i, :] / admit_s[i, :]
            delta[i, :] = (
                2 * np.pi * lyr.thick * np.sqrt(
                dialec_films[i, :] - dialec_med * np.sin(theta)**2)) / lyr.waves

        # Flip layer-based arrays ns_film, np_film, delta
        # since the last layer is the top layer
        admit_s = np.flipud(admit_s)
        admit_p = np.flipud(admit_p)
        delta = np.flipud(delta)

        return {'s': admit_s, 'p': admit_p, 'delta': delta}

    def char_matrix(self, inc: OpticalMedium, theta: float) -> Dict[str, NDArray]:
        """
        Calculates the characteristic matrix for a thin film stack.

        Parameters
        -----------
        inc: OpticalMedium, refractive indices of incident medium
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

        admit = self.admittance(inc, theta)

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
            'S11': np.ones_like(elements['s11'][0, :]),
            'S12': np.zeros_like(elements['s11'][0, :]),
            'S21': np.zeros_like(elements['s11'][0, :]),
            'S22': np.ones_like(elements['s11'][0, :]),
            'P11': np.ones_like(elements['p11'][0, :]),
            'P12': np.zeros_like(elements['p11'][0, :]),
            'P21': np.zeros_like(elements['p11'][0, :]),
            'P22': np.ones_like(elements['p11'][0, :])
        }

        # calculate the matrix values for each layer
        for i in range(self.num_layers):

            # deep copy previous matrices to avoid
            # complications with mutable objects
            _matrices = copy.deepcopy(matrices)

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
