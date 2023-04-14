"""
This module contains a collection of classes used
to describe thin films and film stacks.
"""

# import dependencies
from typing import SupportsIndex
import numpy as np

class ThinFilm():
    """
    Abstract object representing a thin optical film.
    """

    # attributes
    material = ''
    thickness = 0
    wavelengths = []
    refractive_index = []


class FilmStack():
    """
    Abstract object representing a stack of optical
    thin films.
    """

    stack = []      # list of thin film layers
    total_thick = 0 # total thickness in nm
    num_layers = 0  # number of layers

    def insert_layer(self, layer: ThinFilm, index: SupportsIndex) -> None:
        """
        Insert a new thin film layer before index.
        """
        self.stack.insert(index, layer)

    def pop_layer(self, index: SupportsIndex = -1) -> ThinFilm:
        """
        Remove and return thin film layer at index (default last).
        """
        return self.stack.pop(index)

    def get_thickness(self) -> float:
        """
        Returns total thickness of the stack.
        """
        return self.total_thick
