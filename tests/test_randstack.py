#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the RandomFilmStack
class.

Usage
---------
>>> python -m unittest -v tests.test_randstack
"""

# import external packages
import unittest
from pathlib import Path
import sys
import os
import json
import time
import numpy as np
import numpy.testing as nptest
from tff_lib import ThinFilm, OpticalMedium

# class under test
from tff_lib import RandomFilmStack

class TestRandomFilmStack(unittest.TestCase):
    """
    Test suite for RandomFilmStack() class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # static navigation to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data')
        cls.data_file = os.path.join(cls.dir, 'test_data.json')

        # read in json file with test input data
        with open(cls.data_file, 'r', encoding='utf=8') as dat:
            cls.test_data = json.load(dat)

        # set decimal precision
        cls._precision = 12

        # setup input data from test_expected.json
        cls._material = 'H'
        cls._thickness = 0.5
        cls._wavelengths = cls.test_data['input']['wv']
        cls._high_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low_mat = [complex(x) for x in cls.test_data['input']['low_mat']]
        cls._layers = cls.test_data['input']['layers']
        cls._char_matrix = {
            k: [complex(x) for x in v]
            for k,v in cls.test_data['output']['char_matrix'].items()
        }
        cls._admittance = cls.test_data['output']['admit_delta']

        # generate test FilmStack
        cls._stack = [
            ThinFilm(
                cls._wavelengths,
                cls._high_mat if lyr[0] == 'H' else cls._low_mat,
                lyr[1],
                lyr[0]
            )
            for lyr in cls._layers
        ]

        # test total thick
        cls._total_thick = sum(l[1] for l in cls._layers)

        # test matrix
        cls._matrix = [
            cls._high_mat if l[0] == 'H' else cls._low_mat
            for l in cls._layers
        ]

        # default kwargs
        cls._defaults = {
            'max_total_thick': 20_000,
            'max_layers': 25,
            'min_layers': 5,
            'first_lyr_min_thick': 500,
            'min_thick': 10
        }

        # test kwargs (these are different from defaults)
        cls._kwargs = {
            'max_total_thick': 25_000,
            'max_layers': 20,
            'min_layers': 2,
            'first_lyr_min_thick': 250,
            'min_thick': 50
        }

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']])

    def test_rand_stack_init_defaults(self):
        """
        PENDING -----> test __init__()
        """

    def test_rand_stack_init_kwargs(self):
        """
        PENDING ----->  test __init__(**kwargs)
        """

    def test_rand_stack_invalid_inputs(self):
        """
        PENDING -----> test __init__() with invalid input values
        """

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
