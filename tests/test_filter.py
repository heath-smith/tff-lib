#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the ThinFilmFilter
class.

Usage
---------
>>> python -m unittest -v tests.test_filter
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
from tff_lib import ThinFilm, OpticalMedium, Substrate, FilmStack

# class under test
from tff_lib import ThinFilmFilter

class TestThinFilmFilter(unittest.TestCase):
    """
    Test suite for ThinFilmFilter() class.
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
        cls._theta = 0.0
        cls._sub_thickness = cls.test_data['input']['sub_thick']
        cls._wavelengths = cls.test_data['input']['wv']
        cls._high_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low_mat = [complex(x) for x in cls.test_data['input']['low_mat']]
        cls._layers = cls.test_data['input']['layers']
        cls._sub_ref_index = cls.test_data['input']['substrate']
        cls._fresnel = cls.test_data['output']['fresnel_film']
        cls._filspec = cls.test_data['output']['filspec']

        # generate test ThinFilmFilter
        cls._stack = [
            ThinFilm(
                lyr[0],
                lyr[1],
                cls._wavelengths,
                cls._high_mat if lyr[0] == 'H' else cls._low_mat
            )
            for lyr in cls._layers
        ]

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']],
            'air'
        )

        # test substrate
        cls._substrate = Substrate(
            cls._sub_thickness, cls._wavelengths, cls._sub_ref_index)

        # test FilmStack
        cls._filmstack = FilmStack(cls._stack)

    def test_filter_init_(self):
        """
        test __init__()
        """

    def test_fresnel_coefficients(self):
        """
        PENDING -----> test fresnel_coefficients()
        """

    def test_filter_spectrum(self):
        """
        PENDING -----> test filter_spectrum()
        """

    def test_filter_invalid_inputs(self):
        """
        PENDING -----> test __init__() with invalid input values
        """

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
