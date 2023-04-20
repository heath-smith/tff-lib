#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the Substrate
class.

Usage
---------
>>> python -m unittest -v tests.test_medium
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
from tff_lib import Substrate

class TestFilmStack(unittest.TestCase):
    """
    Test suite for Substrate() class.
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
        cls._wavelengths = cls.test_data['input']['wv']
        cls._admittance = cls.test_data['output']['admit_delta']

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']],
            'air'
        )

    def test_medium_init_defaults(self):
        """
        PENDING -----> test __init__()
        """

    def test_admittance(self):
        """
        PENDING -----> test admittance()
        """

    def test_medium_invalid_inputs(self):
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
