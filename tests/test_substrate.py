#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the Substrate
class.

Usage
---------
>>> python -m unittest -v tests.test_substrate
"""

# import external packages
import unittest
from pathlib import Path
import sys
import os
import json
import time
import numpy.testing as nptest
from tff_lib import OpticalMedium

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
        cls._thickness = cls.test_data['input']['sub_thick']
        cls._wavelengths = cls.test_data['input']['wv']
        cls._ref_index = cls.test_data['input']['substrate']
        cls._admittance = cls.test_data['output']['admit_delta']
        cls._path_length = cls.test_data['output']['path_length']
        cls._fresnel = cls.test_data['output']['fresnel_bare']

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']],
            'air'
        )

    def test_substrate_init(self):
        """
        test __init__()
        """

        sub = Substrate(self._thickness, self._wavelengths, self._ref_index)


        # assert attributes are valid
        self.assertEqual(self._thickness, sub.thickness)
        nptest.assert_array_almost_equal(
            self._wavelengths, sub.wavelengths, decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._ref_index, sub.ref_index, decimal=self._precision)

    def test_absorption_coefficients(self):
        """
        PENDING -----> test absorption_coefficients()
        """

    def test_effective_index(self):
        """
        PENDING -----> test effective_index()
        """

    def test_path_length(self):
        """
        PENDING -----> test path_length()
        """

    def test_fresnel_coefficients(self):
        """
        PENDING -----> test fresnel_coefficients()
        """

    def test_admittance(self):
        """
        PENDING -----> test admittance()
        """

    def test_substrate_invalid_inputs(self):
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
