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
import numpy.testing as nptest

# class under test
from tff_lib import OpticalMedium

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
        cls._precision = 14

        # setup input data from test_expected.json
        cls._theta = 0.0
        cls._name = 'air'
        cls._wavelengths = cls.test_data['input']['wv']
        cls._ref_index = [complex(1.0, 0) for i in cls.test_data['input']['wv']]
        cls._admittance = cls.test_data['output']['admit_delta']


    def test_medium_init(self):
        """
        test __init__()
        """

        opt = OpticalMedium(self._wavelengths, self._ref_index, self._name)

        # assert attributes are valid
        self.assertEqual(self._name, opt.name)
        nptest.assert_array_almost_equal(
            self._wavelengths, opt.wavelengths, decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._ref_index, opt.ref_index, decimal=self._precision)

    def test_admittance(self):
        """
        test admittance()
        """

        opt = OpticalMedium(self._wavelengths, self._ref_index, self._name)

        adm = opt.admittance(self._theta)

        nptest.assert_array_almost_equal(
            self._admittance['ns_inc'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_inc'], adm['p'], decimal=self._precision)

    def test_medium_invalid_inputs(self):
        """
        PENDING -----> test __init__() with invalid input values
        """

        # test with shortened wavelength array
        with self.assertRaises(ValueError):
            OpticalMedium(self._wavelengths[:-2], self._ref_index, self._name)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()
