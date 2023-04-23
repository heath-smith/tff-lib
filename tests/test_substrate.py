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

        # setup input / output data
        cls._theta = 0.0
        cls._thickness = cls.test_data['input']['sub_thick'] * 10**6
        cls._wavelengths = cls.test_data['input']['wv']
        cls._ref_index = cls.test_data['input']['substrate']
        cls._admittance = cls.test_data['output']['admit_delta']
        cls._path_length = cls.test_data['output']['path_length']
        cls._fresnel = cls.test_data['output']['fresnel_bare']

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']])

    def test_substrate_init(self):
        """
        test __init__()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        # assert attributes are valid
        self.assertEqual(self._thickness, sub.thickness)
        nptest.assert_array_almost_equal(
            self._wavelengths, sub.wavelengths, decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._ref_index, sub.ref_index, decimal=self._precision)

    def test_absorption_coefficients(self):
        """
        test absorption_coefficients()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        abs_coeff = sub.absorption_coefficients()

        # expect to be all zero's since substrate has
        # no non-zero imaginary values
        self.assertEqual(0, sum(abs_coeff))

    def test_effective_index(self):
        """
        test effective_index()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        eff = sub.effective_index(self._theta)

        # when theta == 0, expect eff_index == ref_index
        nptest.assert_array_almost_equal(
            self._ref_index, eff, decimal=self._precision)

    def test_path_length(self):
        """
        test path_length()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        p_len = sub.path_length(self._medium, self._theta)

        nptest.assert_array_almost_equal(
            self._path_length, p_len, decimal=self._precision)

    def test_fresnel_coefficients(self):
        """
        test fresnel_coefficients()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        fresnel = sub.fresnel_coefficients(self._medium, self._theta)

        nptest.assert_array_almost_equal(
            self._fresnel['Ts'], fresnel['Ts'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Tp'], fresnel['Tp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Rs'], fresnel['Rs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Rp'], fresnel['Rp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['rs'], fresnel['rs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['rp'], fresnel['rp'], decimal=self._precision)

    def test_admittance(self):
        """
        PENDING -----> test admittance()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        adm = sub.admittance(self._medium, self._theta)

        nptest.assert_array_almost_equal(
            self._admittance['ns_sub'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_sub'], adm['p'], decimal=self._precision)

    def test_eff_admittance(self):
        """
        PENDING -----> test eff_admittance()
        """

        sub = Substrate(self._wavelengths, self._ref_index, self._thickness)

        adm = sub.eff_admittance(self._medium, self._theta)

        # expect effective admittance to equal admittance
        # at theta == 0
        nptest.assert_array_almost_equal(
            self._admittance['ns_sub'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_sub'], adm['p'], decimal=self._precision)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
