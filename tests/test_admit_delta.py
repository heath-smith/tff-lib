#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test class for the
admit_delta() method.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests.test_admit_delta
>>> python -m unittest tests.test_admit_delta.<method>
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json

# import function to test
from tff_lib.tff_lib import admit_delta
from tff_lib.utils import film_matrix, convert_to_numpy

class TestAdmitDelta(unittest.TestCase):
    """
    Test class for admit_delta() method.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """
        # static navigation to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent, r'data')
        cls.out_dir = os.path.join(cls.dir, 'out')

        # read in json file with test input data
        with open(os.path.join(cls.dir, 'test_data.json')) as dat:
            cls.test_data = json.load(dat)

        # -- grab the wavelength values in advance
        # -- incident medium refractive index (assume incident medium is air)
        # -- substrate refractive index
        # -- incident angle theta
        # -- test layers
        # -- define the expected output
        # -- define array comparison threshold
        # -- film refractive indices
        cls.test_wv = np.asarray(cls.test_data['input']['wv'])
        cls.test_med = np.ones(np.shape(cls.test_data['input']['wv'])).astype(np.complex)
        cls.test_sub = np.asarray(cls.test_data['input']['substrate']).astype(np.complex)
        cls.test_theta = 0.0
        cls.test_layers = [tuple(v) for v in cls.test_data['input']['layers']]
        cls.test_exp = convert_to_numpy(cls.test_data['output']['admit_delta'], is_complex=True)
        cls.precision = 13
        cls.test_films = film_matrix(
            cls.test_layers,
            np.asarray(cls.test_data['input']['high_mat']),
            np.asarray(cls.test_data['input']['low_mat']),
        )

    def test_admit_delta_no_units(self):
        """
        ----------> default test case
        """

        # test admit_delta() without 'units'
        _ad = admit_delta(
            self.test_layers, self.test_wv, self.test_sub,
            self.test_med, self.test_films, self.test_theta)

        # verify the output
        nptest.assert_array_almost_equal(
            self.test_exp['ns_inc'], _ad['ns_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_inc'], _ad['np_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_sub'], _ad['ns_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_sub'], _ad['np_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_film'], _ad['ns_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_film'], _ad['np_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['delta'], _ad['delta'], decimal=self.precision)


    def test_admit_delta_deg(self):
        """
        ----------> w/ 'deg' as units arg
        """

        # test admit_delta() without 'units'
        _ad = admit_delta(
            self.test_layers, self.test_wv, self.test_sub,
            self.test_med, self.test_films, self.test_theta, units='deg')

        # verify the output
        nptest.assert_array_almost_equal(
            self.test_exp['ns_inc'], _ad['ns_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_inc'], _ad['np_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_sub'], _ad['ns_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_sub'], _ad['np_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_film'], _ad['ns_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_film'], _ad['np_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['delta'], _ad['delta'], decimal=self.precision)

    def test_admit_delta_errors(self):
        """
        ----------> w/ incorrect input data
        """

        # check for TypeErrors
        with self.assertRaises(TypeError):
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                self.test_med, self.test_films, self.test_theta, units=123)
        with self.assertRaises(TypeError):
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                self.test_med, self.test_films, "wrong theta")
        with self.assertRaises(TypeError):
            admit_delta(
                np.array([]), self.test_wv, self.test_sub,
                self.test_med, self.test_films, self.test_theta)
        with self.assertRaises(TypeError):
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                [], self.test_films, self.test_theta)

        # check for ValueErrors
        with self.assertRaises(ValueError):
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                self.test_med, self.test_films, self.test_theta, units='units')
        with self.assertRaises(ValueError):
            wrong_sub = np.append(self.test_sub, 0)
            admit_delta(
                self.test_layers, self.test_wv, wrong_sub,
                self.test_med, self.test_films, self.test_theta)
        with self.assertRaises(ValueError):
            wrong_med = np.append(self.test_med, 0)
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                wrong_med, self.test_films, self.test_theta)
        with self.assertRaises(ValueError):
            wrong_films = np.append(self.test_films, 0, axis=1)
            admit_delta(
                self.test_layers, self.test_wv, self.test_sub,
                self.test_med, wrong_films, self.test_theta)


    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()