#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test suite for
the fresnel_bare() method.

Examples
---------
>>> python -m unittest -v tests.test_fresnel_bare
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json

# import tff_lib for testing
from tff_lib.tff_lib import fresnel_bare

class TestFresnelBare(unittest.TestCase):
    """
    Test class for fresnel_bare() method.
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

        # incident medium refractive index (assume incident medium is air)
        cls.test_med = np.ones(np.shape(cls.test_data['input']['wv'])).astype(complex)
        # substrate refractive index
        cls.test_sub = np.array(cls.test_data['input']['substrate']).astype(complex)
        # incident angle theta
        cls.test_theta = 0.0
        # define the expected output
        cls.test_exp = cls.test_data['output']['fresnel_bare']

        # define array comparison threshold
        cls.precision = 15

    def test_fresnel_bare_nounits(self):
        """
        ----------> default test case
        """

        # test fresnel_bare() with no 'units' arg
        _fb = fresnel_bare(self.test_sub, self.test_med, self.test_theta)

        # verify the output
        nptest.assert_array_almost_equal(self.test_exp['Ts'], _fb['Ts'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Tp'], _fb['Tp'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Rs'], _fb['Rs'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Rp'], _fb['Rp'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['rs'], _fb['rs'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['rp'], _fb['rp'], decimal=self.precision)


    def test_fresnel_bare_deg(self):
        """
        ----------> w/ 'deg' as units
        """

        # test fresnel_bare() with no 'units' arg
        _fb = fresnel_bare(self.test_sub, self.test_med, self.test_theta, units='deg')

        # verify the output
        nptest.assert_array_almost_equal(self.test_exp['Ts'], _fb['Ts'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Tp'], _fb['Tp'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Rs'], _fb['Rs'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['Rp'], _fb['Rp'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['rs'], _fb['rs'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['rp'], _fb['rp'], decimal=self.precision)

    def test_fresnel_bare_errors(self):
        """
        ----------> w/ wrong input types
        """

        # test for TypeErrors
        with self.assertRaises(TypeError):
            fresnel_bare(self.test_sub, self.test_med, self.test_theta, units=123)
        with self.assertRaises(TypeError):
            fresnel_bare([], self.test_med, self.test_theta)
        with self.assertRaises(TypeError):
            fresnel_bare(self.test_sub, [], self.test_theta)
        with self.assertRaises(TypeError):
            fresnel_bare(self.test_sub, self.test_med, "theta")

        # test for ValueErrors
        with self.assertRaises(ValueError):
            wrong_sub = np.append(self.test_sub, 0)
            fresnel_bare(wrong_sub, self.test_med, self.test_theta)
        with self.assertRaises(ValueError):
            fresnel_bare(self.test_sub, self.test_med, self.test_theta, units="wrong string")


    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()