#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test suite for the
fresnel_film() method.

Examples
---------
>>> python -m unittest -v tests.test_fresnel_film
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
from tff_lib.tff_lib import fresnel_film
from tff_lib.utils import convert_to_numpy

class TestFresnelFilm(unittest.TestCase):
    """
    Test class for fresnel_film() method.
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

        # floating point comparison threshold
        cls.precision = 14

        # initialize test input data
        cls.test_admit_delta = convert_to_numpy(
                            cls.test_data['output']['admit_delta'], is_complex=True)
        cls.test_char_matrix = convert_to_numpy(
                            cls.test_data['output']['char_matrix'], is_complex=True)

        # initialize fresnel_film() expected output
        test_arrays = convert_to_numpy(
                            cls.test_data['output']['fresnel_film'], is_complex=True)
        cls.test_exp = {}
        for k, v in test_arrays.items():
            cls.test_exp[k] = np.conjugate(v)

    def test_fresnel_film(self):
        """
        ----------> default test case
        """

        # test fresnel_film() method
        _ff = fresnel_film(self.test_admit_delta, self.test_char_matrix)

        nptest.assert_array_almost_equal(self.test_exp['Ts'], _ff['Ts'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['Tp'], _ff['Tp'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['Rs'], _ff['Rs'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['Rp'], _ff['Rp'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['ts'], _ff['ts'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['tp'], _ff['tp'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['rs'], _ff['rs'], decimal=self.precision, verbose=True)
        nptest.assert_array_almost_equal(self.test_exp['rp'], _ff['rp'], decimal=self.precision, verbose=True)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('\nCleaning up resources...\nSUCCESS! ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()