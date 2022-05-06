#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suite for the
char_matrix() method.

Examples
---------
>>> python -m unittest -v tests.test_char_matrix
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json
import warnings

# import function to test
from tff_lib.tff_lib import char_matrix
from tff_lib.utils import convert_to_numpy

class TestCharMatrix(unittest.TestCase):
    """
    Test suite for char_matrix() method.
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

        # setup input data from test_expected.json
        cls.test_nsfilm = np.asarray(cls.test_data['output']['admit_delta']['ns_film'])
        cls.test_npfilm = np.asarray(cls.test_data['output']['admit_delta']['np_film'])
        cls.test_delta = np.asarray(cls.test_data['output']['admit_delta']['delta'])
        cls.test_exp = convert_to_numpy(cls.test_data['output']['char_matrix'], is_complex=True)

        # floating point comparison threshold
        cls.precision = 12

    def test_char_matrix(self):
        """
        ----------> default test case
        """

        # test the char_matrix()
        _cmat = char_matrix(self.test_nsfilm, self.test_npfilm, self.test_delta)

        # verify the output
        nptest.assert_array_almost_equal(self.test_exp['S11'], _cmat['S11'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['S12'], _cmat['S12'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['S21'], _cmat['S21'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['S22'], _cmat['S22'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['P11'], _cmat['P11'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['P12'], _cmat['P12'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['P21'], _cmat['P21'], decimal=self.precision)
        nptest.assert_array_almost_equal(self.test_exp['P22'], _cmat['P22'], decimal=self.precision)




    def test_char_matrix_errors(self):
        """
        ----------> w/ incorrect input data
        """

        # check for TypeErrors
        with self.assertRaises(TypeError):
            char_matrix([], self.test_npfilm, self.test_delta)
        with self.assertRaises(TypeError):
            char_matrix(self.test_nsfilm, [], self.test_delta)
        with self.assertRaises(TypeError):
            char_matrix(self.test_nsfilm, self.test_npfilm, [])

        # check value errors
        with self.assertRaises(ValueError):
            wrong_nsfilm = np.append(self.test_nsfilm, 0)
            char_matrix(wrong_nsfilm, self.test_npfilm, self.test_delta)
        with self.assertRaises(ValueError):
            wrong_npfilm = np.append(self.test_npfilm, 0)
            char_matrix(self.test_nsfilm, wrong_npfilm, self.test_delta)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()