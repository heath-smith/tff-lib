#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test suite for utils.path_len()
Examples
---------
>>> python -m unittest -v tests.test_path_length
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
from tff_lib.utils import path_length

class TestPathLength(unittest.TestCase):
    """
    Test class for utils.path_length()
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

        # get the test input data
        cls.test_thick = float(cls.test_data['input']['sub_thick'])
        cls.test_med = np.ones(np.shape(cls.test_data['input']['wv'])).astype(np.complex)
        cls.test_sub_eff = np.asarray(cls.test_data['output']['effective_index'])
        cls.test_theta = 0.0
        cls.test_exp = np.asarray(cls.test_data['output']['path_length'])

        # floating point comparison threshold
        cls.precision = 14

    def test_path_length(self):
        """
        ----------> default test case
        """

        # test path_length()
        _len = path_length(self.test_thick, self.test_med, self.test_sub_eff, self.test_theta)

        # verify the output
        nptest.assert_array_almost_equal(self.test_exp, _len, decimal=self.precision)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('\nRunning teardown procedure...\nDONE!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()