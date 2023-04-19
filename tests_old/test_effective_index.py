#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test suite for tff_lib.effective_index().


Examples
---------
>>> python -m unittest -v tests.test_effective_index
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
from tff_lib.utils import effective_index

class TestEffectiveIndex(unittest.TestCase):
    """
    Test class for effective_index()
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

        # get test substrate array
        cls.test_sub = np.asarray(cls.test_data['input']['substrate']).astype(np.complex128)
        cls.test_theta = 0
        cls.test_exp = np.asarray(cls.test_data['output']['effective_index']).astype(np.complex128)

        # floating point comparison threshold
        cls.precision = 14

    def test_effective_index(self):
        """
        ----------> default test case
        """

        # test effective_index()
        _eff = effective_index(self.test_sub, self.test_theta)

        # verify the output
        nptest.assert_array_almost_equal(self.test_exp, _eff, decimal=self.precision)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('Running teardown procedure...\nDONE!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()