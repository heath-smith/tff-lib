#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module test_tff_lib.py contains the test suite for the
ThinFilmFilter class and methods.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests/test_tff_lib.py
>>> python -m unittest tests.test_tff_lib.TestThinFilmFilter.test_fresnel_bare
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

# import tff_lib for testing
from tff_lib.tff_lib import ThinFilmFilter as tff
from tff_lib.exceptions import UnitError

class TestThinFilmFilter(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment. All tests here will use data from the
        'test_input.json' file in the 'data' folder. Note that this is not
        a class method. It will be executed before each test method.
        """

        # write status to output stream
        sys.stdout.write('\nSetting up test...')

        # static navigation to top-level package directory
        self.package_dir = Path(__file__).resolve().parent.parent

        # define directory with data files
        self.data_dir = os.path.join(self.package_dir, 'data')

        # define input/output file names
        self.input_file = os.path.join(self.data_dir, 'test_input.json')
        self.output_file = os.path.join(self.data_dir, 'test_expected.json')

        # read in json file with test input data
        with open(self.input_file) as in_file:
            self.input_data = json.load(in_file)

        # read in json file with test expected output data
        with open(self.output_file) as out_file:
            self.output_data = json.load(out_file)

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_fresnel_film(self):
        """
        Test the fresnel_film() method. Creates an instance of MOE class
        and pulls input data from test_expected.json.
        """

        sys.stdout.write('\nTesting fresnel_film()... ')

        # new instance of MOE class
        test_moe_ff = MOE(self.data_path)

        # initialize test input data
        test_admit_delta_out = self.expected_data['admit_delta_expected']
        test_cmat_out = self.expected_data['c_mat_expected']

        # call fresnel_film() and capture output
        test_ff_output = test_moe_ff.fresnel_film(test_admit_delta_out, test_cmat_out)

        # initialize fresnel_film() expected output
        ff_expected = self.expected_data['fresnel_film_expected']

        # threshold for floating point comparison
        thresh = 10

        # assert test output is equal to expected values
        test_fails = 0
        for key in test_ff_output:
            try:
                nptest.assert_allclose(np.array(test_ff_output[key]).astype(complex),
                    np.array(ff_expected[key]).astype(complex),
                    rtol=1e-14, atol=5)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | rtol=1e-14, atol=5')

    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()