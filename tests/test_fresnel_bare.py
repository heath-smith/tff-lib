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

    def test_fresnel_bare(self):
        """
        Test fresnel_bare() method by initializing test input data and asserting
        the test output arrays are equal to the expected output arrays.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays. See FILSPEC lines 447-464
        in moe.py module for definition of the test arrays.
        """

        # update output stream
        sys.stdout.write('\nTesting fresnel_bare()... ')

        # get spectral info from initial_conditions in json
        test_cond = self.input_data['initial_conditions']
        # define expected test results
        test_exp = self.output_data['fresnel_bare_expected']

        # incident medium refractive index (assume incident medium is air)
        test_i_n = np.ones(np.shape(test_cond['wv'])).astype(complex)
        # substrate refractive index
        test_s_n = np.array(test_cond['substrate']).astype(complex)
        # incident angle theta
        test_theta = 0.0

        #--------- test w/ no 'units' -----------#
        sys.stdout.write('\n\t*w/o units param... ')
        test_fb_no_units = tff.fresnel_bare(test_i_n, test_s_n, test_theta)

        # set decimal threshold for array comparison
        thresh = 4

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in test_exp:
            try:
                nptest.assert_array_almost_equal(test_fb_no_units[key],
                    test_exp[key], decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #--------- test w/ 'units' param ----------#
        sys.stdout.write('\n\t*with units = "deg"... ')
        test_fb_units = tff.fresnel_bare(test_i_n, test_s_n, test_theta, units='deg')
        for key in test_exp:
            try:
                nptest.assert_array_almost_equal(test_fb_no_units[key],
                    test_exp[key], decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #--------- test w/ non-string as 'units' param ----------#
        sys.stdout.write('\n\t*with units = "some string"... ')
        # assert that a UnitError exception is raised
        with self.assertRaises(UnitError):
            tff.fresnel_bare(test_i_n, test_s_n, test_theta,
                                units='some string')

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()