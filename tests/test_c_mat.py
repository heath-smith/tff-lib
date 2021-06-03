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
        Set up the test environment. All tests will use data from the
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

    def test_c_mat(self):
        """
        Test c_mat() method. Use expected output from admit_delta in
        the test_expected.json file to supply input parameters for c_mat().
        """

        sys.stdout.write('\nTesting c_mat()... ')

        # setup input data from test_expected.json
        cmat_input = self.output_data['admit_delta_expected']
        test_nsfilm = np.array(cmat_input['ns_film'])
        test_npfilm = np.array(cmat_input['np_film'])
        test_delta = np.array(cmat_input['delta'])
        # get expected values from test_expected.json
        cmat_expected = self.output_data['c_mat_expected']

        #------------ test c_mat() -----------#
        sys.stdout.write('\n\tcorrect data types... ')
        test_cmat = tff.c_mat(test_nsfilm, test_npfilm, test_delta)

        # floating point comparison threshold
        thresh = 1

        # assert output is equal to expected
        test_fails = 0
        for key in test_cmat:
            try:
                nptest.assert_almost_equal(np.array(test_cmat[key]).astype(complex),
                    np.array(cmat_expected[key]).astype(complex),
                    decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #---------- test c_mat() with incorrect data types ------------#
        sys.stdout.write('\n\tincorrect data types... ')
        nsfilm_wrong_type = np.array(test_nsfilm).astype(complex)
        npfilm_wrong_type = np.array(test_npfilm).astype(complex)
        delta_wrong_type = np.array(test_delta).astype(complex)

        # assert that a TypeError exception is raised
        # wrong type ns_film
        with self.assertRaises(TypeError):
            tff.c_mat(nsfilm_wrong_type, test_npfilm, test_delta)
        # assert that a TypeError exception is raised
        # wrong type np_film
        with self.assertRaises(TypeError):
            tff.c_mat(test_nsfilm, npfilm_wrong_type, test_delta)
        # assert that a TypeError exception is raised
        # wrong type delta
        with self.assertRaises(TypeError):
            tff.c_mat(test_nsfilm, test_npfilm, delta_wrong_type)

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()