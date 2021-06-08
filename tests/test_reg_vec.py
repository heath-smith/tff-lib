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

class TestRegVec(unittest.TestCase):
    """
    Test class for the reg_vec() method of Thin Film Filter
    Library.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment. All tests here will use data from the
        'test_input.json' file in the 'data' folder. Note that this is not
        a class method. It will be executed before each test method.
        """

        # write status to output stream
        sys.stdout.write('\nSetting up test...')

        # static navigation to top-level package directory
        cls.package_dir = Path(__file__).resolve().parent.parent

        # define directory with data files
        cls.data_dir = os.path.join(cls.package_dir, 'data')

        # define input/output file names
        cls.input_file = os.path.join(cls.data_dir, 'test_input.json')
        cls.output_file = os.path.join(cls.data_dir, 'test_expected.json')

        # read in json file with test input data
        with open(cls.input_file) as in_file:
            cls.input_data = json.load(in_file)

        # read in json file with test expected output data
        with open(cls.output_file) as out_file:
            cls.output_data = json.load(out_file)

        # set up test data
        cls.test_wv = cls.input_data['initial_conditions']['wv']

        # set up expected outputs
        cls.T_in = np.array(cls.output_data['filspec_expected']['T'])
        cls.R_in = np.array(cls.output_data['filspec_expected']['R'])

        # threshold for floating point comparison
        cls.thresh = 14

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_reg_vec_norm(self):
        """
        Test the reg_vec() method with normal inputs. Use filspec_expected to setup
        test input values.
        """

        # update output stream
        sys.stdout.write('\nTesting reg_vec_norm()... ')



        # iterate range of oc values (0-11)
        test_fails = 0
        for exp_oc in range(0, 12):

            # calculation based on 'oc' parameter
            if exp_oc in (0, 6):
                exp_rv = self.T_in - self.R_in
            elif exp_oc in (1, 7):
                exp_rv = (self.T_in - self.R_in) / (self.T_in + self.R_in)
            elif exp_oc in (2, 8):
                exp_rv = self.T_in - .5 * self.R_in
            elif exp_oc in (3, 9):
                exp_rv = 2 * self.T_in - np.ones(np.shape(self.test_wv)[1])
            elif exp_oc in (4, 5, 10, 11):
                exp_rv = self.T_in

            # call reg_vec() method
            test_rv = tff.reg_vec(self.test_wv, exp_oc, self.T_in, self.R_in)

            # assert that the test value equals the expected value
            try:
                nptest.assert_almost_equal(np.array(test_rv).astype(complex),
                    np.array(exp_rv).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: OC = '
                                + str(exp_oc)
                                + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()