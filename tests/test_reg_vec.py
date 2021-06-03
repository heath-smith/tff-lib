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

    def test_reg_vec(self):
        """
        Test the reg_vec() method. Use filspec_expected to setup
        test input values.
        """

        sys.stdout.write('\nTesting reg_vec()... ')

        # new instance of MOE class
        test_moe = MOE(self.data_path)

        # set up expected outputs
        T_in = np.array(self.expected_data['filspec_expected']['T'])
        R_in = np.array(self.expected_data['filspec_expected']['R'])

        # threshold for floating point comparison
        thresh = 12

        # store the default 'opt_comp' setting
        def_oc = test_moe.design_settings['opt_comp']

        # iterate range of oc values (0-11)
        test_fails = 0
        for exp_oc in range(0, 12):

            # update design_settings['opt_comp']
            test_moe.design_settings['opt_comp'] = exp_oc

            # calculation based on 'oc' parameter
            if exp_oc == 0:
                exp_rv = T_in - R_in
            elif exp_oc == 1:
                exp_rv = (T_in - R_in) / (T_in + R_in)
            elif exp_oc == 2:
                exp_rv = T_in - .5 * R_in
            elif exp_oc == 3:
                exp_rv = 2 * T_in - np.ones(np.shape(test_moe.init_conditions['wv'])[1])
            elif exp_oc == 4:
                exp_rv = T_in
            elif exp_oc == 5:
                exp_rv = T_in
            elif exp_oc == 6:
                exp_rv = T_in - R_in
            elif exp_oc == 7:
                exp_rv = (T_in - R_in) / (T_in + R_in)
            elif exp_oc == 8:
                exp_rv = T_in - .5 * R_in
            elif exp_oc == 9:
                exp_rv = 2 * T_in - np.ones(np.shape(test_moe.init_conditions['wv'])[1])
            elif exp_oc == 10:
                exp_rv = T_in
            elif exp_oc == 11:
                exp_rv = T_in

            # call reg_vec() method
            test_rv = test_moe.reg_vec(T_in, R_in)

            # assert that the test value equals the expected value
            try:
                nptest.assert_almost_equal(np.array(test_rv).astype(complex),
                    np.array(exp_rv).astype(complex),
                    decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: OC = '
                                + str(exp_oc)
                                + str(err))

        # reset 'opt_comp' to default value
        test_moe.design_settings['opt_comp'] = def_oc

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | Thresh: ' + str(thresh))


    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()