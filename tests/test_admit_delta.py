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

    def test_admit_delta(self):
        """
        Test the admit_delta() method.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays.
        See FILSPEC lines 447-464 in moe.py module for definition
        of the test arrays.
        """

        # update output stream
        sys.stdout.write('\nTesting admit_delta()... ')

        #---------- DEFINE TEST INPUT DATA -----------#
        # get spectral info from initial_conditions in json
        test_cond = self.input_data['initial_conditions']
        # define expected test results
        test_exp = self.output_data['admit_delta_expected']
        # test wavelength range
        test_wv_range = test_cond['wv']
        # input layer stack
        test_layers = test_cond['layers']
        # incident angle theta
        test_theta = 0.0
        # incident medium refractive index (assume incident medium is air)
        test_i_n = np.ones(np.shape(test_cond['wv'])).astype(complex)
        # substrate refractive index
        test_s_n = np.array(test_cond['substrate']).astype(complex)
        # film refractive indices
        test_f_n = np.zeros((len(test_cond['layers']),
                np.shape(test_cond['wv'])[1])).astype(complex)
        # get measured substrate & thin film optical constant data
        for i in range(0, len(test_cond['materials'])):
            if test_cond['materials'][i] == "H":
                test_f_n[i, :] = np.array(test_cond['high_mat'])
            else:
                test_f_n[i, :] = np.array(test_cond['low_mat'])

        # set thresholds for decimal comparison
        r_tol = 1e-14
        a_tol = .1

        #-------- test admit_delta() with no 'units' arg ---------#
        sys.stdout.write('\n\t* no units param... ')
        test_ad_no_units = tff.admit_delta(test_wv_range, test_layers, test_theta,
                                            test_i_n, test_s_n, test_f_n)

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in test_exp:
            try:
                nptest.assert_allclose(test_ad_no_units[key],
                    np.array(test_exp[key]).astype(complex),
                    rtol=r_tol, atol=a_tol)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #-------- test admit_delta() with units = 'rad' ---------#
        sys.stdout.write('\n\t* units="rad"... ')
        test_ad_units = tff.admit_delta(test_wv_range, test_layers,
                                    test_theta, test_i_n, test_s_n,
                                    test_f_n, units='rad')

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        for key in test_exp:
            try:
                nptest.assert_allclose(test_ad_units[key],
                    np.array(test_exp[key]).astype(complex),
                    rtol=r_tol, atol=a_tol)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #--- test admit_delta() with wrong shape on i_n, s_n, f_n ----#
        sys.stdout.write('\n\t* wrong array shapes... ')
        # delete elements from test_i_n then run admit_delta()
        wrong_i_n = np.delete(test_i_n, [1, 4, 20, 56, 89], axis=1)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(test_wv_range, test_layers,
                test_theta, wrong_i_n, test_s_n, test_f_n)
        # delete elements from test_s_n
        wrong_s_n = np.delete(test_s_n, [5, 8, 25, 47, 68], axis=1)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(test_wv_range, test_layers,
                test_theta, test_i_n, wrong_s_n, test_f_n)
        # delete elements from test_s_n
        wrong_f_n = np.delete(test_f_n, [1, 3, 5, 8], axis=0)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(test_wv_range, test_layers,
                    test_theta,test_i_n, test_s_n, wrong_f_n)

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        #---- test admit_delta() with wrong array types -----#
        sys.stdout.write('\n\t* wrong array types... ')
        warnings.filterwarnings("ignore") # temporarily disable warnings
        wrong_type_i_n = test_i_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(test_wv_range, test_layers,
                test_theta, wrong_type_i_n, test_s_n, test_f_n)
        wrong_type_s_n = test_s_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(test_wv_range, test_layers,
                test_theta, test_i_n, wrong_type_s_n, test_f_n)
        wrong_type_f_n = test_s_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(test_wv_range, test_layers,
                test_theta, test_i_n, test_s_n, wrong_type_f_n)

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

        # reset warnings
        warnings.filterwarnings("default")

    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()