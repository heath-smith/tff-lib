#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test class for the
admit_delta() method.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests.test_admit_delta
>>> python -m unittest tests.test_admit_delta.<method>
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

class TestAdmitDelta(unittest.TestCase):
    """
    Test class for admit_delta() method.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment. All tests here will use data from the
        'test_input.json' file in the 'data' folder. Note that this is not
        a class method. It will be executed before each test method.
        """

        # write status to output stream
        sys.stdout.write('\nSetting up test class...')

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

        #---------- DEFINE TEST INPUT DATA -----------#
        # get spectral info from initial_conditions in json
        cls.test_cond = cls.input_data['initial_conditions']
        # define expected test results
        cls.test_exp = cls.output_data['admit_delta_expected']
        # test wavelength range
        cls.test_wv_range = cls.test_cond['wv']
        # input layer stack
        cls.test_layers = cls.test_cond['layers']
        # incident angle theta
        cls.test_theta = 0.0
        # incident medium refractive index (assume incident medium is air)
        cls.test_i_n = np.ones(np.shape(cls.test_cond['wv'])).astype(complex)
        # substrate refractive index
        cls.test_s_n = np.array(cls.test_cond['substrate']).astype(complex)
        # film refractive indices
        cls.test_f_n = np.zeros((len(cls.test_cond['layers']),
                np.shape(cls.test_cond['wv'])[1])).astype(complex)
        # get measured substrate & thin film optical constant data
        for i in range(0, len(cls.test_cond['materials'])):
            if cls.test_cond['materials'][i] == "H":
                cls.test_f_n[i, :] = np.array(cls.test_cond['high_mat'])
            else:
                cls.test_f_n[i, :] = np.array(cls.test_cond['low_mat'])

        # set thresholds for decimal comparison
        cls.thresh = 12

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_admit_delta_nounits(self):
        """
        Test the admit_delta() method with no 'units' arg.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays.
        See FILSPEC lines 447-464 in moe.py module for definition
        of the test arrays.
        """

        # update output stream
        sys.stdout.write('\nTesting admit_delta_nounits()... ')

        #-------- test admit_delta() with no 'units' arg ---------#
        test_ad_no_units = tff.admit_delta(self.test_wv_range, self.test_layers,
                                            self.test_theta, self.test_i_n,
                                            self.test_s_n, self.test_f_n)

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in self.test_exp:
            try:
                nptest.assert_array_almost_equal(test_ad_no_units[key],
                    np.array(self.test_exp[key]).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def test_admit_delta_deg(self):
        """
        Test admit_delta() with 'deg' as units param.
        """

        #-------- test admit_delta() with units = 'rad' ---------#
        sys.stdout.write('\nTesting admit_delta_deg()... ')
        test_ad_units = tff.admit_delta(self.test_wv_range, self.test_layers,
                                self.test_theta, self.test_i_n, self.test_s_n,
                                self.test_f_n, units='deg')

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in self.test_exp:
            try:
                nptest.assert_array_almost_equal(test_ad_units[key],
                    np.array(self.test_exp[key]).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def test_admit_delta_wrong_shapes(self):
        """
        Test admit_delta() with wrong array shapes.
        """

        #--- test admit_delta() with wrong shape on i_n, s_n, f_n ----#
        sys.stdout.write('\nTesting admit_delta_wrong_shapes()... ')

        # delete elements from test_i_n then run admit_delta()
        wrong_i_n = np.delete(self.test_i_n, [1, 4, 20, 56, 89], axis=1)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, wrong_i_n,
                            self.test_s_n, self.test_f_n)
        # delete elements from test_s_n
        wrong_s_n = np.delete(self.test_s_n, [5, 8, 25, 47, 68], axis=1)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, self.test_i_n,
                            wrong_s_n, self.test_f_n)
        # delete elements from test_s_n
        wrong_f_n = np.delete(self.test_f_n, [1, 3, 5, 8], axis=0)
        # assert that a ValueError exception is raised
        with self.assertRaises(ValueError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, self.test_i_n,
                            self.test_s_n, wrong_f_n)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    def test_admit_delta_wrong_types(self):
        """
        Test admit_delta() with wrong array types.
        """

        #---- test admit_delta() with wrong array types -----#
        sys.stdout.write('\nTesting admit_delta_wrong_types()... ')
        warnings.filterwarnings("ignore") # temporarily disable warnings
        wrong_type_i_n = self.test_i_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, wrong_type_i_n,
                            self.test_s_n, self.test_f_n)
        wrong_type_s_n = self.test_s_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, self.test_i_n,
                            wrong_type_s_n, self.test_f_n)
        wrong_type_f_n = self.test_s_n.astype(float)
        # assert that a TypeError exception is raised
        with self.assertRaises(TypeError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, self.test_i_n,
                            self.test_s_n, wrong_type_f_n)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

        # reset warnings
        warnings.filterwarnings("default")

    def test_admit_delta_wrong_args(self):
        """
        Test admit_delta() with incorrect number of input args.
        """

        sys.stdout.write('\nTesting admit_delta_wrong_args()... ')
        # assert that a ValueError exception is raised
        # not enough args
        with self.assertRaises(ValueError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, self.test_f_n)
        # assert that a ValueError exception is raised
        # too many args
        with self.assertRaises(ValueError):
            tff.admit_delta(self.test_wv_range, self.test_layers,
                            self.test_theta, 0, 0,
                            self.test_s_n, self.test_f_n)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()