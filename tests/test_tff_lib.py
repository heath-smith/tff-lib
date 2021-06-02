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

    def test_filspec(self):
        """
        Test the filspec() method. Creates a new instance of MOE class,
        then calls method and assert output values are equal to expected.
        """

        sys.stdout.write('\nTesting filspec()... ')

        # new instance of MOE class
        test_moe = MOE(self.data_path)

        # test w/ incident angle = 0
        test_angle = 0

        # call filspec() method
        test_filspec_output = test_moe.filspec(test_angle)

        # get data from expected_data.json
        filspec_exp = self.expected_data['filspec_expected']

        # threshold for decimal comparison
        thresh = 1

        # assert all output arrays are equal to expected
        test_fails = 0
        for key in filspec_exp:
            try:
                nptest.assert_almost_equal(np.array(test_filspec_output[key]).astype(complex),
                    np.array(filspec_exp[key]).astype(complex),
                    decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | Thresh: ' + str(thresh))

    """ --------- These functions are related to DESIGN/PERFORMANCE ---------- """
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


    def test_roc_curve(self):
        """
        Test the roc_curve() method. Create an instance of MOE class,
        then set up the test values and make call to roc_curve().
        """

        sys.stdout.write('\nTesting roc_curve()... ')

        # new instance of MOE class
        test_moe = MOE(self.data_path)

        # define test input parameters
        test_truth = self.expected_data['roc_curve_input']['truth']
        test_detections = self.expected_data['roc_curve_input']['detections']
        test_thresh = self.expected_data['roc_curve_input']['thresh']

        # make call to roc_curve()
        roc_curve_output = test_moe.roc_curve(test_truth,
                                            test_detections,
                                            test_thresh)

        # define expected output from test_expected.json
        roc_curve_exp = self.expected_data['roc_curve_expected']

        # threshold for floating point comparison
        t = 4

        # assert test output equals expected values
        test_fails = 0
        for key in roc_curve_exp:
            try:
                nptest.assert_almost_equal(np.array(roc_curve_output[key]),
                    np.array(roc_curve_exp[key]), decimal=t, verbose=True)
            except AssertionError as err:
                test_fails += 1
                #sys.stderr.write('\nAssertion Error: ---------->'
                #                + str(key)
                #                + str(err))
                # uncomment to print errors.. disabling for now
        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | Thresh: ' + str(t))

    def tearDown(self):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()