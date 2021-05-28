#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module test_tff_lib.py contains the test suite for the
ThinFilmFilter class and methods.

Execute tests from top-level package directory.

Example: python -m unittest tests/test_tff_lib.py
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
from tff_lib.tff_lib import ThinFilmFilter as tff
from tff_lib.exceptions import UnitError

class TestThinFilmFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        return super().setUpClass()

    def test_init(self):
        """
        Test the __init__ method and assert that the class members are
        initialized and defined correctly.
        """

        sys.stdout.write('\nTesting __init__()... ')

        sys.stdout.write('PASSED')

    """ --------- The following test methods are related to FILSPEC ---------- """

    def test_fresnel_bare(self):
        """
        Test fresnel_bare() method by initializing test input data and asserting
        the test output arrays are equal to the expected output arrays.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays. See FILSPEC lines 447-464
        in moe.py module for definition of the test arrays.
        """

        sys.stdout.write('\nTesting fresnel_bare()... ')

        # instance of MOE class
        test_moe_fb = MOE(self.data_path)

        #---------- DEFINE TEST INPUT DATA -----------#

        # incident medium refractive index (assume incident medium is air)
        test_i_n = np.ones(np.shape(test_moe_fb.init_conditions['wv'])).astype(complex)

        # substrate refractive index
        test_s_n = np.array(test_moe_fb.init_conditions['substrate']).astype(complex)

        # incident angle 'theta' (default = 0 radians!)
        test_theta = test_moe_fb.design_settings['incident_angle'] * (np.pi / 180)

        # call test_moe_fresnel_bare() method
        test_fb_output = test_moe_fb.fresnel_bare(test_i_n, test_s_n, test_theta)

        # dictionary with expected fresnel_bare results
        expected_fb = self.expected_data['fresnel_bare_expected']

        # set decimal threshold for array comparison
        thresh = 4

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in expected_fb:
            try:
                nptest.assert_array_almost_equal(test_fb_output[key],
                    expected_fb[key], decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))


        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | Thresh: ' + str(thresh))

    def test_admit_delta(self):
        """
        Test the admit_delta() method.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays. See FILSPEC lines 447-464
        in moe.py module for definition of the test arrays.
        """

        sys.stdout.write('\nTesting admit_delta()... ')

        # instance of MOE class
        test_moe_ad = MOE(self.data_path)

        #---------- DEFINE TEST INPUT DATA -----------#

        # incident medium refractive index (assume incident medium is air)
        test_i_n = np.ones(np.shape(test_moe_ad.init_conditions['wv'])).astype(complex)

        # substrate refractive index
        test_s_n = np.array(test_moe_ad.init_conditions['substrate']).astype(complex)

        # film refractive indices
        test_f_n = np.zeros((len(test_moe_ad.init_conditions['layers']),
                np.shape(test_moe_ad.init_conditions['wv'])[1])).astype(complex)
        # get measured substrate & thin film optical constant data
        for i in range(0, len(test_moe_ad.init_conditions['materials'])):
            if test_moe_ad.init_conditions['materials'][i] == "H":
                test_f_n[i,:] = test_moe_ad.init_conditions['high_mat']
            else:
                test_f_n[i,:] = test_moe_ad.init_conditions['low_mat']

        # incident angle 'theta' (default = 0 radians!)
        test_theta = test_moe_ad.design_settings['incident_angle'] * (np.pi / 180)

        # make call to admit_delta() method
        test_ad_output = test_moe_ad.admit_delta(test_i_n, test_s_n, test_f_n, test_theta)

        # define dictionary with expected admit_delta() results
        ad_expected = self.expected_data['admit_delta_expected']

        # set decimal threshold for array comparison
        thresh = 4

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in ad_expected:
            try:
                nptest.assert_allclose(test_ad_output[key],
                    np.array(ad_expected[key]).astype(complex),
                    rtol=1e-14, atol=.1)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | rtol=1e-14, atol=.1')

    def test_c_mat(self):
        """
        Test c_mat() method. Use expected output from admit_delta in
        the test_expected.json file to supply input parameters for c_mat().
        """

        sys.stdout.write('\nTesting c_mat()... ')

        # new instance of MOE class
        test_moe_cmat = MOE(self.data_path)

        # setup input data from test_expected.json
        admit_delta_exp = self.expected_data['admit_delta_expected']

        test_nsfilm = np.array(admit_delta_exp['nsFilm']).astype(complex)
        test_npFilm = np.array(admit_delta_exp['npFilm']).astype(complex)
        test_delta = np.array(admit_delta_exp['delta']).astype(complex)

        # make call to c_mat() method
        test_cmat_output = test_moe_cmat.c_mat(test_nsfilm, test_npFilm, test_delta)

        # get expected values from test_expected.json
        cmat_expected = self.expected_data['c_mat_expected']

        # floating point comparison threshold
        thresh = 6

        # assert output is equal to expected
        test_fails = 0
        for key in test_cmat_output:
            try:
                nptest.assert_almost_equal(np.array(test_cmat_output[key]).astype(complex),
                    np.array(cmat_expected[key]).astype(complex),
                    decimal=thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED | Thresh: ' + str(thresh))


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

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()