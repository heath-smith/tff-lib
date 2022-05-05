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

# import function to test
from tff_lib.tff_lib import admit_delta
from tff_lib.utils import film_matrix, convert_to_numpy

class TestAdmitDelta(unittest.TestCase):
    """
    Test class for admit_delta() method.
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

        # -- grab the wavelength values in advance
        # -- incident medium refractive index (assume incident medium is air)
        # -- substrate refractive index
        # -- incident angle theta
        # -- test layers
        # -- define the expected output
        # -- define array comparison threshold
        # -- film refractive indices
        cls.test_wv = np.asarray(cls.test_data['input']['wv'])
        cls.test_med = np.ones(np.shape(cls.test_data['input']['wv'])).astype(np.complex)
        cls.test_sub = np.asarray(cls.test_data['input']['substrate']).astype(np.complex)
        cls.test_theta = 0.0
        cls.test_layers = [tuple(v) for v in cls.test_data['input']['layers']]
        cls.test_exp = convert_to_numpy(cls.test_data['output']['admit_delta'], is_complex=True)
        cls.precision = 13
        cls.test_films = film_matrix(
            cls.test_layers,
            np.asarray(cls.test_data['input']['high_mat']),
            np.asarray(cls.test_data['input']['low_mat']),
        )

    def test_admit_delta_no_units(self):
        """
        ----------> default test case
        """

        # test admit_delta() without 'units'
        _ad = admit_delta(
             self.test_layers, self.test_wv, self.test_sub,
            self.test_med, self.test_films, self.test_theta)

        # verify the output
        nptest.assert_array_almost_equal(
            self.test_exp['ns_inc'], _ad['ns_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_inc'], _ad['np_inc'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_sub'], _ad['ns_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_sub'], _ad['np_sub'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['ns_film'], _ad['ns_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['np_film'], _ad['np_film'], decimal=self.precision)
        nptest.assert_array_almost_equal(
            self.test_exp['delta'], _ad['delta'], decimal=self.precision)


##    def test_admit_delta_deg(self):
##        """
##        Test admit_delta() with 'deg' as units param.
##        """
##
##        #-------- test admit_delta() with units = 'rad' ---------#
##        sys.stdout.write('\nTesting admit_delta_deg()... ')
##        test_ad_units = tff.admit_delta(self.test_wv_range, self.test_layers,
##                                self.test_theta, self.test_i_n, self.test_s_n,
##                                self.test_f_n, units='deg')
##
##        # assert equality in test results vs. expected results
##        # use numpy testing functions for array comparison
##        test_fails = 0
##        for key in self.test_exp:
##            try:
##                nptest.assert_array_almost_equal(test_ad_units[key],
##                    np.array(self.test_exp[key]).astype(complex),
##                    decimal=self.thresh, verbose=True)
##            except AssertionError as err:
##                test_fails += 1
##                sys.stderr.write('\nAssertion Error: ' + key + str(err))
##
##        if test_fails == 0:
##            # write 'PASSED' to output stream if
##            # all assertions pass
##            sys.stdout.write('PASSED')
##
##    def test_admit_delta_wrong_shapes(self):
##        """
##        Test admit_delta() with wrong array shapes.
##        """
##
##        #--- test admit_delta() with wrong shape on i_n, s_n, f_n ----#
##        sys.stdout.write('\nTesting admit_delta_wrong_shapes()... ')
##
##        # delete elements from test_i_n then run admit_delta()
##        wrong_i_n = np.delete(self.test_i_n, [1, 4, 20, 56, 89], axis=1)
##        # assert that a ValueError exception is raised
##        with self.assertRaises(ValueError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, wrong_i_n,
##                            self.test_s_n, self.test_f_n)
##        # delete elements from test_s_n
##        wrong_s_n = np.delete(self.test_s_n, [5, 8, 25, 47, 68], axis=1)
##        # assert that a ValueError exception is raised
##        with self.assertRaises(ValueError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, self.test_i_n,
##                            wrong_s_n, self.test_f_n)
##        # delete elements from test_s_n
##        wrong_f_n = np.delete(self.test_f_n, [1, 3, 5, 8], axis=0)
##        # assert that a ValueError exception is raised
##        with self.assertRaises(ValueError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, self.test_i_n,
##                            self.test_s_n, wrong_f_n)
##
##        # write 'PASSED' to output stream if
##        # all assertions pass
##        sys.stdout.write('PASSED')
##
##    def test_admit_delta_wrong_types(self):
##        """
##        Test admit_delta() with wrong array types.
##        """
##
##        #---- test admit_delta() with wrong array types -----#
##        sys.stdout.write('\nTesting admit_delta_wrong_types()... ')
##        warnings.filterwarnings("ignore") # temporarily disable warnings
##        wrong_type_i_n = self.test_i_n.astype(float)
##        # assert that a TypeError exception is raised
##        with self.assertRaises(TypeError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, wrong_type_i_n,
##                            self.test_s_n, self.test_f_n)
##        wrong_type_s_n = self.test_s_n.astype(float)
##        # assert that a TypeError exception is raised
##        with self.assertRaises(TypeError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, self.test_i_n,
##                            wrong_type_s_n, self.test_f_n)
##        wrong_type_f_n = self.test_s_n.astype(float)
##        # assert that a TypeError exception is raised
##        with self.assertRaises(TypeError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, self.test_i_n,
##                            self.test_s_n, wrong_type_f_n)
##
##        # write 'PASSED' to output stream if
##        # all assertions pass
##        sys.stdout.write('PASSED')
##
##        # reset warnings
##        warnings.filterwarnings("default")
##
##    def test_admit_delta_wrong_args(self):
##        """
##        Test admit_delta() with incorrect number of input args.
##        """
##
##        sys.stdout.write('\nTesting admit_delta_wrong_args()... ')
##        # assert that a ValueError exception is raised
##        # not enough args
##        with self.assertRaises(ValueError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, self.test_f_n)
##        # assert that a ValueError exception is raised
##        # too many args
##        with self.assertRaises(ValueError):
##            tff.admit_delta(self.test_wv_range, self.test_layers,
##                            self.test_theta, 0, 0,
##                            self.test_s_n, self.test_f_n)
##
##        # write 'PASSED' to output stream if
##        # all assertions pass
##        sys.stdout.write('PASSED')
##
    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()