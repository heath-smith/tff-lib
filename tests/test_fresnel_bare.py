#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test class for
the fresnel_bare() method.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests.test_fresnel_bare
>>> python -m unittest tests.test_fresnel_bare.<method>
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

class TestFresnelBare(unittest.TestCase):
    """
    Test case for fresnel_bare() method.
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

       # get spectral info from initial_conditions in json
        cls.test_cond = cls.input_data['initial_conditions']
        # define expected test results
        cls.test_exp = cls.output_data['fresnel_bare_expected']

        # incident medium refractive index (assume incident medium is air)
        cls.test_i_n = np.ones(np.shape(cls.test_cond['wv'])).astype(complex)
        # substrate refractive index
        cls.test_s_n = np.array(cls.test_cond['substrate']).astype(complex)
        # incident angle theta
        cls.test_theta = 0.0

        # set decimal threshold for array comparison
        cls.thresh = 4

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_fresnel_bare_nounits(self):
        """
        Test fresnel_bare() method by initializing test input data and asserting
        the test output arrays are equal to the expected output arrays.

        Comments
        -------------
        Use numpy testing functionality to compare numpy arrays. See FILSPEC lines 447-464
        in moe.py module for definition of the test arrays.
        """

        # update output stream
        sys.stdout.write('\nTesting fresnel_bare_nounits()... ')

        # test fresnel_bare() with no 'units' arg
        test_fb_no_units = tff.fresnel_bare(self.test_i_n,
                                            self.test_s_n,
                                            self.test_theta)

        # assert equality in test results vs. expected results
        # use numpy testing functions for array comparison
        test_fails = 0
        for key in self.test_exp:
            try:
                nptest.assert_array_almost_equal(test_fb_no_units[key],
                    self.test_exp[key], decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def test_fresnel_bare_units(self):

        #--------- test w/ 'units' param ----------#
        sys.stdout.write('\nTesting fresnel_bare_units(units="deg")... ')

        # test fresnel_bare() with units='deg'
        test_fb_units = tff.fresnel_bare(self.test_i_n,
                                        self.test_s_n,
                                        self.test_theta,
                                        units='deg')
        test_fails = 0
        for key in self.test_exp:
            try:
                nptest.assert_array_almost_equal(test_fb_units[key],
                    self.test_exp[key], decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def test_fresnel_bare_wrong_utype(self):
        """
        Test w/ non-string as 'units' param
        """
        sys.stdout.write('\nTesting fresnel_bare_wrong_utype(units="some string")... ')

        # assert that a UnitError exception is raised
        with self.assertRaises(UnitError):
            tff.fresnel_bare(self.test_i_n,
                            self.test_s_n,
                            self.test_theta,
                            units='some string')

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()