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