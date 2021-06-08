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

class TestRocCurve(unittest.TestCase):
    """
    Test class for roc_curve() method from Thin Film Filter
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

        # define test input parameters
        cls.test_truth = np.array(cls.output_data['roc_curve_input']['truth'])
        cls.test_detections = np.array(cls.output_data['roc_curve_input']['detections'])
        cls.test_thresh = np.array(cls.output_data['roc_curve_input']['thresh'])

       # define expected output from test_expected.json
        cls.roc_curve_exp = cls.output_data['roc_curve_expected']

        # threshold for floating point comparison
        cls.thresh = 12

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_roc_curve_norm(self):
        """
        Test the roc_curve() method with normal inputs. Create an instance of MOE class,
        then set up the test values and make call to roc_curve().
        """

        # update output stream
        sys.stdout.write('\nTesting roc_curve_norm()... ')

        # make call to roc_curve()
        test_roc_curve = tff.roc_curve(self.test_truth,
                                        self.test_detections,
                                        self.test_thresh)

        # assert test output equals expected values
        test_fails = 0
        for key in self.roc_curve_exp:
            try:
                nptest.assert_almost_equal(
                    np.array(test_roc_curve[key]),
                    np.array(self.roc_curve_exp[key]),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: '
                                + str(key)
                                + str(err))
                # add 'return' to exit loop on first failure
                # helpful for dealing with one assertion
                # error at a time!
                #return

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