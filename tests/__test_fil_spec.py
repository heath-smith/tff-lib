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

class TestFilSpec(unittest.TestCase):
    """
    Test class for fil_spec() method.
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

        # get data from expected_data.json
        cls.filspec_exp = cls.output_data['filspec_expected']

        #----- Setup test input parameters -------#
        cls.test_cond  = cls.input_data['initial_conditions']
        # test wv_range
        cls.test_wv_range = cls.test_cond['wv']
        # test substrate
        cls.test_substrate = cls.test_cond['substrate']
        # test high material
        cls.test_h_mat = cls.test_cond['high_mat']
        # test low material
        cls.test_l_mat = cls.test_cond['low_mat']
        # test layer stack
        cls.test_layer_stack = cls.test_cond['layers']
        # test materials
        cls.test_materials = cls.test_cond['materials']
        # test substrate thickness
        cls.test_sub_thick = cls.test_cond['sub_thick']
        # test w/ incident angle = 0
        cls.test_angle = 0

        # threshold for floating point comparison
        cls.thresh = 1

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_fil_spec(self):
        """
        Test the filspec() method. Creates a new instance of MOE class,
        then calls method and assert output values are equal to expected.
        """

        sys.stdout.write('\nTesting fil_spec()... ')

        # call filspec() method
        test_filspec = tff.fil_spec(self.test_wv_range, self.test_substrate,
                                    self.test_h_mat, self.test_l_mat,
                                    self.test_layer_stack, self.test_materials,
                                    self.test_angle, self.test_sub_thick, units='rad')

        # assert all output arrays are equal to expected
        test_fails = 0
        for key in self.filspec_exp:
            try:
                nptest.assert_almost_equal(np.array(test_filspec[key]).astype(complex),
                    np.array(self.filspec_exp[key]).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

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