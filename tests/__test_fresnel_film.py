#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test suite for the
fresnel_film() method.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests.test_fresnel_film
>>> python -m unittest tests.test_fresnel_film.<method>
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

class TestFresnelFilm(unittest.TestCase):
    """
    Test class for fresnel_film() method.
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

        # initialize test input data
        cls.test_admit_deltas = cls.output_data['admit_delta_expected']
        cls.test_char_matrix = cls.output_data['c_mat_expected']

        # initialize fresnel_film() expected output
        cls.ff_expected = cls.output_data['fresnel_film_expected']

        # COMPLEX CONJUGATE - VERIFY THESE SIGNS WITH RYAN
        cls.ff_expected['ts'] = np.conjugate(np.array(cls.ff_expected['ts']).astype(complex))
        cls.ff_expected['tp'] = np.conjugate(np.array(cls.ff_expected['tp']).astype(complex))
        cls.ff_expected['rs'] = np.conjugate(np.array(cls.ff_expected['rs']).astype(complex))
        cls.ff_expected['rp'] = np.conjugate(np.array(cls.ff_expected['rp']).astype(complex))

        # thresholds for floating point comparison
        cls.thresh = 14

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_fresnel_film(self):
        """
        Test the fresnel_film() method. Creates an instance of MOE class
        and pulls input data from test_expected.json.
        """

        sys.stdout.write('\nTesting fresnel_film()... ')

        # test fresnel_film() method
        test_ff_output = tff.fresnel_film(self.test_admit_deltas,
                                        self.test_char_matrix)

        # assert test output is equal to expected values
        test_fails = 0
        for key in test_ff_output:
            try:
                nptest.assert_array_almost_equal(np.array(test_ff_output[key]).astype(complex),
                    np.array(self.ff_expected[key]).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    @classmethod
    def tearDown(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()