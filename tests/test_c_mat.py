#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module contains the test class for the
c_mat() method.

Execute tests from top-level package directory.

Examples
---------
>>> python -m unittest tests.test_c_mat
>>> python -m unittest tests.test_c_mat.<method>
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

class TestCMat(unittest.TestCase):
    """
    Test class for c_mat() method.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment. All tests will use data from the
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

        # setup input data from test_expected.json
        cls.cmat_input = cls.output_data['admit_delta_expected']
        cls.test_nsfilm = np.array(cls.cmat_input['ns_film'])
        cls.test_npfilm = np.array(cls.cmat_input['np_film'])
        cls.test_delta = np.array(cls.cmat_input['delta'])
        # get expected values from test_expected.json
        cls.cmat_expected = cls.output_data['c_mat_expected']

        # floating point comparison threshold
        cls.thresh = 12

        # update output stream
        sys.stdout.write('SUCCESS')

    def test_c_mat(self):
        """
        Test c_mat() method. Use expected output from admit_delta in
        the test_expected.json file to supply input parameters for c_mat().
        """

        sys.stdout.write('\nTesting c_mat()... ')

        #------------ test c_mat() -----------#
        test_cmat = tff.c_mat(self.test_nsfilm,
                            self.test_npfilm,
                            self.test_delta)

        # assert output is equal to expected
        test_fails = 0
        for key in test_cmat:
            try:
                nptest.assert_almost_equal(np.array(test_cmat[key]).astype(complex),
                    np.array(self.cmat_expected[key]).astype(complex),
                    decimal=self.thresh, verbose=True)
            except AssertionError as err:
                test_fails += 1
                sys.stderr.write('\nAssertion Error: ' + key + str(err))

        if test_fails == 0:
            # write 'PASSED' to output stream if
            # all assertions pass
            sys.stdout.write('PASSED')

    def test_c_mat_wrong_types(self):
        """
        Test c_mat() with wrong data types.
        """

        #---------- test c_mat() with incorrect data types ------------#
        sys.stdout.write('\nTesting c_mat_wrong_types()... ')
        nsfilm_wrong_type = np.array(self.test_nsfilm).astype(complex)
        npfilm_wrong_type = np.array(self.test_npfilm).astype(complex)
        delta_wrong_type = np.array(self.test_delta).astype(complex)

        # assert that a TypeError exception is raised
        # wrong type ns_film
        with self.assertRaises(TypeError):
            tff.c_mat(nsfilm_wrong_type,
                    self.test_npfilm,
                    self.test_delta)
        # assert that a TypeError exception is raised
        # wrong type np_film
        with self.assertRaises(TypeError):
            tff.c_mat(self.test_nsfilm,
                    npfilm_wrong_type,
                    self.test_delta)
        # assert that a TypeError exception is raised
        # wrong type delta
        with self.assertRaises(TypeError):
            tff.c_mat(self.test_nsfilm,
                    self.test_npfilm,
                    delta_wrong_type)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    def test_c_mat_wrong_struct(self):
        """
        Test c_mat() using lists instead of numpy.ndarray's.
        """

        #---------- test c_mat() with incorrect data structures ------------#
        sys.stdout.write('\nTesting c_mat_wrong_struct()... ')
        nsfilm_wrong_type = self.test_nsfilm.tolist()
        npfilm_wrong_type = self.test_npfilm.tolist()
        delta_wrong_type = self.test_delta.tolist()

        # assert that a TypeError exception is raised
        # wrong type ns_film
        with self.assertRaises(TypeError):
            tff.c_mat(nsfilm_wrong_type,
                    self.test_npfilm,
                    self.test_delta)
        # assert that a TypeError exception is raised
        # wrong type np_film
        with self.assertRaises(TypeError):
            tff.c_mat(self.test_nsfilm,
                    npfilm_wrong_type,
                    self.test_delta)
        # assert that a TypeError exception is raised
        # wrong type delta
        with self.assertRaises(TypeError):
            tff.c_mat(self.test_nsfilm,
                    self.test_npfilm,
                    delta_wrong_type)

        # write 'PASSED' to output stream if
        # all assertions pass
        sys.stdout.write('PASSED')

    #-----------------------------------------------------#
    #------- NEED TO TEST INCORRECT ARRAY SHAPES ---------#
    #-----------------------------------------------------#

    @classmethod
    def tearDownClass(cls):

        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')

        sys.stdout.close()

if __name__=='__main__':
    unittest.main()