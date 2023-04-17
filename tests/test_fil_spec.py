#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suite for fil_spec()

Examples
---------
>>> python -m unittest -v tests.test_fil_spec
"""

# import external packages
import numpy as np
from numpy import testing as nptest
import unittest
from pathlib import Path
import sys
import os
import json
import matplotlib.pyplot as plt
import time

# import function to test
from tff_lib.utils import fil_spec
from tff_lib.utils import film_matrix, convert_to_numpy, plot_data

class TestFilSpec(unittest.TestCase):
    """
    Test class for fil_spec() method.
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
        cls.test_med = np.ones(np.shape(cls.test_data['input']['wv'])).astype(np.complex128)
        cls.test_sub = np.asarray(cls.test_data['input']['substrate']).astype(np.complex128)
        cls.test_theta = 0.0
        cls.test_layers = [tuple(v) for v in cls.test_data['input']['layers']]
        cls.test_exp = convert_to_numpy(cls.test_data['output']['admit_delta'], is_complex=True)
        cls.test_sub_thick = float(cls.test_data['input']['sub_thick'])
        cls.test_films = film_matrix(
            cls.test_layers,
            np.asarray(cls.test_data['input']['high_mat']),
            np.asarray(cls.test_data['input']['low_mat']),
        )

        # threshold for floating point comparison
        cls.precision = 12

        # get expected results
        cls.test_exp = convert_to_numpy(cls.test_data['output']['filspec'])


    def test_fil_spec(self):
        """
        ---------> default test case
        """

        # time the execution
        start = time.perf_counter()

        # test fil_spec()
        _filspec = fil_spec(
            self.test_wv, self.test_sub, self.test_med, self.test_films,
            self.test_layers, self.test_sub_thick, self.test_theta)

        # end time
        end = time.perf_counter()
        t = (end - start)
        sys.stdout.write(f"\nElapsed Time= {round(t, 4)} seconds.\n")

        # verify the output
        nptest.assert_almost_equal(
            self.test_exp['T'], _filspec['T'], decimal=self.precision, verbose=True)
        nptest.assert_almost_equal(
            self.test_exp['Ts'], _filspec['Ts'], decimal=self.precision, verbose=True)
        nptest.assert_almost_equal(
            self.test_exp['Tp'], _filspec['Tp'], decimal=self.precision, verbose=True)
        nptest.assert_almost_equal(
            self.test_exp['R'], _filspec['R'], decimal=self.precision, verbose=True)
        nptest.assert_almost_equal(
            self.test_exp['Rs'], _filspec['Rs'], decimal=self.precision, verbose=True)
        nptest.assert_almost_equal(
            self.test_exp['Rp'], _filspec['Rp'], decimal=self.precision, verbose=True)

        # plot the output
        ## plot_var = 'T'
        ## plt.figure(figsize=(12, 6))
        ## plt.plot(self.test_wv, self.test_exp[plot_var], label="Expected")
        ## plt.plot(self.test_wv, _filspec[plot_var], label="Calculated")
        ## plt.title("Test Results")
        ## plt.xlabel("Wavelength (nm)")
        ## plt.ylabel("Values")
        ## plt.legend(loc="lower right")
        ## plt.tight_layout()
        ## plt.show()

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

        sys.stdout.write('\nRunning teardown procedure...\nDONE!')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()