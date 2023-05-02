#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the ThinFilmFilter
class.

Usage
---------
>>> python -m unittest -v tests.test_filter
"""

# import external packages
import unittest
from pathlib import Path
import sys
import os
import json
import time
import numpy as np
import numpy.testing as nptest
from tff_lib import ThinFilm, OpticalMedium, FilmStack

# class under test
from tff_lib import ThinFilmFilter

class TestThinFilmFilter(unittest.TestCase):
    """
    Test suite for ThinFilmFilter() class.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment.
        """

        # static navigation to data directory and output directory
        cls.dir = os.path.join(Path(__file__).resolve().parent.parent, r'data')
        cls.data_file = os.path.join(cls.dir, 'test_data.json')

        # read in json file with test input data
        with open(cls.data_file, 'r', encoding='utf=8') as dat:
            cls.test_data = json.load(dat)

        # set decimal precision
        cls._precision = 12

        # setup input data from test_expected.json
        cls._theta = 0.0
        cls._sub_thickness = cls.test_data['input']['sub_thick'] * 10**6
        cls._wavelengths = cls.test_data['input']['wv']
        cls._high_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low_mat = [complex(x) for x in cls.test_data['input']['low_mat']]
        cls._layers = cls.test_data['input']['layers']
        cls._sub_ref_index = [complex(x) for x in cls.test_data['input']['substrate']]
        cls._fresnel = cls.test_data['output']['fresnel_film']
        cls._filspec = cls.test_data['output']['filspec']

        # generate test ThinFilmFilter
        cls._films = [
            ThinFilm(
                cls._wavelengths,
                cls._high_mat if lyr[0] == 1 else cls._low_mat,
                thick=lyr[1],
                ntype=lyr[0]
            )
            for lyr in cls._layers
        ]

        # test OpticalMedium (air)
        cls._inc = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']])

        # test substrate
        cls._sub = OpticalMedium(
            cls._wavelengths, cls._sub_ref_index, thick=cls._sub_thickness)

        # test FilmStack
        cls._stack = FilmStack(cls._films)

        # use complex conjugates for fresnel coefficients
        cls._fresnel_conj = {
            k: np.conjugate(np.asarray(v).astype(np.complex128))
            for k,v in cls._fresnel.items()
        }

    def test_filter_init_(self):
        """
        test __init__()
        """

        tff = ThinFilmFilter(self._sub,
                             self._stack,
                             self._inc)

        self.assertEqual(self._sub, tff.sub)
        self.assertEqual(self._stack, tff.stack)
        self.assertEqual(self._inc, tff.inc)

    def test_fresnel_coeffs(self):
        """
        test fresnel_coeffs()
        """

        tff = ThinFilmFilter(self._sub,
                             self._stack,
                             self._inc)

        # test with 'medium' reflection
        fresnel = tff.fresnel_coeffs(self._theta, 'medium')

        nptest.assert_array_almost_equal(
            self._fresnel_conj['Ts'], fresnel['Ts'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel_conj['Tp'], fresnel['Tp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel_conj['Rs'], fresnel['Rs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel_conj['Rp'], fresnel['Rp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel_conj['rs'], fresnel['rs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel_conj['rp'], fresnel['rp'], decimal=self._precision)

    def test_filter_spectrum(self):
        """
        test filter_spectrum()
        """

        tff = ThinFilmFilter(self._sub,
                             self._stack,
                             self._inc)

        filspec = tff.filter_spectrum(self._theta)

        nptest.assert_almost_equal(
            self._filspec['T'], filspec['T'], decimal=self._precision, verbose=True)
        nptest.assert_almost_equal(
            self._filspec['Ts'], filspec['Ts'], decimal=self._precision, verbose=True)
        nptest.assert_almost_equal(
            self._filspec['Tp'], filspec['Tp'], decimal=self._precision, verbose=True)
        nptest.assert_almost_equal(
            self._filspec['R'], filspec['R'], decimal=self._precision, verbose=True)
        nptest.assert_almost_equal(
            self._filspec['Rs'], filspec['Rs'], decimal=self._precision, verbose=True)
        nptest.assert_almost_equal(
            self._filspec['Rp'], filspec['Rp'], decimal=self._precision, verbose=True)


    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
