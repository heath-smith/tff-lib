#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the ThinFilm
class.

Usage
---------
>>> python -m unittest -v tests.test_thinfilm
"""

# import external packages
import unittest
from pathlib import Path
import sys
import os
import json

# import class to test
from tff_lib import ThinFilm

class TestThinFilm(unittest.TestCase):
    """
    Test suite for ThinFilm() class.
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

        # setup input data from test_expected.json
        cls._material = 'H'
        cls._thickness = 0.5
        cls._wavelengths = cls.test_data['input']['wv']
        cls._ref_index = cls.test_data['input']['high_mat']

    def test_thin_film_init(self):
        """
        test __init__()
        """

        test_tf = ThinFilm(self._wavelengths,
                           self._ref_index,
                           self._thickness,
                           self._material)

        # assert wavelength and ref_index have equal length
        self.assertEqual(len(test_tf.wavelengths), len(test_tf.ref_index))
        # assert thickness value is valid
        self.assertEqual(test_tf.thickness, self._thickness)

    def test_thin_film_invalid_inputs(self):
        """
        test __init__() with invalid input values
        """

        with self.assertRaises(ValueError):
            # test with negative thickness value
            ThinFilm(self._wavelengths,
                     self._ref_index,
                     -1,
                     self._material)

        with self.assertRaises(ValueError):
            # take a slice of wavelengths
            ThinFilm(self._wavelengths[:-5],
                     self._ref_index,
                     self._thickness,
                     self._material)

        with self.assertRaises(ValueError):
            # use invalid material
            ThinFilm(self._wavelengths,
                     self._ref_index,
                     self._thickness,
                     'wrong material string')

    def test_thin_film_add_subtract(self):
        """
        test operators +/-
        """

        tf1 = ThinFilm(self._wavelengths,
                       self._ref_index,
                       self._thickness,
                       self._material)


        tf2 = ThinFilm(self._wavelengths,
                       self._ref_index,
                       1.0,
                       self._material)

        tf3 = tf1 + tf2
        tf4 = tf2 - tf1

        self.assertEqual(tf3.thickness, 1.5)
        self.assertEqual(tf4.thickness, 0.5)

    def test_thin_film_split_layer(self):
        """
        test split_layer()
        """

        tf1 = ThinFilm(self._wavelengths,
                       self._ref_index,
                       self._thickness,
                       self._material)

        tf2 = tf1.split_film()

        self.assertEqual(tf1.thickness, self._thickness * 0.5)
        self.assertEqual(tf2.thickness, self._thickness * 0.5)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
