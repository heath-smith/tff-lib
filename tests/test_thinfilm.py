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
        cls._ntype = 1  # 1 = high index
        cls._thick = 0.5
        cls._waves = cls.test_data['input']['wv']
        cls._nref = cls.test_data['input']['high_mat']

    def test_thin_film_init(self):
        """
        test __init__()
        """

        tf = ThinFilm(self._waves,
                      self._nref,
                      thick=self._thick,
                      ntype=self._ntype)

        # assert wavelength and nref have equal length
        self.assertEqual(len(tf.waves), len(tf.nref))
        # assert thickness value is valid
        self.assertEqual(tf.thick, self._thick)

    def test_thin_film_invalid_inputs(self):
        """
        test __init__() with invalid input values
        """

        with self.assertRaises(ValueError):
            # test with negative thickness value
            ThinFilm(self._waves,
                     self._nref,
                     thick=-10,
                     ntype=self._ntype)

        with self.assertRaises(ValueError):
            # take a slice of waves
            ThinFilm(self._waves[:-5],
                     self._nref,
                     thick=self._thick,
                     ntype=self._ntype)


    def test_thin_film_add_subtract(self):
        """
        test operators +/-
        """

        tf1 = ThinFilm(self._waves,
                       self._nref,
                       thick=self._thick,
                       ntype=self._ntype)


        tf2 = ThinFilm(self._waves,
                       self._nref,
                       thick=1.0,
                       ntype=self._ntype)

        tf3 = tf1 + tf2
        tf4 = tf2 - tf1

        self.assertEqual(tf3.thick, 1.5)
        self.assertEqual(tf4.thick, 0.5)

    def test_thin_film_split_layer(self):
        """
        test split_layer()
        """

        tf1 = ThinFilm(self._waves,
                       self._nref,
                       thick=self._thick,
                       ntype=self._ntype)

        tf2 = tf1.split_film()

        self.assertEqual(tf1.thick, self._thick * 0.5)
        self.assertEqual(tf2.thick, self._thick * 0.5)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
