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
from tff_lib.films import ThinFilm

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

        # read in json file with test input data
        with open(os.path.join(cls.dir, 'test_data.json')) as dat:
            cls.test_data = json.load(dat)

        # setup input data from test_expected.json
        cls.test_material = 'H'
        cls.test_thickness = 0.5
        cls.test_wavelengths = cls.test_data['input']['wv']
        cls.test_ref_index = cls.test_data['input']['high_mat']

    def test_thin_film_init(self):
        """
        test __init__()
        """

        test_tf = ThinFilm(
            self.test_material,
            self.test_thickness,
            self.test_wavelengths,
            self.test_ref_index)

        # assert values in ref_index are complex typed
        self.assertEqual(complex, type(test_tf.ref_index[0]))
        # assert wavelength and ref_index have equal length
        self.assertEqual(len(test_tf.wavelengths), len(test_tf.ref_index))
        # assert thickness value is valid
        self.assertEqual(test_tf.thickness, self.test_thickness)

    def test_thin_film_invalid_inputs(self):
        """
        test __init__() with invalid input values
        """

        with self.assertRaises(ValueError):
            # test with negative thickness value
            ThinFilm(
                self.test_material,
                -1,
                self.test_wavelengths,
                self.test_ref_index)

        with self.assertRaises(ValueError):
            # take a slice of wavelengths
            ThinFilm(
                self.test_material,
                self.test_thickness,
                self.test_wavelengths[:-5],
                self.test_ref_index)

        with self.assertRaises(ValueError):
            # use invalid material
            ThinFilm(
                'wrong material string',
                self.test_thickness,
                self.test_wavelengths,
                self.test_ref_index)


    def test_thin_film_add_subtract(self):
        """
        test operators +/-
        """

        t1 = ThinFilm(
            self.test_material,
            self.test_thickness,
            self.test_wavelengths,
            self.test_ref_index)


        t2 = ThinFilm(
            self.test_material,
            1.0,
            self.test_wavelengths,
            self.test_ref_index)

        t3 = t1 + t2
        t4 = t2 - t1

        self.assertEqual(t3.thickness, 1.5)
        self.assertEqual(t4.thickness, 0.5)

    def test_thin_film_split_layer(self):
        """
        test split_layer()
        """

        t1 = ThinFilm(
            self.test_material,
            self.test_thickness,
            self.test_wavelengths,
            self.test_ref_index)

        t2 = t1.split_film()

        self.assertEqual(t1.thickness, self.test_thickness * 0.5)
        self.assertEqual(t2.thickness, self.test_thickness * 0.5)

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()