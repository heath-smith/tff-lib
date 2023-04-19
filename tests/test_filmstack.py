#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suit for the FilmStack
class.

Usage
---------
>>> python -m unittest -v tests.test_filmstack
"""

# import external packages
import unittest
from pathlib import Path
import sys
import os
import json
from tff_lib import ThinFilm

# class under test
from tff_lib import FilmStack

class TestFilmStack(unittest.TestCase):
    """
    Test suite for FilmStack() class.
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
        cls._material = 'H'
        cls._thickness = 0.5
        cls._wavelengths = cls.test_data['input']['wv']
        cls._high_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._layers = cls.test_data['input']['layers']

        # generate test FilmStack
        cls._stack = [
            ThinFilm(
                lyr[0],
                lyr[1],
                cls._wavelengths,
                cls._high_mat if lyr[0] == 'H' else cls._low_mat
            )
            for lyr in cls._layers
        ]

        # test total thick
        cls._total_thick = sum([l[1] for l in cls._layers])

        # test matrix
        cls._matrix = [
            cls._high_mat if l[0] == 'H' else cls._low_mat
            for l in cls._layers
        ]

        # default kwargs
        cls._defaults = {
            'max_total_thick': 20_000,
            'max_layers': 25,
            'min_layers': 5,
            'first_lyr_min_thick': 500,
            'min_thick': 10
        }

        # test kwargs (these are different from defaults)
        cls._kwargs = {
            'max_total_thick': 25_000,
            'max_layers': 20,
            'min_layers': 2,
            'first_lyr_min_thick': 250,
            'min_thick': 50
        }

    def test_film_stack_init_defaults(self):
        """
        test __init__()
        """

        stk = FilmStack(self._stack)

        # assert default values
        self.assertEqual(stk.max_total_thick, self._defaults['max_total_thick'])
        self.assertEqual(stk.max_layers, self._defaults['max_layers'])
        self.assertEqual(stk.first_lyr_min_thick, self._defaults['first_lyr_min_thick'])
        self.assertEqual(stk.min_thick, self._defaults['min_thick'])
        self.assertEqual(stk.min_layers, self._defaults['min_layers'])

        # assert stack property
        self.assertListEqual(self._stack, stk.stack)

        # assert total_thick property
        self.assertEqual(self._total_thick, stk.total_thick)

        # assert num_layers property
        self.assertEqual(len(self._layers), stk.num_layers)

        # assert layers property (validates thickness values match)
        self.assertListEqual(stk.layers, [l[1] for l in self._layers])

        # assert matrix property shape is valid
        self.assertTupleEqual(
            (len(self._matrix), len(self._matrix[0])),
            (len(stk.matrix), len(stk.matrix[0])))

        # assert matrix property values are valid
        for x in range(len(self._matrix)):
            for y in range(len(self._matrix[x])):
                self.assertEqual(stk.matrix[x][y], self._matrix[x][y])

    def test_film_stack_init_kwargs(self):
        """
        test __init__(**kwargs)
        """

        stk = FilmStack(self._stack, **self._kwargs)

        # assert default values
        self.assertEqual(stk.max_total_thick, self._kwargs['max_total_thick'])
        self.assertEqual(stk.max_layers, self._kwargs['max_layers'])
        self.assertEqual(stk.first_lyr_min_thick, self._kwargs['first_lyr_min_thick'])
        self.assertEqual(stk.min_thick, self._kwargs['min_thick'])
        self.assertEqual(stk.min_layers, self._kwargs['min_layers'])

    def test_insert_layer(self):
        """
        test insert_layer() ---> PENDING
        """

    def test_append_layer(self):
        """
        test append_layer()
        """

        stk1 = FilmStack(self._stack)

        # this layer should work as expected
        lyr1 = ThinFilm('L', 500.0, self._wavelengths, self._low_mat)
        stk1.append_layer(lyr1)

        self.assertEqual(stk1.stack[-1], lyr1)


        # test film stack 2
        stk2 = FilmStack(self._stack)

        with self.assertRaises(ValueError):
            # this layer should give a ValueError
            lyr2 = ThinFilm('H', 500.0, self._wavelengths, self._high_mat)
            stk2.append_layer(lyr2)

    def test_remove_layer(self):
        """
        test remove_layer()
        """

    def test_get_layer(self):
        """
        test get_layer()
        """

    def test_admittance(self):
        """
        test admittance()
        """

    def test_char_matrix(self):
        """
        test char_matrix()
        """


    ##def test_film_stack_invalid_inputs(self):
    ##    """
    ##    test __init__() with invalid input values
    ##    """

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()