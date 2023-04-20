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
import time
import numpy as np
import numpy.testing as nptest
from tff_lib import ThinFilm, OpticalMedium

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
        cls.data_file = os.path.join(cls.dir, 'test_data.json')

        # read in json file with test input data
        with open(cls.data_file, 'r', encoding='utf=8') as dat:
            cls.test_data = json.load(dat)

        # set decimal precision
        cls._precision = 12

        # setup input data from test_expected.json
        cls._material = 'H'
        cls._thickness = 0.5
        cls._wavelengths = cls.test_data['input']['wv']
        cls._high_mat = [complex(x) for x in cls.test_data['input']['high_mat']]
        cls._low_mat = [complex(x) for x in cls.test_data['input']['low_mat']]
        cls._layers = cls.test_data['input']['layers']
        cls._char_matrix = {
            k: [complex(x) for x in v]
            for k,v in cls.test_data['output']['char_matrix'].items()
        }
        cls._admittance = cls.test_data['output']['admit_delta']

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
        cls._total_thick = sum(l[1] for l in cls._layers)

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

        # test OpticalMedium (air)
        cls._medium = OpticalMedium(
            cls.test_data['input']['wv'],
            [complex(1.0, 0) for i in cls.test_data['input']['wv']],
            'air'
        )

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

        # The following assertions are enough to validate
        # that self._stack == stk.stack. The stack objects
        # cannot be compared directly because FilmStack makes
        # a deep copy of the input stack to avoid unwanted
        # mutations to the original object

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
        for i,xval in enumerate(self._matrix):
            for j,yval in enumerate(xval):
                self.assertEqual(stk.matrix[i][j], yval)

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

    def test_append_layer(self):
        """
        test append_layer()
        """

        stk1 = FilmStack(self._stack)

        # this layer should work as expected
        lyr1 = ThinFilm('L', 500.0, self._wavelengths, self._low_mat)
        stk1.append_layer(lyr1)

        # validate last layer is appended correctly
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

        # remove layer[5] from test layers, combine surrounding layers
        new_lyr = [[self._layers[4][0], self._layers[4][1] + self._layers[6][1]]]
        mock_lyrs = self._layers[0:4] + new_lyr + self._layers[7:]
        mock_stack = [
            ThinFilm(
                lyr[0],
                lyr[1],
                self._wavelengths,
                self._high_mat if lyr[0] == 'H' else self._low_mat
            )
            for lyr in mock_lyrs
        ]
        mock_stack = FilmStack(mock_stack)

        test_stk = FilmStack(self._stack)
        test_stk.remove_layer(5)

        self.assertListEqual(mock_stack.layers, test_stk.layers)

    def test_get_layer(self):
        """
        test get_layer()
        """
        stk = FilmStack(self._stack)
        lyr = stk.get_layer(5)

        self.assertEqual(lyr.material, self._stack[5].material)
        self.assertEqual(lyr.thickness, self._stack[5].thickness)

    def test_admittance(self):
        """
        test admittance()
        """

        stk = FilmStack(self._stack)

        t_avg = np.zeros(1000)
        for i in range(1000):
            # time the execution
            start = time.perf_counter()

            adm = stk.admittance(self._medium, 0.0)

            # end time
            end = time.perf_counter()
            t_avg[i] = end - start

        # average time
        t_avg = np.mean(t_avg)
        sys.stdout.write(f"\nAvg Time= {round(t_avg, 4)} seconds.\n")

        nptest.assert_array_almost_equal(
            self._admittance['ns_film'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_film'], adm['p'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['delta'], adm['delta'], decimal=self._precision)

    def test_char_matrix(self):
        """
        test char_matrix()
        """

        stk = FilmStack(self._stack)

        t_avg = np.zeros(1000)
        for i in range(1000):
            # time the execution
            start = time.perf_counter()

            # test the char_matrix()
            cmat = stk.char_matrix(self._medium, 0.0)

            # end time
            end = time.perf_counter()
            t_avg[i] = end - start

        # average time
        t_avg = np.mean(t_avg)
        sys.stdout.write(f"\nAvg Time= {round(t_avg, 4)} seconds.\n")

        nptest.assert_array_almost_equal(
            self._char_matrix['S11'], cmat['S11'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['S12'], cmat['S12'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['S21'], cmat['S21'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['S22'], cmat['S22'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['P11'], cmat['P11'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['P12'], cmat['P12'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['P21'], cmat['P21'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._char_matrix['P22'], cmat['P22'], decimal=self._precision)

    def test_insert_layer(self):
        """
        PENDING -----> test insert_layer()
        """

    def test_film_stack_invalid_inputs(self):
        """
        PENDING -----> test __init__() with invalid input values
        """

    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """
        sys.stdout.write('\nRunning teardown procedure... SUCCESS ')
        sys.stdout.close()

if __name__=='__main__':
    unittest.main()