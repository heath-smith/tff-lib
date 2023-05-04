#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suite for the FilmStack
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
        cls._ntype = 1  # 1 = high index material
        cls._thick = 0.5
        cls._waves = cls.test_data['input']['wv']
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
                cls._waves,
                cls._high_mat if lyr[0] == 1 else cls._low_mat,
                thick=lyr[1],
                ntype=lyr[0]
            )
            for lyr in cls._layers
        ]

        # test total thick
        cls._total_thick = sum(l[1] for l in cls._layers)

        # test matrix
        cls._matrix = [
            cls._high_mat if l[0] == 1 else cls._low_mat
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
        lyr1 = ThinFilm(self._waves, self._low_mat, thick=500.0, ntype=0)
        stk1.append_layer(lyr1)

        # validate last layer is appended correctly
        self.assertEqual(stk1.stack[-1], lyr1)


    def test_remove_layer(self):
        """
        test remove_layer()
        """

        # remove layer[5] from test layers, combine surrounding layers
        new_lyr = [[self._layers[4][0], self._layers[4][1] + self._layers[6][1]]]
        mock_lyrs = self._layers[0:4] + new_lyr + self._layers[7:]
        mock_stack = [
            ThinFilm(
                self._waves,
                self._high_mat if lyr[0] == 1 else self._low_mat,
                thick=lyr[1],
                ntype=lyr[0]
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

        self.assertEqual(lyr.ntype, self._stack[5].ntype)
        self.assertEqual(lyr.thick, self._stack[5].thick)

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
        sys.stdout.write(f"\nFilmStack.admittance() Avg Time= {round(t_avg, 4)} seconds.\n")

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
        sys.stdout.write(f"\nFilmStack.char_matrix() Avg Time= {round(t_avg, 4)} seconds.\n")

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
        test insert_layer()
        """

        stk1 = FilmStack(self._stack)

        # this layer should work as expected
        lyr1 = ThinFilm(self._waves, self._low_mat, thick=500.0, ntype=0)
        stk1.insert_layer(lyr1, 4)

        # assert number of layers is correct (should be n + 1)
        self.assertEqual(stk1.num_layers, len(self._stack) + 1)

        # assert new_lyr[4] == inserted layer
        self.assertEqual(stk1.stack[4].thick, lyr1.thick)


    def test_insert_split_layer(self):
        """
        test insert_split_layer()
        """

        stk1 = FilmStack(self._stack)

        # this layer should work as expected
        lyr1 = ThinFilm(self._waves, self._low_mat, thick=500.0, ntype=0)
        stk1.insert_split_layer(lyr1, 4)

        # assert number of layers is correct (should be n + 2)
        self.assertEqual(stk1.num_layers, len(self._stack) + 2)

        # assert new_lyr[5] == inserted layer
        self.assertEqual(stk1.stack[5].thick, lyr1.thick)

        # assert thickness of new_lyr[i] and new_lyr[i + 2] == 1/2 lyr[i].thick
        self.assertEqual(stk1.stack[4].thick, self._stack[4].thick * 0.5)
        self.assertEqual(stk1.stack[6].thick, self._stack[4].thick * 0.5)


    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
