#!user/bin/python
# -*- coding: utf-8 -*-
"""
This module contains the test suite for the OpticalMedium
class.

Usage
---------
>>> python -m unittest -v tests.test_medium
"""

# import external packages
import unittest
from pathlib import Path
import os
import json
import numpy.testing as nptest

# class under test
from tff_lib import OpticalMedium

class TestMedium(unittest.TestCase):
    """
    Test suite for Substrate() class.
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
        cls._precision = 14

        # test inputs
        cls._theta = 0.0
        cls._thick = cls.test_data['input']['sub_thick'] * 10**6
        cls._ntype = -1
        cls._waves = cls.test_data['input']['wv']
        cls._nref = [complex(n, 0) for n in cls.test_data['input']['substrate']]

        # test incident medium (air)
        cls._incident = [complex(1.0, 0) for i in cls.test_data['input']['wv']]

        # test expected outputs
        cls._admittance = cls.test_data['output']['admit_delta']
        cls._nref_effective = cls.test_data['output']['effective_index']
        cls._path_length = cls.test_data['output']['path_length']
        cls._fresnel = cls.test_data['output']['fresnel_bare']

    def test__init__(self):
        """
        test __init__()
        """

        opt = OpticalMedium(self._waves,
                            self._nref,
                            thick=self._thick,
                            ntype=self._ntype)

        # assert attributes are valid
        self.assertEqual(self._thick, opt.thick)
        self.assertEqual(self._ntype, opt.ntype)
        nptest.assert_array_almost_equal(
            self._waves, opt.waves, decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._nref, opt.nref, decimal=self._precision)

    def test__init__invalid_inputs(self):
        """
        test __init__() with invalid input values
        """

        # test with shortened wavelength array
        with self.assertRaises(ValueError):
            OpticalMedium(self._waves[:-2], self._nref)

        # test with shortened nref array
        with self.assertRaises(ValueError):
            OpticalMedium(self._waves, self._nref[:-2])

        # test with invalid thickness value
        with self.assertRaises(ValueError):
            OpticalMedium(self._waves, self._nref, thick=-5)


    def test_thickness_getset(self):
        """
        test thickness @getter @setter
        """

        opt = OpticalMedium(self._waves, self._nref)

        # change the thickness
        opt.thick = 250.0
        self.assertEqual(250.0, opt.thick)

        # set invalid thickness
        with self.assertRaises(ValueError):
            opt.thick = -5  # can only be > 0 or == -1

    def test_ntype_getset(self):
        """
        test ntype @getter @setter
        """

        opt = OpticalMedium(self._waves, self._nref)

        # change the thickness
        opt.ntype = 2
        self.assertEqual(2, opt.ntype)


    def test_absorption_coeffs(self):
        """
        test absorption_coeffs()
        """

        opt = OpticalMedium(self._waves, self._nref)

        abs_coeffs = opt.absorption_coeffs()

        # expect to be all zero's since substrate has
        # no non-zero imaginary values
        self.assertEqual(0, sum(abs_coeffs))

    def test_nref_eff(self):
        """
        test nref_eff()
        """

        # test effective index using substrate as medium
        opt = OpticalMedium(self._waves, self._nref)

        nref_eff = opt.nref_eff(self._theta)

        # when theta == 0, expect eff_index == ref_index
        nptest.assert_array_almost_equal(
            self._nref, nref_eff, decimal=self._precision)

    def test_admittance(self):
        """
        test admittance()
        """

        # test admittance using air as both medium and incident medium
        med = OpticalMedium(self._waves, self._incident)
        inc = OpticalMedium(self._waves, self._incident)

        # calculate air-air interface admittance
        adm = med.admittance(inc, self._theta)

        nptest.assert_array_almost_equal(
            self._admittance['ns_inc'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_inc'], adm['p'], decimal=self._precision)

        # test admittance using substrate with air as incident medium
        sub = OpticalMedium(self._waves, self._nref)

        # calculate air-air interface admittance
        sub_adm = sub.admittance(inc, self._theta)

        nptest.assert_array_almost_equal(
            self._admittance['ns_sub'], sub_adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_sub'], sub_adm['p'], decimal=self._precision)

    def test_path_length(self):
        """
        test path_length()
        """

        # create a medium using a substrate
        med = OpticalMedium(self._waves, self._nref, thick=self._thick)

        # create an incident medium of air with infinite thickness
        inc = OpticalMedium(self._waves, self._incident)

        p_len = med.path_length(inc, self._theta)

        nptest.assert_array_almost_equal(
            self._path_length, p_len, decimal=self._precision)

    def test_path_length_negative_thick(self):
        """
        test path_length() with negative thickness
        """

        # create a medium with default -1 thickness
        med = OpticalMedium(self._waves, self._nref)

        # create an incident medium of air with infinite thickness
        inc = OpticalMedium(self._waves, self._incident)

        # med.thick == -1, should raise error
        with self.assertRaises(ValueError):
            med.path_length(inc, self._theta)

    def test_admittance_eff(self):
        """
        test admittance_eff()
        """

        # create a finite medium
        med = OpticalMedium(self._waves, self._nref, thick=self._thick)

        # create infinite incident medium 'air'
        inc = OpticalMedium(self._waves, self._incident)

        # test admittance_eff() method
        adm = med.admittance_eff(inc, self._theta)

        # expect effective admittance to equal admittance
        # at theta == 0
        nptest.assert_array_almost_equal(
            self._admittance['ns_sub'], adm['s'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._admittance['np_sub'], adm['p'], decimal=self._precision)

    def test_fresnel_coeffs(self):
        """
        test fresnel_coeffs()
        """

        # create an optical medium with finite thickness
        med = OpticalMedium(self._waves, self._nref, thick=self._thick)

        # create an infinite incident medium of air
        inc = OpticalMedium(self._waves, self._incident)

        fresnel = med.fresnel_coeffs(inc, self._theta)

        nptest.assert_array_almost_equal(
            self._fresnel['Ts'], fresnel['Ts'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Tp'], fresnel['Tp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Rs'], fresnel['Rs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Rp'], fresnel['Rp'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Fs'], fresnel['Fs'], decimal=self._precision)
        nptest.assert_array_almost_equal(
            self._fresnel['Fp'], fresnel['Fp'], decimal=self._precision)



    @classmethod
    def tearDownClass(cls):
        """
        Cleans up any open resources.
        """

if __name__=='__main__':
    unittest.main()
