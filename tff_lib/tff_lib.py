"""
The filters.py module contains common functions for calculating characteristics
of optical filters and thin film stacks. The Filters class contains static methods
that can be called without any class instance.

To install via pip:

>>> python -m pip install git+<url>
"""

# import external packages
import numpy as np

# import custom exceptions classes
from tff_lib.exceptions import UnitError

class ThinFilmFilter:
    """
    The ThinFilmFilter class is the main class of the tff_lib package. This class
    contains static methods that are used to calculate the properties of
    thin film interference filters.
    """

    @staticmethod
    def fresnel_bare(i_n, s_n, theta, units='rad'):
        """
        (static method) Calculates the fresnel amplitudes & intensities of the substrate.

        Parameters
        -----------
        i_n (array): complex refractive index on incident medium (i_n = x+y*j)\n
        s_n (array): complex refractive index of substrate (s_n = x+y*j) \n
        theta (float): angle of incidence of radiation.\n
        units (str): 'deg' or 'rad'. Default is 'rad' if not specified.

        Returns
        -----------
        dictionary object of transmission and reflection value arrays\n
        {key : result array}\n
        { 'Ts' : s-polarized Fresnel Transmission Intensity,\n
        'Tp' : p-polarized Fresnel Transmission Intensity,\n
        'Rs' : s-polarized Fresnel Reflection Intensity,\n
        'Rp' : p-polarized Fresnel Reflection Intensity,\n
        'rs' : s-polarized Fresnel Reflection Amplitude,\n
        'rp' : p-polarized Fresnel Reflection Amplitude }

        Raises
        -----------
        UnitError if input 'units' parameter is not 'rad' or 'deg'.\n
        TypeError if 'units' is not a 'str' type.\n
        TypeError if 's_n' or 'f_n' is not 'complex' or 'complex128' type
                    or if data structure is not numpy.ndarray.

        Examples
        --------------
        >>> fr_bare = fresnel_bare(i_n, s_n, theta, units='rad')
        >>> big_t_s = fr_bare['Ts']
        >>> big_r_s = fr_bare['Rs']
        """

        # check if 'units' param is valid
        if units not in ('rad', 'deg'):
            # raise a custom 'UnitError' if 'units' not valid
            err_msg = "Invalid Units. Valid inputs are 'rad' or 'deg'."
            raise UnitError(units, err_msg)

        if not isinstance(units, str):
            # raise a TypeError if 'units' is not a string
            err_msg = "TypeError: 'units' parameter expects type 'str'."
            raise TypeError(err_msg)

        if units == 'deg':
            # if input theta is 'deg', convert to radians
            theta = theta * (np.pi / 180)

        # validate i_n, and s_n input arrays, they should be 'complex' or 'complex128'
        # data structure should be 'np.ndarray'
        for arr in [i_n, s_n]:
            if arr.dtype != 'complex128' or arr.dtype != 'complex':
                # raise TypeError if any array is not complex
                raise TypeError("Incorrect type: expected 'complex' type but received "
                                 + str(arr.dtype))
            if not isinstance(arr, np.ndarray):
                # raise TypeError if not a numpy ndarray
                raise TypeError("Bad Data Structure: Expected 'numpy.ndarray' but received "
                                + type(arr))

        # Calculation of the Fresnel Amplitude Coefficients
        rs_num = (i_n * np.cos(theta) - np.sqrt(np.square(s_n)
                - np.square(i_n) * np.square(np.sin(theta))))
        rs_den = (i_n * np.cos(theta) + np.sqrt(np.square(s_n)
                - np.square(i_n) * np.square(np.sin(theta))))
        r_s = rs_num / rs_den

        rp_num = -(np.square(s_n) * np.cos(theta) - i_n * np.sqrt(np.square(s_n)
                - np.square(i_n) * np.square(np.sin(theta))))
        rp_den = (np.square(s_n) * np.cos(theta) + i_n * np.sqrt(np.square(s_n)
                - np.square(i_n) * np.square(np.sin(theta))))
        r_p = rp_num / rp_den

        # Calculation of Fresnel Intensities for bare substrate interface
        big_r_s = np.square(np.abs(r_s))
        big_r_p = np.square(np.abs(r_p))

        return {'Ts':(1 - big_r_s), 'Tp':(1 - big_r_p), 'Rs':big_r_s,
                'Rp':big_r_p, 'rs':r_s, 'rp':r_p}

    @staticmethod
    def admit_delta(*args, **kwargs):
        """
        (static method) Calculates filter admittances of incident, substrate,
        and film as well as the phase (delta) upon reflection for each film.

        Parameters
        -------------
        *args:\n
        wv_range (array): wavelength values in nanometers.\n
        layer_stack (array): thickness values of each filter layer in nanometers.\n
        theta (float): angle of incidence of radiation\n
        i_n (array): complex refractive index of incident medium (i_n = n + i*k)\n
        s_n (array): complex refractive index of substrate (s_n = n + i*k)\n
        f_n (array): complex refractive index of thin films (f_n = n + i*k)

        **kwargs: (optional)\n
        units (str): units of measure for angle theta. Valid options are
                    'rad' and 'deg'. Default value is 'rad'.

        Returns
        --------------
        dictionary object of transmission and reflection value arrays\n
        {key : result array}\n
        { 'ns_inc' : s-polarized admittance of the incident medium,\n
        'np_inc' : p-polarized admittance of the incident medium,\n
        'ns_sub' : s-polarized admittance of the substrate medium,\n
        'np_sub' : p-polarized admittance of the substrate medium,\n
        'ns_film' : s-polarized admittance of the film stack layers,\n
        'np_film' : p-polarized admittance of the film stack layers,\n
        'delta' : phase upon reflection for each film }

        Raises
        ------------
        TypeError if either i_n, s_n, or f_n is not type 'complex' or
        'complex128'. Also raises TypeError if input arrays are not
        'numpy.ndarray' types.\n
        TypeError if 'units' param is not a string.\n
        UnitError if units is not 'deg' or 'rad'.\n
        ValueError if array shapes do not match.

        Notes
        --------------
        Input arrays should be row-based and be equal to shape of wv_range.
        ie: shape = (1, N)

        Examples
        --------------
        >>> admittance = admit_delta(wv_range, layer_stack, i_n, s_n, f_n, theta, units='deg')
        >>> ns_inc = admittance['ns_inc']
        >>> d = admittance['delta']

        """

        # dictionary to store input args
        input_args = {'wv_range':[],
                    'layer_stack':[],
                    'theta':[],
                    'i_n':[],
                    's_n':[],
                    'f_n':[]}

        # check if args list is correct length
        if len(args) == 6:
            # iterate input_args keys
            for i, k in enumerate(input_args):
                # define input_args key/value pairs
                input_args[k] = args[i]
        elif len(args) != 6:
            # raise ValueError exception if args len incorrect
            if len(args) > 6:
                raise ValueError("Too many input arguments. Expected 8 but received "
                                + str(len(args)))
            if len(args) < 6:
                raise ValueError("Not enough input arguments. Expected 8 but received "
                                + str(len(args)))

        # check if 'units' param is valid
        if 'units' in kwargs:
            if kwargs['units'] not in ('rad', 'deg'):
                # raise a custom 'UnitError' if 'units' not valid
                raise UnitError(kwargs['units'], "Invalid Units. Valid inputs are 'rad' or 'deg'.")

            if not isinstance(kwargs['units'], str):
                # raise a TypeError if 'units' is not a string
                raise TypeError("TypeError: 'units' parameter expects type 'str' but received "
                                + type(kwargs['units']))

            if kwargs['units'] == 'deg':
                # if input theta is 'deg', convert to radians
                input_args['theta'] = input_args['theta'] * (np.pi / 180)

        # validate i_n, s_n, and f_n input arrays they should be 'complex' or 'complex128'
        # data structure should be 'np.ndarray' and all shapes should match wv_range
        for i, arr in enumerate([input_args['i_n'], input_args['s_n'], input_args['f_n']]):
            if arr.dtype != 'complex128' or arr.dtype != 'complex':
                # raise TypeError if any array is not complex
                raise TypeError("Incorrect type: expected 'complex' type but received "
                                 + str(arr.dtype))
            if not isinstance(arr, np.ndarray):
                # raise TypeError if not a numpy ndarray
                raise TypeError("Bad Data Structure: Expected 'numpy.ndarray' but received "
                                + type(arr))

            # validate that shapes of i_n, s_n are equal to the wv_range shape
            if i < 2:
                if np.shape(arr) != np.shape(input_args['wv_range']):
                    raise ValueError("ValueError: Expected arrays of shape "
                                    + str(np.shape(input_args['wv_range'])) + " but received "
                                    + str(np.shape(arr)))
            # validate f_n shape is equal to LEN(layer_stack) X LEN(wl_range)
            if i == 2:
                if np.shape(arr) != (len(input_args['layer_stack']),
                                np.shape(input_args['wv_range'])[1]):
                    raise ValueError("ValueError: Expected arrays of shape ("
                        + str(len(input_args['layer_stack'])) + ', '
                        + str(np.shape(input_args['wv_range'])[1])
                        + ") but received " + str(np.shape(arr)))

        # initialize dictionary to store admit_delta calculations
        admit_calc = {}

        # Calculation of the complex dielectric constants from measured
        # optical constants
        admit_calc['i_e'] = np.square(input_args['i_n']) # incident medium (air)
        admit_calc['s_e'] = np.square(input_args['s_n']) # substrate
        admit_calc['f_e'] = np.square(input_args['f_n']) # thin films

        # Calculation of the admittances of the incident and substrate media
        admit_calc['ns_inc'] = np.sqrt(admit_calc['i_e']
                            - admit_calc['i_e']
                            * np.sin(input_args['theta'])**2)
        admit_calc['np_inc'] = (admit_calc['i_e']
                            / admit_calc['ns_inc'])
        admit_calc['ns_sub'] = np.sqrt(admit_calc['s_e']
                            - admit_calc['i_e']
                            * np.sin(input_args['theta'])**2)
        admit_calc['np_sub'] = (admit_calc['s_e']
                            / admit_calc['ns_sub'])

        # Calculation of the admittances & phase factors
        # for each layer of the film stack
        admit_calc['ns_film'] = np.ones((len(input_args['layer_stack']),
                    np.shape(input_args['wv_range'])[1])).astype(complex)
        admit_calc['np_film'] = np.ones((len(input_args['layer_stack']),
                    np.shape(input_args['wv_range'])[1])).astype(complex)
        admit_calc['delta'] = np.ones((len(input_args['layer_stack']),
                    np.shape(input_args['wv_range'])[1])).astype(complex)

        # enter loop if the substrate has thin film layers
        if len(input_args['f_n'][:, 0]) >= 1:
            for i, layer in enumerate(input_args['layer_stack']):
                admit_calc['ns_film'][i, :] = np.sqrt(admit_calc['f_e'][i, :]
                                        - admit_calc['i_e']
                                        * np.sin(input_args['theta'])**2)
                admit_calc['np_film'][i, :] = (admit_calc['f_e'][i, :]
                                        / admit_calc['ns_film'][i, :])
                admit_calc['delta'][i, :] = ((2 * np.pi * layer
                            * np.sqrt(admit_calc['f_e'][i, :]
                            - admit_calc['i_e']
                            * np.sin(input_args['theta'])**2))
                            / np.array(input_args['wv_range']))

        admit_calc['ns_film'] = np.flipud(admit_calc['ns_film'])
        admit_calc['np_film'] = np.flipud(admit_calc['np_film'])
        admit_calc['delta'] = np.flipud(admit_calc['delta'])

        # Flip layer-based arrays ns_film, np_film, delta
        # since the last layer is the top layer
        return admit_calc

    @staticmethod
    def c_mat(ns_film, np_film, delta):
        """
        (static method) Calculates the characteristic matrix for multiple thin film layers.

        Parameters
        -----------
        'ns_film' (array): s-polarized admittance of the film stack layers.\n
        'np_film' (array): p-polarized admittance of the film stack layers.\n
        'delta' (array): phase upon reflection for each film.


        Returns
        ------------
        dictionary object of characteristic matrix\n
        {key : result array}\n
        { S11 (array): matrix entry\n
        S12 (array): matrix entry\n
        S21 (array): matrix entry\n
        S22 (array): matrix entry\n
        P11 (array): matrix entry\n
        P12 (array): matrix entry\n
        P21 (array): matrix entry\n
        P22 (array): matrix entry }

        Raises
        --------------
        ValueError if dimensions of ns_film, np_film, and delta do not match.
        TypeError if array types are not float.
        TypeError if arrays are not numpy.ndarray type.

        Examples
        ------------
        >>> char_matrix = c_mat(nsFilm, npFilm, delta)
        >>> ns_film = char_matrix[ 'nsFilm' ]
        >>> d = char_matrix[ 'delta' ]
        """

        # validate input arguments
        for arr in [ns_film, np_film, delta]:
            # if np.ndarray's, check the data type
            if isinstance(arr, np.ndarray):
                if arr.dtype != 'float64':
                    # raise TypeError if any array is not complex
                    raise TypeError("Expected 'float64' type but received "
                                    + str(arr.dtype))
            # if not instance of np.ndarray, raise exception
            if not isinstance(arr, np.ndarray):
                # raise TypeError if not a numpy ndarray
                raise TypeError("Expected 'numpy.ndarray' but received "
                                + type(arr))

        # Calculation of the characteristic matrix elements
        elements = {'s11': np.cos(delta),
                    's22': np.cos(delta),
                    'p11': np.cos(delta),
                    'p22': np.cos(delta),
                    's12': (1j / ns_film) * np.sin(delta),
                    'p12': (-1j / np_film) * np.sin(delta),
                    's21': (1j * ns_film) * np.sin(delta),
                    'p21': (-1j * np_film) * np.sin(delta)}

        # Initialize the characteristic matrices
        char_mat = {'S11':np.ones((1, np.shape(elements['s11'])[1])).astype(complex),
                    'S12':np.zeros((1, np.shape(elements['s11'])[1])).astype(complex),
                    'S21':np.zeros((1, np.shape(elements['s11'])[1])).astype(complex),
                    'S22':np.ones((1, np.shape(elements['s11'])[1])).astype(complex),
                    'P11':np.ones((1, np.shape(elements['p11'])[1])).astype(complex),
                    'P12':np.zeros((1, np.shape(elements['p11'])[1])).astype(complex),
                    'P21':np.zeros((1, np.shape(elements['p11'])[1])).astype(complex),
                    'P22':np.ones((1, np.shape(elements['p11'])[1])).astype(complex)}

        # Multiply all of the individual layer characteristic matrices together
        for i in range(np.shape(elements['s11'])[0]):
            A = char_mat['S11']
            B = char_mat['S12']
            C = char_mat['S21']
            D = char_mat['S22']
            char_mat['S11'] = (A * elements['s11'][i, :] + B * elements['s21'][i, :])
            char_mat['S12'] = (A * elements['s12'][i, :] + B * elements['s22'][i, :])
            char_mat['S21'] = (C * elements['s11'][i, :] + D * elements['s21'][i, :])
            char_mat['S22'] = (C * elements['s12'][i, :] + D * elements['s22'][i, :])

        for i in range(np.shape(elements['p11'])[0]):
            A = char_mat['P11']
            B = char_mat['P12']
            C = char_mat['P21']
            D = char_mat['P22']
            char_mat['P11'] = (A * elements['p11'][i, :] + B * elements['p21'][i, :])
            char_mat['P12'] = (A * elements['p12'][i, :] + B * elements['p22'][i, :])
            char_mat['P21'] = (C * elements['p11'][i, :] + D * elements['p21'][i, :])
            char_mat['P22'] = (C * elements['p12'][i, :] + D * elements['p22'][i, :])

        return char_mat

    @staticmethod
    def fresnel_film(admit_deltas, char_matrix):
        """
        Calculates the fresnel amplitudes & intensities of film. Requires
        output from admit_delta() and c_mat() methods.

        Parameters
        ------------
        admit_deltas (dict): results of admit_delta() method\n
        char_matrix (dict): results of c_mat() method\n

        Returns
        -------------
        dictionary object of transmission and reflection value arrays\n
        {key : result array}\n
        { 'Ts' : s-polarized Fresnel Transmission Intensity,\n
        'Tp' : p-polarized Fresnel Transmission Intensity,\n
        'Rs' : s-polarized Fresnel Reflection Intensity,\n
        'Rp' : p-polarized Fresnel Reflection Intensity,\n
        'ts' : s-polarized Fresnel Transmission Amplitude,\n
        'tp' : p-polarized Fresnel Transmission Amplitude,\n
        'rs' : s-polarized Fresnel Reflection Amplitude,\n
        'rp' : p-polarized Fresnel Reflection Amplitude }

        Raises
        ------------

        Examples
        --------------
        fresnel = fresnel_film(admit_deltas, char_matrix)\n
        T_p = fresnel[ 'Tp' ]\n
        R_s = fresnel[ 'Rs' ]
        """

        # define dict for incident interface admittances
        admit_inc = {}

        # Calculation of the admittances of the incident interface
        admit_inc['ns_f'] = ((np.array(char_matrix['S21']).astype(complex)
                        - np.array(admit_deltas['ns_sub']).astype(complex)
                        * np.array(char_matrix['S22']).astype(complex))
                        / (np.array(char_matrix['S11']).astype(complex)
                        - np.array(admit_deltas['ns_sub']).astype(complex)
                        * np.array(char_matrix['S12']).astype(complex)))
        admit_inc['np_f'] = ((np.array(char_matrix['P21']).astype(complex)
                        + np.array(admit_deltas['np_sub']).astype(complex)
                        * np.array(char_matrix['P22']).astype(complex))
                        / (np.array(char_matrix['P11']).astype(complex)
                        + np.array(admit_deltas['np_sub']).astype(complex)
                        * np.array(char_matrix['P12']).astype(complex)))

        # define a dict to store fresnel amplitude calculations
        fresnel_calcs = {}

        # Calculation of the Fresnel Amplitude Coefficients
        fresnel_calcs['rs'] = ((np.array(admit_deltas['ns_inc']).astype(complex)
                                + admit_inc['ns_f'])
                                / (np.array(admit_deltas['ns_inc']).astype(complex)
                                - admit_inc['ns_f']))
        fresnel_calcs['rp'] = ((np.array(admit_deltas['np_inc']).astype(complex)
                                - admit_inc['np_f'])
                                / (np.array(admit_deltas['np_inc']).astype(complex)
                                + admit_inc['np_f']))
        fresnel_calcs['ts'] = ((1 + fresnel_calcs['rs'])
                                / (np.array(char_matrix['S11']).astype(complex)
                                - np.array(char_matrix['S12']).astype(complex)
                                * np.array(admit_deltas['ns_sub']).astype(complex)))
        fresnel_calcs['tp'] = ((1 + fresnel_calcs['rp'])
                                / (np.array(char_matrix['P11']).astype(complex)
                                + np.array(char_matrix['P12']).astype(complex)
                                * np.array(admit_deltas['np_sub']).astype(complex)))

        # Calculation of the Fresnel Amplitude Intensities
        fresnel_calcs['Rs'] = np.square(np.abs(fresnel_calcs['rs']))
        fresnel_calcs['Rp'] = np.square(np.abs(fresnel_calcs['rp']))
        fresnel_calcs['Ts'] = (np.real(np.array(admit_deltas['ns_sub']).astype(complex)
                            / np.array(admit_deltas['ns_inc']).astype(complex))
                            * np.square(np.abs(fresnel_calcs['ts'])))
        fresnel_calcs['Tp'] = (np.real(np.array(admit_deltas['np_sub']).astype(complex)
                            / np.array(admit_deltas['np_inc']).astype(complex))
                            * np.square(np.abs(fresnel_calcs['tp'])))

        return fresnel_calcs

    @staticmethod
    def sub_n_eff(s_n, theta, units='rad'):
        """
        Calculates the effective substrate refractive index for abs. substrates.

        Parameters
        -----------

        Returns
        ----------
        (numpy.ndarray) Effective substrate refractive index.
        """

        if units == 'deg':
            # convert incident angle from degrees to radians
            theta = theta * (np.pi / 180)

        inside1 = (np.imag(s_n)**2 + np.real(s_n)**2)**2 + 2 * (np.imag(s_n) - \
            np.real(s_n)) * (np.imag(s_n) + np.real(s_n)) * np.sin(theta)**2 + \
            np.sin(theta)**4
        inside1 = np.array(inside1, dtype=np.float64)
        inside2 = 0.5 * (-np.imag(s_n)**2 + np.real(s_n)**2 + np.sin(theta)**2 + \
            np.sqrt(inside1))
        inside2 = np.array(inside2, dtype=np.float64)
        sn_eff = np.sqrt(inside2)

        return np.array(sn_eff).astype(complex)

    @staticmethod
    def path_len(sub_thick, i_n, sn_eff, theta, units='rad'):
        """
        Calculates the optical path length through a given substrate.

        Parameters
        -------------
        sub_thick
        theta
        units

        Returns
        ------------
        (float) Length of the path through the substrate.
        """

        if units == 'deg':
            # convert incident angle from degrees to radians
            theta = theta * (np.pi / 180)

        len_path = ((sub_thick * (10**6))
                / np.sqrt(1 - (np.abs(i_n)**2
                * (np.sin(theta)**2) / sn_eff**2)))
        len_path = len_path.astype(complex)

        return len_path

    @staticmethod
    def incident_reflection(wv_range, layers, i_n, s_n, f_n, theta, units='rad'):
        """
        Calculates the reflection that originates from the incident medium.

        Parameters
        -----------

        Returns
        -----------


        """

        if units == 'deg':
            # convert incident angle from degrees to radians
            theta = theta * (np.pi / 180)

        inc_ref = {}

        # calculate admittances
        inc_ref['admit_delta'] = ThinFilmFilter.admit_delta(wv_range, layers, theta, i_n, s_n, f_n)

        # calculate the characteristic matrix
        inc_ref['c_mat'] = ThinFilmFilter.c_mat(
                            np.array(inc_ref['admit_delta']['ns_film']).astype('float64'),
                            np.array(inc_ref['admit_delta']['np_film']).astype('float64'),
                            np.array(inc_ref['admit_delta']['delta']).astype('float64'))

        # calculate fresnel intensities and amplitudes
        inc_ref['fresnel_film'] = ThinFilmFilter.fresnel_film(inc_ref['admit_delta'],
                                                            inc_ref['c_mat'])

        return inc_ref

    @staticmethod
    def substrate_reflection(wv_range, layers, i_n, s_n, f_n, theta, sn_eff, units='rad'):
        """
        Calculates the reflection that originates from the substrate medium.

        Parameters
        ----------

        Returns
        ----------

        """

        if units == 'deg':
            # convert incident angle from degrees to radians
            theta = theta * (np.pi / 180)

        sub_ref = {}

        # angle of substrate reflectance
        theta = np.arcsin(i_n / s_n * np.sin(theta))

        # calculate admittances
        sub_ref['admit_delta'] = ThinFilmFilter.admit_delta(wv_range, layers, theta,
                                                sn_eff, i_n, np.flipud(f_n))

        # calculate characteristic matrix
        sub_ref['c_mat'] = ThinFilmFilter.c_mat(
                            np.array(sub_ref['admit_delta']['ns_film']).astype('float64'),
                            np.array(sub_ref['admit_delta']['np_film']).astype('float64'),
                            np.array(sub_ref['admit_delta']['delta']).astype('float64'))

        # calculate fresnel intensities & amplitudes
        sub_ref['fresnel_film'] = ThinFilmFilter.fresnel_film(sub_ref['admit_delta'],
                                                            sub_ref['c_mat'])

        return sub_ref

    @staticmethod
    def fil_spec(*args, **kwargs):
        #(units='deg'):
        """
        Calculates the transmission and reflection spectra of a
        thin-film interference filter.

        Parameters
        ------------
        *args:\n
        wv_range
        substrate
        h_mat
        l_mat
        layer_stack
        materials
        theta
        sub_thick

        **kwargs:\n
        units='rad'

        Returns
        -----------
        dictionary object of transmission and reflection value arrays\n
        {key : result array}\n
        { 'T' : average transmission spectrum over wavelength range ([Tp + Ts] / 2),\n
        'Ts' : s-polarized transmission spectrum wavelength range,\n
        'Tp' : p-polarized transmission spectrum over wavelength range,\n
        'R' : average reflection spectrum over wavelength range,\n
        'Rs' : s-polarized reflection spectrum over wavelength range,\n
        'Rp' : p-polarized reflection spectrum over wavelength range }

        See Also
        -------------
        """

        # dictionary to store input args
        input_args = {'wv_range':None,
                    'substrate':None,
                    'h_mat':None,
                    'l_mat':None,
                    'layer_stack':None,
                    'materials':None,
                    'theta':None,
                    'sub_thick':None}

        # check if args list is correct length
        if len(args) == 8:
            # iterate input_args keys
            for i, k in enumerate(input_args):
                # define input_args key/value pairs
                input_args[k] = args[i]
        elif len(args) != 8:
            # raise ValueError exception if args len incorrect
            if len(args) > 8:
                raise ValueError("Too many input arguments. Expected 8 but received "
                                + str(len(args)))
            if len(args) < 8:
                raise ValueError("Not enough input arguments. Expected 8 but received "
                                + str(len(args)))

        if kwargs:
            if kwargs['units'] == 'deg':
                # convert incident angle from degrees to radians
                theta = theta * (np.pi / 180)

        # define dict to store filspec calculations
        spec = {}

        # dict to store refractive index values
        ind = {}

        # substrate refractive index
        ind['s_n'] = np.array(input_args['substrate']).astype(complex)
        # film refractive indices
        ind['f_n'] = np.zeros((len(input_args['layer_stack']),
                np.shape(input_args['wv_range'])[1])).astype(complex)
        # incident medium refractive index (assume incident medium is air)
        ind['i_n'] = np.ones(np.shape(input_args['wv_range'])).astype(complex)

        # get measured substrate & thin film optical constant data
        for i, mat in enumerate(input_args['materials']):
            if mat == "H":
                ind['f_n'][i, :] = np.array(input_args['h_mat']).astype(complex)
            else:
                ind['f_n'][i, :] = np.array(input_args['l_mat']).astype(complex)

        # calculate the effective substrate refractive index (for abs. substrates)
        sn_eff = ThinFilmFilter.sub_n_eff(ind['s_n'], input_args['theta'])

        # calculate the absorption coefficient for multiple reflections
        alpha = (4 * np.pi * np.imag(ind['s_n'])) / np.array(input_args['wv_range'])
        alpha = alpha.astype(complex)

        # Calculate the path length through the substrate
        p_len = ThinFilmFilter.path_len(input_args['sub_thick'], ind['i_n'],
                                        sn_eff, input_args['theta'])

        # Fresnel Amplitudes & Intensities for incident medium / substrate interface
        fr_bare = ThinFilmFilter.fresnel_bare(ind['i_n'], ind['s_n'], input_args['theta'])

        # reflection that originates from the incident medium
        i_ref = ThinFilmFilter.incident_reflection(input_args['wv_range'],
                            input_args['layer_stack'], ind['i_n'], ind['s_n'],
                            ind['f_n'], input_args['theta'])

        # reflection that originates from the substrate medium
        s_ref = ThinFilmFilter.substrate_reflection(input_args['wv_range'],
                            input_args['layer_stack'], ind['i_n'], ind['s_n'],
                            ind['f_n'], input_args['theta'], sn_eff)

        # calculate filter reflection
        spec['Rs'] = (i_ref['fresnel_film']['Rs']
                        + ((i_ref['fresnel_film']['Ts']**2)
                        * fr_bare['Rs'] * np.exp(-2 * alpha * p_len))
                        / (1 - (s_ref['fresnel_film']['Rs']
                        * fr_bare['Rs'] * np.exp(-2 * alpha * p_len))))
        spec['Rp'] = (i_ref['fresnel_film']['Rp']
                        + ((i_ref['fresnel_film']['Tp']**2)
                        * fr_bare['Rp'] * np.exp(-2 * alpha * p_len))
                        / (1 - (s_ref['fresnel_film']['Rp']
                        * fr_bare['Rp'] * np.exp(-2 * alpha * p_len))))
        spec['R'] = (spec['Rs'] + spec['Rp']) / 2

        # calculate filter transmission
        spec['Ts'] = ((i_ref['fresnel_film']['Ts']
                        * fr_bare['Ts'] * np.exp(-alpha * p_len))
                        / (1 - (s_ref['fresnel_film']['Rs']
                        * fr_bare['Rs'] * np.exp(-2 * alpha * p_len))))
        spec['Tp'] = ((i_ref['fresnel_film']['Tp']
                        * fr_bare['Tp'] * np.exp(-alpha * p_len))
                        / (1 - (s_ref['fresnel_film']['Rp']
                        * fr_bare['Rp'] * np.exp(-2 * alpha * p_len))))
        spec['T'] = (spec['Ts'] + spec['Tp']) / 2

        return spec

    @staticmethod
    def reg_vec(wv_range, opt_comp, transmission, reflection):
        """
        Calculates the optical computation regression vector.

        Parameters
        ----------
        wv_range (array):\n
        opt_comp (int):\n
        transmission (array):\n
        reflection (array):\n

        Returns
        ----------
        (array) computed regression vector

        """

        # initialize r_vec
        r_vec = []

        # calculation based on 'opt_comp' parameter
        if opt_comp in (0, 6):
            r_vec = transmission - reflection
        elif opt_comp in (1, 7):
            r_vec = ((transmission - reflection)
                    / (transmission + reflection))
        elif opt_comp in (2, 8):
            r_vec = transmission - .5 * reflection
        elif opt_comp in (3, 9):
            r_vec = (2 * transmission
                    - np.ones(np.shape(wv_range)[1]))
        elif opt_comp in (4, 5, 10, 11):
            r_vec = transmission

        # calculation result
        return np.array(r_vec)

    @staticmethod
    def roc_curve(truth, detections, thresh):
        """
        Processes a set of detections, ground truth values and a threshold array
        into a ROC curve.

        Parameters
        -----------
        truth (): logical index or binary values indicating class membership
                    detections - measured results (or prediction scores) from sensor\n
        detections (): measured results (or prediction scores) from sensor\n
        thresh (): array of user specified threshold values for computing the ROC curve

        Returns
        ---------
        dictionary object of transmission and reflection value arrays\n
        {key : value}\n
        { 'AUROC' : Area Under the Receiver Operator Curve,\n
        'Pd' : Probability of detection (or sensitivity),\n
        'Pfa' : Probability of false alarm (or 1 - specificity),\n
        't_ind' : index of optimal threshold,\n
        't_val' : Optimal threshold based on distance to origin,\n
        'Se' : Optimal sensitivity based upon optimal threshold,\n
        'Sp' : Optimal specificity based upon optimal threshold }
        """

        truth = np.array(truth)
        detections = np.array(detections)
        thresh = np.array(thresh)
        # define a dictionary containing function variable names
        # Detections | True Positives | True Negatives | False Positives | False Negatives
        # Probability of detection | Probability of false alarm
        roc_vals = {'detects':np.zeros((1, np.shape(truth)[1])),
                    'true_pos':np.zeros((1, len(thresh))),
                    'true_neg':np.zeros((1, len(thresh))),
                    'false_pos':np.zeros((1, len(thresh))),
                    'false_neg':np.zeros((1, len(thresh))),
                    'prob_det':np.zeros((1, len(thresh))),
                    'prob_fa':np.zeros((1, len(thresh)))}

        # Run loop to threshold the detections data and calculate TP, TN, FP & FN
        for i, val in enumerate(thresh):
            roc_vals['detects'][:, i] = detections[:, i] >= val
            roc_vals['true_pos'][:, i] = np.sum(np.sum(truth * roc_vals['detects']))
            roc_vals['true_neg'][:, i] = np.sum(np.sum((1 - truth) * (1 - roc_vals['detects'])))
            roc_vals['false_pos'][:, i] = np.sum(roc_vals['detects']) - roc_vals['true_pos'][:, i]
            roc_vals['false_neg'][:, i] = np.sum(truth) - roc_vals['true_pos'][:, i]

        # Calculate Pd and Pfa
        roc_vals['prob_det'] = (roc_vals['true_pos']
                            / (roc_vals['true_pos']
                            + roc_vals['false_neg']))
        roc_vals['prob_fa'] = (1 - (roc_vals['true_neg']
                            / (roc_vals['false_pos']
                            + roc_vals['true_neg'])))

        # Calculate the optimal threshold
        # Define a vector of multiple indices of the ROC curve origin (upperleft)
        orig = np.tile([0, 1], (np.shape(roc_vals['prob_fa'])[1], 1))

        # Calculate the Euclidian distance from each point to the origin
        euc_dist = np.sqrt(np.square(orig[:, 0] - np.squeeze(roc_vals['prob_fa'].T))
            + np.square(orig[:, 1] - np.squeeze(roc_vals['prob_det'].T)))

        # Find the best threshold index
        t_ind = np.argmin(euc_dist)

        # Find the best threshold value
        t_value = thresh[t_ind]

        # Calculate the optimal sensitivity and specificity
        sensitivity = roc_vals['prob_det'][0, t_ind]
        specificity = 1 - roc_vals['prob_fa'][0, t_ind]

        # Calculate the AUROC using a simple summation of rectangles
        au_roc = -np.trapz([1, roc_vals['prob_fa']], [roc_vals['prob_det'], 0])

        return {'AUROC':au_roc, 'Pd':roc_vals['prob_det'], 'Pfa':roc_vals['prob_fa'],
                't_val':t_value, 'Se':sensitivity, 'Sp':specificity}
