"""
The filters.py module contains common functions for calculating characteristics
of optical filters and thin film stacks. The Filters class contains static methods
that can be called without any class instance.

To install via pip:

>>> python -m pip install git+<url>
"""

# import external packages
import numpy as np
from typing import Union
import sys

def fresnel_bare(
    sub:np.ndarray, med:np.ndarray, theta:Union[int, float], units:str='rad') -> dict:
    """
    Calculates the fresnel amplitudes & intensities of the bare substrate.

    Parameters
    -----------
    sub = complex refractive index of substrate (s_n = x+y*j).\n
    med = complex refractive index on incident medium (i_n = x+y*j).\n
    theta = angle of incidence of radiation.\n
    units = 'deg' or 'rad'. Default is 'rad' if not specified.

    Returns
    -----------
    (dict)  transmission and reflection value arrays\n
    { 'Ts' : s-polarized Fresnel Transmission Intensity,\n
    'Tp' : p-polarized Fresnel Transmission Intensity,\n
    'Rs' : s-polarized Fresnel Reflection Intensity,\n
    'Rp' : p-polarized Fresnel Reflection Intensity,\n
    'rs' : s-polarized Fresnel Reflection Amplitude,\n
    'rp' : p-polarized Fresnel Reflection Amplitude }

    Raises
    -----------
    ValueError, TypeError

    Examples
    --------------
    >>> fb = fresnel_bare(i_n, s_n, theta, units='rad')
    >>> big_t_s = fr_bare['Ts']
    >>> big_r_s = fr_bare['Rs']
    """

    # validate input data
    if not isinstance(sub, np.ndarray):
        raise TypeError(f'"sub" expects <np.ndarray>, received{type(sub)}.')
    if not isinstance(med, np.ndarray):
        raise TypeError(f'"med" expects <np.ndarray>, received{type(med)}.')
    if not type(theta) in (int, float):
        raise TypeError(f'"theta" expects <int> or <float>, received {type(theta)}.')
    if not isinstance(units, str):
        raise TypeError(f'"units" expects <str> but received {type(units)}.')
    if units not in ('rad', 'deg'):
        raise ValueError(f'valid units are "rad" or "deg". received {units}.')
    if not sub.shape == med.shape:
        raise ValueError(f'shape mismatch -----> {sub.shape} != {med.shape}.')

    # convert arrays to complex, if needed
    sub = sub.astype(np.complex) if not sub.dtype == np.complex else sub
    med = med.astype(np.complex) if not med.dtype == np.complex else med

    # convert to radians
    theta = theta * (np.pi / 180) if units == 'deg' else theta

    # Calculation of the Fresnel Amplitude Coefficients
    rs_num = (med * np.cos(theta)) - np.sqrt(sub**2 - med**2 * np.sin(theta)**2)
    rs_den = (med * np.cos(theta)) + np.sqrt(sub**2 - med**2 * np.sin(theta)**2)
    r_s = rs_num / rs_den
    rp_num = sub**2 * np.cos(theta) - med * np.sqrt(sub**2 - med**2 * np.sin(theta)**2)
    rp_den = sub**2 * np.cos(theta) + med * np.sqrt(sub**2 - med**2 * np.sin(theta)**2)
    r_p = -rp_num / rp_den

    # Calculation of Fresnel Intensities for bare substrate interface
    big_r_s = np.abs(r_s)**2
    big_r_p = np.abs(r_p)**2

    return {
        'Ts': (1 - big_r_s),     # S-polarized transmission
        'Tp': (1 - big_r_p),     # P-polarized transmission
        'Rs': big_r_s,           # S-polarized reflection
        'Rp': big_r_p,           # P-polarized reflection
        'rs': r_s,               # S-polarized fresnel Amplitude Coefficient
        'rp': r_p                # P-polarized fresnel Amplitude Coefficient
    }


def admit_delta(
    layers:list, waves:np.ndarray, sub:np.ndarray,
    med:np.ndarray, films:np.ndarray, theta:Union[int, float], units:str='rad') -> dict:
    """
    Calculates filter admittances of incident, substrate,
    and film as well as the phase (delta) upon reflection for each film.

    Parameters
    -------------
    layers = thickness values of each filter layer in nanometers.\n
    waves = wavelength values in nanometers.\n
    sub = complex refractive index of substrate\n
    med = complex refractive index of incident medium\n
    films = complex refractive index of thin films\n
    theta = angle of incidence of radiation\n
    units = units of measure for angle theta. Valid options are 'rad'
            and 'deg'. Default value is 'rad'.

    Returns
    --------------
    (dict) transmission and reflection calculation results\n
    { 'ns_inc' : s-polarized admittance of the incident medium,\n
    'np_inc' : p-polarized admittance of the incident medium,\n
    'ns_sub' : s-polarized admittance of the substrate medium,\n
    'np_sub' : p-polarized admittance of the substrate medium,\n
    'ns_film' : s-polarized admittance of the film stack layers,\n
    'np_film' : p-polarized admittance of the film stack layers,\n
    'delta' : phase upon reflection for each film }

    Raises
    ------------
    TypeError, ValueError

    Examples
    --------------
    >>> adm = admit_delta(waves, layers, med, sub, films, theta, units='deg')
    >>> ns_inc = adm['ns_inc']
    >>> delta = adm['delta']
    """

    # validate the input data
    if not isinstance(units, str):
        raise TypeError(f'"units" expects <str>, received {type(units)}.')
    if units not in ('rad', 'deg'):
        raise ValueError(f'"units" expects "rad" or "deg", received "{units}"')
    if not type(theta) in (float, int):
        raise TypeError(f'"theta" expects <int> or <float>, received {type(theta)}.')
    if not isinstance(layers, list):
        raise TypeError(f'"layers" expects <list>, received {type(layers)}.')
    # check the arrays for consistency
    for arr in (waves, med, sub, films):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'-----> expected <np.ndarray>, received {type(arr)}.')
    if not sub.shape == med.shape:
        raise ValueError(f'"sub" != "waves" ----> {sub.shape} != {med.shape}')
    if not med.shape == waves.shape:
        raise ValueError(f'"med" != "waves" ----> {med.shape} != {waves.shape}')
    if not films.shape == (len(layers), len(waves)):
        raise ValueError(
            f'"films" shape mismatch ----> {films.shape} != ({len(layers)}, {len(waves)}).')

    # convert refractive indices to complex, if needed,
    # and calculate complex dialectric constants (square the values)
    sub = sub.astype(np.complex)**2 if not sub.dtype == np.complex else sub**2
    med = med.astype(np.complex)**2 if not med.dtype == np.complex else med**2
    films = films.astype(np.complex)**2 if not films.dtype == np.complex else films**2

    # convert theta to radians, if needed, ensure positive
    theta = theta * (np.pi / 180) if units == 'deg' else theta

    # initialize dictionary to store admit_delta calculations
    admit = {}

    # Calculate admittances of the incident and substrate media
    admit['ns_inc'] = np.sqrt(med - med * np.sin(theta)**2)
    admit['np_inc'] = med / admit['ns_inc']
    admit['ns_sub'] = np.sqrt(sub - med * np.sin(theta)**2)
    admit['np_sub'] = sub / admit['ns_sub']

    # Calculate admittances & phase factors for each layer
    admit['ns_film'] = np.ones((len(layers), len(waves))).astype(complex)
    admit['np_film'] = np.ones((len(layers), len(waves))).astype(complex)
    admit['delta'] = np.ones((len(layers), len(waves))).astype(complex)

    # iterate each layer in thin film stack
    for i, lyr in enumerate(layers):
        admit['ns_film'][i, :] = np.sqrt(films[i, :] - med * np.sin(theta)**2)
        admit['np_film'][i, :] = films[i, :] / admit['ns_film'][i, :]
        admit['delta'][i, :] = (2 * np.pi * lyr[1] * np.sqrt(films[i, :] - med * np.sin(theta)**2)) / waves

    # Flip layer-based arrays ns_film, np_film, delta
    # since the last layer is the top layer
    admit['ns_film'] = np.flipud(admit['ns_film'])
    admit['np_film'] = np.flipud(admit['np_film'])
    admit['delta'] = np.flipud(admit['delta'])

    return admit


def char_mat(nsf:np.ndarray, npf:np.ndarray, delta:np.ndarray) -> dict:
    """
    Calculates the characteristic matrix for multiple thin film layers.

    Parameters
    -----------
    nsf = s-polarized admittance of the film stack layers.\n
    npf = p-polarized admittance of the film stack layers.\n
    delta = phase upon reflection for each film.

    Returns
    ------------
    (dict) characteristic matrix\n
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
    for arr in [nsf, npf, delta]:
        # if np.ndarray's, check the data type
        if isinstance(arr, np.ndarray):
            if arr.dtype not in ('float64', 'complex', 'complex128'):
                # raise TypeError if any array is not complex
                raise TypeError("Expected 'float64', 'complex', or 'complex128'"
                                + " type but received "
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
                's12': (1j / nsf) * np.sin(delta),
                'p12': (-1j / npf) * np.sin(delta),
                's21': (1j * nsf) * np.sin(delta),
                'p21': (-1j * npf) * np.sin(delta)}

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
        temp_a = char_mat['S11']
        temp_b = char_mat['S12']
        temp_c = char_mat['S21']
        temp_d = char_mat['S22']
        char_mat['S11'] = (temp_a * elements['s11'][i, :] + temp_b * elements['s21'][i, :])
        char_mat['S12'] = (temp_a * elements['s12'][i, :] + temp_b * elements['s22'][i, :])
        char_mat['S21'] = (temp_c * elements['s11'][i, :] + temp_d * elements['s21'][i, :])
        char_mat['S22'] = (temp_c * elements['s12'][i, :] + temp_d * elements['s22'][i, :])

    for i in range(np.shape(elements['p11'])[0]):
        temp_a = char_mat['P11']
        temp_b = char_mat['P12']
        temp_c = char_mat['P21']
        temp_d = char_mat['P22']
        char_mat['P11'] = (temp_a * elements['p11'][i, :] + temp_b * elements['p21'][i, :])
        char_mat['P12'] = (temp_a * elements['p12'][i, :] + temp_b * elements['p22'][i, :])
        char_mat['P21'] = (temp_c * elements['p11'][i, :] + temp_d * elements['p21'][i, :])
        char_mat['P22'] = (temp_c * elements['p12'][i, :] + temp_d * elements['p22'][i, :])

    return char_mat


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


def sub_n_eff(s_n, theta, units='rad'):
    """
    Calculates the effective substrate refractive index for abs. substrates.

    Parameters
    -----------
    *args:\n
    s_n (array):\n
    theta (float):

    **kwargs: (optional)\n
    units (str):

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


def path_len(sub_thick, i_n, sn_eff, theta, units='rad'):
    """
    Calculates the optical path length through a given substrate.

    Parameters
    -------------
    *args:\n
    sub_thick (float):\n
    theta (float):

    **kwargs: (optional)\n
    units (str):

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


def incident_reflection(*args, **kwargs):
    """
    Calculates the reflection that originates from the incident medium.

    Parameters
    -----------
    *args:\n

    **kwargs: (optional)\n
    units (str):

    Returns
    -----------
    dictionary object of transmission and reflection value arrays\n
    {key : result array}\n
    { }
    """

    # dictionary to store input args
    input_args = {'wv_range':[],
                'layers':[],
                'i_n':[],
                's_n':[],
                'f_n':[],
                'theta':[] }

    # check if args list is correct length
    if len(args) == 6:
        # iterate input_args keys
        for i, k in enumerate(input_args):
            # define input_args key/value pairs
            input_args[k] = args[i]
    # raise ValueError exception if args len incorrect
    elif len(args) != 6:
        raise ValueError(" Incorrect number of args. Expected 6 but received "
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
            # convert incident angle from degrees to radians
            input_args['theta'] = input_args['theta'] * (np.pi / 180)

    inc_ref = {}

    # calculate admittances
    inc_ref['admit_delta'] = ThinFilmFilter.admit_delta(input_args['wv_range'],
                                                        input_args['layers'],
                                                        input_args['theta'],
                                                        input_args['i_n'],
                                                        input_args['s_n'],
                                                        input_args['f_n'])

    # calculate the characteristic matrix
    inc_ref['c_mat'] = ThinFilmFilter.c_mat(
                        np.array(inc_ref['admit_delta']['ns_film']),
                        np.array(inc_ref['admit_delta']['np_film']),
                        np.array(inc_ref['admit_delta']['delta']))

    # calculate fresnel intensities and amplitudes
    inc_ref['fresnel_film'] = ThinFilmFilter.fresnel_film(inc_ref['admit_delta'],
                                                        inc_ref['c_mat'])

    return inc_ref


def substrate_reflection(*args, **kwargs):
    """
    Calculates the reflection that originates from the substrate medium.

    Parameters
    ----------
    *args:\n

    **kwargs: (optional)\n


    Returns
    ----------
    nested dictionary object with admit_delta() results and
    the results from c_mat().\n
    {key : result array}\n
    { }

    See Also
    ----------
    admit_delta()
    c_mat()
    """

    # dictionary to store input args
    input_args = {'wv_range':[],
                'layers':[],
                'i_n':[],
                's_n':[],
                'f_n':[],
                'theta':[],
                'sn_eff':[]}

    # check if args list is correct length
    if len(args) == 7:
        # iterate input_args keys
        for i, k in enumerate(input_args):
            # define input_args key/value pairs
            input_args[k] = args[i]
    # raise ValueError exception if args len incorrect
    elif len(args) != 7:
        raise ValueError(" Incorrect number of args. Expected 7 but received "
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
            # convert incident angle from degrees to radians
            input_args['theta'] = input_args['theta'] * (np.pi / 180)

    sub_ref = {}

    # angle of substrate reflectance
    input_args['theta'] = np.arcsin(input_args['i_n']
                            / input_args['s_n']
                            * np.sin(input_args['theta']))

    # calculate admittances
    sub_ref['admit_delta'] = ThinFilmFilter.admit_delta(input_args['wv_range'],
                                                        input_args['layers'],
                                                        input_args['theta'],
                                                        input_args['sn_eff'],
                                                        input_args['i_n'],
                                                        np.flipud(input_args['f_n']))

    # calculate characteristic matrix
    sub_ref['c_mat'] = ThinFilmFilter.c_mat(
                        np.array(sub_ref['admit_delta']['ns_film']),
                        np.array(sub_ref['admit_delta']['np_film']),
                        np.array(sub_ref['admit_delta']['delta']))

    # calculate fresnel intensities & amplitudes
    sub_ref['fresnel_film'] = ThinFilmFilter.fresnel_film(sub_ref['admit_delta'],
                                                        sub_ref['c_mat'])

    return sub_ref


def fil_spec(*args, **kwargs):
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
    sub_n_eff(*args, **kwargs)
    path_len(*args, **kwargs)
    incident_reflection(*args, **kwargs)
    substrate_reflection(*args, **kwargs)
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
    # raise ValueError exception if args len incorrect
    if len(args) > 8:
        raise ValueError("Too many input arguments. Expected 8 but received "
                        + str(len(args)))
    if len(args) < 8:
        raise ValueError("Not enough input arguments. Expected 8 but received "
                        + str(len(args)))
    if len(args) == 8:
        # iterate input_args keys
        for i, k in enumerate(input_args):
            # define input_args key/value pairs
            input_args[k] = args[i]

    # parse kwargs dictionary
    if kwargs:
        if kwargs['units'] == 'deg':
            # convert incident angle from degrees to radians
            input_args['theta'] = input_args['theta'] * (np.pi / 180)

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

