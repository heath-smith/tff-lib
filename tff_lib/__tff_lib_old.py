"""
The filters.py module contains common functions for calculating characteristics
of optical filters and thin film stacks. The Filters class contains static methods
that can be called without any class instance.

To install via pip:

>>> python -m pip install git+<url>
"""

# import external packages
from numpy import testing as nptest
import numpy as np
from typing import Union
from tff_lib import __utils_old as utils

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
    sub = sub.astype(np.complex128) if not sub.dtype == np.complex128 else sub
    med = med.astype(np.complex128) if not med.dtype == np.complex128 else med

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
    layers:list,
    waves:np.ndarray,
    sub:np.ndarray,
    med:np.ndarray,
    films:np.ndarray,
    theta:Union[int, float, np.ndarray],
    units:str='rad') -> dict:
    """
    Calculates filter admittances of incident, substrate,
    and film as well as the phase (delta) upon reflection for each film.

    Parameters
    -------------
    layers = thickness values of each filter layer in nanometers.\n
    waves = wavelength values in nanometers.\n
    sub = complex refractive index of substrate\n
    med = complex refractive index of incident medium\n
    films = 2-D array of complex refractive index of thin films\n
    theta = angle of incidence of radiation (can be number or array)\n
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

    # convert refractive indices to complex, if needed,
    # and calculate complex dialectric constants (square the values)
    sub = sub.astype(np.complex128)**2 if not sub.dtype == np.complex128 else sub**2
    med = med.astype(np.complex128)**2 if not med.dtype == np.complex128 else med**2
    films = films.astype(np.complex128)**2 if not films.dtype == np.complex128 else films**2

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
    admit['ns_film'] = np.ones((len(layers), len(waves))).astype(np.complex128)
    admit['np_film'] = np.ones((len(layers), len(waves))).astype(np.complex128)
    admit['delta'] = np.ones((len(layers), len(waves))).astype(np.complex128)

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


def char_matrix(ns_film:np.ndarray, np_film:np.ndarray, delta:np.ndarray) -> dict:
    """
    Calculates the characteristic matrix for multiple thin film layers.

    Parameters
    -----------
    ns_film = s-polarized admittance of the film stack layers.\n
    np_film = p-polarized admittance of the film stack layers.\n
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
    """

    # validate input types
    if not isinstance(ns_film, np.ndarray):
        raise TypeError(f'"s_adm" expects <np.ndarray>, received {type(ns_film)}.')
    if not isinstance(np_film, np.ndarray):
        raise TypeError(f'"p_adm" expects <np.ndarray>, received {type(np_film)}.')
    if not isinstance(delta, np.ndarray):
        raise TypeError(f'"delta" expects <np.ndarray>, received {type(delta)}.')

    # validate the input array shapes
    if not ns_film.shape == np_film.shape:
        raise ValueError(f'shape mismatch -----> {ns_film.shape} != {np_film.shape}.')

    # Calculation of the characteristic matrix elements
    # shape of 'delta' is (N-layers X len(wavelength range))
    elements = {'s11': np.cos(delta),
                's22': np.cos(delta),
                'p11': np.cos(delta),
                'p22': np.cos(delta),
                's12': (1j / ns_film) * np.sin(delta),
                'p12': (-1j / np_film) * np.sin(delta),
                's21': (1j * ns_film) * np.sin(delta),
                'p21': (-1j * np_film) * np.sin(delta)}

    # Initialize the characteristic matrices
    char_mat = {'S11':np.ones(np.shape(elements['s11'])[1]).astype(np.complex128),
                'S12':np.zeros(np.shape(elements['s11'])[1]).astype(np.complex128),
                'S21':np.zeros(np.shape(elements['s11'])[1]).astype(np.complex128),
                'S22':np.ones(np.shape(elements['s11'])[1]).astype(np.complex128),
                'P11':np.ones(np.shape(elements['p11'])[1]).astype(np.complex128),
                'P12':np.zeros(np.shape(elements['p11'])[1]).astype(np.complex128),
                'P21':np.zeros(np.shape(elements['p11'])[1]).astype(np.complex128),
                'P22':np.ones(np.shape(elements['p11'])[1]).astype(np.complex128)}

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


def fresnel_film(admittance:dict, char_matrix:dict) -> dict:
    """
    Calculates the fresnel amplitudes & intensities of film. Requires
    output from admit_delta() and c_mat() methods.

    Parameters
    ------------
    admittance = results of admit_delta() method\n
    char_matrix = results of char_matrix() method

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
    ValueError, TypeError

    Examples
    --------------
    >>> fresnel = fresnel_film(admittance, char_matrix)\n
    >>> T_p = fresnel[ 'Tp' ]\n
    >>> R_s = fresnel[ 'Rs' ]

    See Also
    ---------------
    >>> tff_lib.admit_delta()
    >>> tff_lib.char_matrix()
    """

    # Calculation of the admittances of the incident interface
    ns_f = ((char_matrix['S21'] - admittance['ns_sub'] * char_matrix['S22'])
        / (char_matrix['S11'] - admittance['ns_sub'] * char_matrix['S12']))
    np_f = ((char_matrix['P21'] + admittance['np_sub'] * char_matrix['P22'])
        / (char_matrix['P11'] + admittance['np_sub'] * char_matrix['P12']))

    # Calculation of the Fresnel Amplitude Coefficients
    fresnel = {
        'rs': (admittance['ns_inc'] + ns_f) / (admittance['ns_inc'] - ns_f),
        'rp': (admittance['np_inc'] - np_f) / (admittance['np_inc'] + np_f)
    }
    fresnel['ts'] = ((1 + fresnel['rs'])
                    / (char_matrix['S11'] - char_matrix['S12']  * admittance['ns_sub']))
    fresnel['tp'] = ((1 + fresnel['rp'])
                    / (char_matrix['P11'] + char_matrix['P12'] * admittance['np_sub']))

    # Calculation of the Fresnel Amplitude Intensities
    fresnel['Rs'] = np.abs(fresnel['rs'])**2
    fresnel['Rp'] = np.abs(fresnel['rp'])**2
    fresnel['Ts'] = np.real(admittance['ns_sub'] / admittance['ns_inc']) * np.abs(fresnel['ts'])**2
    fresnel['Tp'] = np.real(admittance['np_sub'] / admittance['np_inc']) * np.abs(fresnel['tp'])**2

    return fresnel


def fil_spec(
    waves:np.ndarray, sub:np.ndarray, med:np.ndarray, films:np.ndarray, layers:list,
    sub_thick:Union[int, float], theta:Union[int, float], units:str='rad') -> dict:
    """
    Calculates the transmission and reflection spectra of a
    thin-film interference filter.

    Parameters
    ------------
    waves =
    sub =
    med =
    high =
    low =
    films =
    layers =
    sub_thick =
    theta =
    units =

    Returns
    -----------
    (dict) transmission and reflection value arrays\n
    { 'T' : average transmission spectrum over wavelength range ([Tp + Ts] / 2),\n
    'Ts' : s-polarized transmission spectrum wavelength range,\n
    'Tp' : p-polarized transmission spectrum over wavelength range,\n
    'R' : average reflection spectrum over wavelength range,\n
    'Rs' : s-polarized reflection spectrum over wavelength range,\n
    'Rp' : p-polarized reflection spectrum over wavelength range }

    Raises
    -------------
    ValueError, TypeError

    See Also
    -------------
    utils.sub_n_eff()
    utils.path_len()
    utils.film_matrix()
    """

    # convert incident angle from degrees to radians
    theta = theta * (np.pi / 180) if units == 'deg' else theta

    # calculate effective substrate refractive index
    sn_eff = utils.effective_index(sub, theta)

    # calculate the absorption coefficient for multiple reflections
    alpha = (4 * np.pi * np.imag(sub)) / waves

    # Calculate the path length through the substrate
    p_len = utils.path_length(sub_thick, med, sn_eff, theta)

    # Fresnel Amplitudes & Intensities
    # incident medium / substrate interface
    fr_bare = fresnel_bare(sub, med, theta)

    # reflection originates from incident medium
    med_adm = admit_delta(layers, waves, sub, med, films, theta)
    med_char = char_matrix(med_adm['ns_film'], med_adm['np_film'], med_adm['delta'])
    med_ref = fresnel_film(med_adm, med_char)

    # reflection originates from substrate
    theta_inv = np.arcsin(med / sub * np.sin(theta))
    layers_inv = [(str(v[0]), float(v[1])) for v in np.flipud(layers)]
    sub_adm = admit_delta(layers_inv, waves, sn_eff, med, np.flipud(films), theta_inv)
    sub_char = char_matrix(sub_adm['ns_film'], sub_adm['np_film'], np.flipud(sub_adm['delta']))
    sub_ref = fresnel_film(sub_adm, sub_char)

    nptest.assert_almost_equal(sub_adm['np_sub'], med_adm['np_sub'], decimal=13, verbose=2)

    nptest.assert_almost_equal(med_adm['ns_film'], sub_adm['ns_film'], decimal=13, verbose=2)
    nptest.assert_almost_equal(med_adm['np_film'], sub_adm['np_film'], decimal=13, verbose=2)

    # calculate filter reflection
    spec = {'Rs': (
        med_ref['Rs'] + ((med_ref['Ts']**2) * fr_bare['Rs'] * np.exp(-2 * alpha * p_len))
        / (1 - (sub_ref['Rs'] * fr_bare['Rs'] * np.exp(-2 * alpha * p_len)))
    )}
    spec['Rp'] = (
        med_ref['Rp']  + ((med_ref['Tp']**2)  * fr_bare['Rp'] * np.exp(-2 * alpha * p_len))
        / (1 - (sub_ref['Rp']  * fr_bare['Rp'] * np.exp(-2 * alpha * p_len)))
    )
    spec['R'] = (spec['Rs'] + spec['Rp']) / 2

    # calculate filter transmission
    spec['Ts'] = (
        (med_ref['Ts'] * fr_bare['Ts'] * np.exp(-alpha * p_len))
        / (1 - (sub_ref['Rs'] * fr_bare['Rs'] * np.exp(-2 * alpha * p_len)))
    )
    spec['Tp'] = (
        (med_ref['Tp'] * fr_bare['Tp'] * np.exp(-alpha * p_len))
        / (1 - (sub_ref['Rp'] * fr_bare['Rp'] * np.exp(-2 * alpha * p_len)))
    )
    spec['T'] = (spec['Ts'] + spec['Tp']) / 2

    return spec

