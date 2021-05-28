"""
The filters.py module contains common functions for calculating characteristics
of optical filters and thin film stacks. The Filters class contains static methods
that can be called without any class instance.
"""

# import external packages
import numpy as np

# import custom exceptions classes
from tff_lib.exceptions import UnitError

class ThinFilmFilter:

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
        if units != 'rad' and units != 'deg':
            # raise a custom 'UnitError' if 'units' not valid
            err_msg = "Invalid Units. Valid inputs are 'rad' or 'deg'."
            raise UnitError(units, err_msg)

        if type(units) != str:
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
            if type(arr) != np.ndarray:
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
        big_t_s = 1 - big_r_s
        big_t_p = 1 - big_r_p

        return {'Ts':big_t_s, 'Tp':big_t_p, 'Rs':big_r_s,
                'Rp':big_r_p, 'rs':r_s, 'rp':r_p}

    @staticmethod
    def admit_delta(wv_range, layer_stack, theta, i_n, s_n, f_n, units='deg'):
        """
        (static method) Calculates filter admittances of incident, substrate,
        and film as well as the phase (delta) upon reflection for each film.

        Parameters
        -------------
        wv_range (array): wavelength values in nanometers.\n
        layer_stack (array): thickness values of each filter layer in nanometers.\n
        theta (float): angle of incidence of radiation\n
        i_n (array): complex refractive index of incident medium (i_n = n + i*k)\n
        s_n (array): complex refractive index of substrate (s_n = n + i*k)\n
        f_n (array): complex refractive index of thin films (f_n = n + i*k)

        optional:\n
        units (str): units of measure for angle theta. Valid options are
                    'rad' and 'deg'. Default value is 'deg'.

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

        # check if 'units' param is valid
        if units != 'rad' or units != 'deg':
            # raise a custom 'UnitError' if 'units' not valid
            raise UnitError(units, "Invalid Units. Valid inputs are 'rad' or 'deg'.")

        elif type(units) != str:
            # raise a TypeError if 'units' is not a string
            raise TypeError("TypeError: 'units' parameter expects type 'str' but received "
                            + type(units))

        elif units == 'deg':
            # if input theta is 'deg', convert to radians
            theta = theta * (np.pi / 180)

        # validate i_n, s_n, and f_n input arrays they should be 'complex' or 'complex128'
        # data structure should be 'np.ndarray' and all shapes should match wv_range
        for arr in [i_n, s_n, f_n]:
            if arr.dtype != 'complex128' or arr.dtype != 'complex':
                # raise TypeError if any array is not complex
                raise TypeError("Incorrect type: expected 'complex' type but received "
                                 + str(arr.dtype))
            if type(arr) != np.ndarray:
                # raise TypeError if not a numpy ndarray
                raise TypeError("Bad Data Structure: Expected 'numpy.ndarray' but received "
                                + type(arr))

            # validate that shapes are equal to the wv_range shape
            if np.shape(arr) != np.shape(wv_range):
                raise ValueError("ValueError: Expected arrays of shape "
                                + str(np.shape(wv_range)) + " but received "
                                + str(np.shape(arr)))


        # Calculation of the complex dielectric constants from measured
        # optical constants
        i_e = np.square(i_n) # incident medium (air)
        s_e = np.square(s_n) # substrate
        f_e = np.square(f_n) # thin films

        # Calculation of the admittances of the incident and substrate media
        ns_inc = np.sqrt(i_e - i_e * np.sin(theta)**2)
        np_inc = i_e / ns_inc
        ns_sub = np.sqrt(s_e - i_e * np.sin(theta)**2)
        np_sub = s_e / ns_sub

        # Calculation of the admittances & phase factors
        # for each layer of the film stack
        ns_film = np.ones((len(layer_stack),
                np.shape(wv_range)[1])).astype(complex)
        np_film = np.ones((len(layer_stack),
                np.shape(wv_range)[1])).astype(complex)
        delta = np.ones((len(layer_stack),
                np.shape(wv_range)[1])).astype(complex)

        # enter loop if the substrate has thin film layers
        if len(f_n[:, 0]) >= 1:
            for i, layer in enumerate(layer_stack):
                ns_film[i, :] = np.sqrt(f_e[i, :] -
                                i_e * np.sin(theta)**2)
                np_film[i, :] = f_e[i, :] / ns_film[i, :]
                delta[i, :] = ((2 * np.pi * layer
                            * np.sqrt(f_e[i, :] - i_e
                            * np.sin(theta)**2))
                            / np.array(wv_range))


        # Flip layer-based arrays ns_film, np_film, delta
        # since the last layer is the top layer
        return {'ns_inc':ns_inc, 'np_inc':np_inc,
                'ns_sub':ns_sub, 'np_sub':np_sub,
                'ns_film':np.flipud(ns_film),
                'np_film':np.flipud(np_film),
                'delta':np.flipud(delta)}

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

        Examples
        ------------
        >>> char_matrix = c_mat(nsFilm, npFilm, delta)
        >>> ns_film = char_matrix[ 'nsFilm' ]
        >>> d = char_matrix[ 'delta' ]
        """

        # Calculation of the characteristic matrix elements
        s_11 = np.cos(delta)
        s_22 = np.cos(delta)
        p_11 = np.cos(delta)
        p_22 = np.cos(delta)
        s_12 = (1j / ns_film) * np.sin(delta)
        p_12 = (-1j / np_film) * np.sin(delta)
        s_21 = (1j * ns_film) * np.sin(delta)
        p_21 = (-1j * np_film) * np.sin(delta)

        # Initialize the characteristic matrices
        big_s_11 = np.ones((1, len(s_11[0,:]))).astype(complex)
        big_s_12 = np.zeros((1, len(s_11[0,:]))).astype(complex)
        big_s_21 = np.zeros((1, len(s_11[0,:]))).astype(complex)
        big_s_22 = np.ones((1, len(s_11[0,:]))).astype(complex)
        big_p_11 = np.ones((1, len(p_11[0,:]))).astype(complex)
        big_p_12 = np.zeros((1, len(p_11[0,:]))).astype(complex)
        big_p_21 = np.zeros((1, len(p_11[0,:]))).astype(complex)
        big_p_22 = np.ones((1, len(p_11[0,:]))).astype(complex)

        # Multiply all of the individual layer characteristic matrices together
        for i in range(len(s_11[:,0])):
            big_a_s = big_s_11
            big_b_s = big_s_12
            big_c_s = big_s_21
            big_d_s = big_s_22
            big_s_11 = big_a_s * s_11[i, :] + big_b_s * s_21[i, :]
            big_s_12 = big_a_s * s_12[i, :] + big_b_s * s_22[i, :]
            big_s_21 = big_c_s * s_11[i, :] + big_d_s * s_21[i, :]
            big_s_22 = big_c_s * s_12[i, :] + big_d_s * s_22[i, :]

        for i in range(len(p_11[:,0])):
            big_a_p = big_p_11
            big_b_p = big_p_12
            big_c_p = big_p_21
            big_d_p = big_p_22
            big_p_11 = big_a_p * p_11[i, :] + big_b_p * p_21[i, :]
            big_p_12 = big_a_p * p_12[i, :] + big_b_p * p_22[i, :]
            big_p_21 = big_c_p * p_11[i, :] + big_d_p * p_21[i, :]
            big_p_22 = big_c_p * p_12[i, :] + big_d_p * p_22[i, :]

        return {'S11':big_s_11, 'S12':big_s_12, 'S21':big_s_21, 'S22':big_s_22,
                'P11':big_p_11, 'P12':big_p_12, 'P21':big_p_21, 'P22':big_p_22}

    @staticmethod
    def fresnel_film(admit_delta_output, cmat_output):
        """
        Calculates the fresnel amplitudes & intensities of film. Requires
        output from admit_delta() and c_mat() methods.

        Parameters
        ------------
        admit_delta_output (dict): results of admit_delta() method\n
        cmat_output (dict): results of c_mat() method\n

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
        fresnel = fresnel_film(admit_delta_output, cmat_output)\n
        T_p = fresnel[ 'Tp' ]\n
        R_s = fresnel[ 'Rs' ]

        """

        # Calculation of the admittances of the incident interface
        ns_f = ((np.array(cmat_output['S21']).astype(complex)
                - np.array(admit_delta_output['ns_sub']).astype(complex)
                * np.array(cmat_output['S22']).astype(complex))
                / (np.array(cmat_output['S11']).astype(complex)
                - np.array(admit_delta_output['ns_sub']).astype(complex)
                * np.array(cmat_output['S12']).astype(complex)))
        np_f = ((np.array(cmat_output['P21']).astype(complex)
                + np.array(admit_delta_output['np_sub']).astype(complex)
                * np.array(cmat_output['P22']).astype(complex))
                / (np.array(cmat_output['P11']).astype(complex)
                + np.array(admit_delta_output['np_sub']).astype(complex)
                * np.array(cmat_output['P12']).astype(complex)))

        # Calculation of the Fresnel Amplitude Coefficients
        r_s = ((np.array(admit_delta_output['ns_inc']).astype(complex) + ns_f)
            / (np.array(admit_delta_output['ns_inc']).astype(complex) - ns_f))
        r_p = ((np.array(admit_delta_output['np_inc']).astype(complex) - np_f)
            / (np.array(admit_delta_output['np_inc']).astype(complex) + np_f))
        t_s = ((1 + r_s)
            / (np.array(cmat_output['S11']).astype(complex)
            - np.array(cmat_output['S12']).astype(complex)
            * np.array(admit_delta_output['ns_sub']).astype(complex)))
        t_p = ((1 + r_p)
            / (np.array(cmat_output['P11']).astype(complex)
            + np.array(cmat_output['P12']).astype(complex)
            * np.array(admit_delta_output['np_sub']).astype(complex)))

        # Calculation of the Fresnel Amplitude Intensities
        big_r_s = np.square(np.abs(r_s))
        big_r_p = np.square(np.abs(r_p))
        big_t_s = (np.real(np.array(admit_delta_output['ns_sub']).astype(complex)
            / np.array(admit_delta_output['ns_inc']).astype(complex)) * np.square(np.abs(t_s)))
        big_t_p = (np.real(np.array(admit_delta_output['np_sub']).astype(complex)
            / np.array(admit_delta_output['np_inc']).astype(complex)) * np.square(np.abs(t_p)))

        return {'Ts':big_t_s, 'Tp':big_t_p, 'Rs':big_r_s, 'Rp':big_r_p,
                'ts':t_s, 'tp':t_p, 'rs':r_s, 'rp':r_p}

    @staticmethod
    def fil_spec(wv_range, substrate, h_mat, l_mat, layer_stack,
                        materials, theta, sub_thick, units='deg'):
        """
        Calculates the transmission and reflection spectra of a thin-film interference filter.

        Parameters
        ------------
        wv_range
        substrate
        h_mat
        l_mat
        layer_stack
        materials
        theta
        sub_thick
        units='deg'

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

        # substrate refractive index
        s_n = np.array(substrate).astype(complex)
        # film refractive indices
        f_n = np.zeros((len(layer_stack),
                np.shape(wv_range)[1])).astype(complex)
        # incident medium refractive index (assume incident medium is air)
        i_n = np.ones(np.shape(wv_range)).astype(complex)

        # convert incident angle from degrees to radians
        theta = theta * (np.pi / 180)

        # get measured substrate & thin film optical constant data
        for i in range(0, len(materials)):
            if materials[i] == "H":
                f_n[i,:] = h_mat
            else:
                f_n[i,:] = l_mat

        # calculate the effective substrate refractive index (for abs. substrates)
        inside1 = (np.imag(s_n)**2 + np.real(s_n)**2)**2 + 2*(np.imag(s_n) - \
            np.real(s_n))*(np.imag(s_n) + np.real(s_n))*np.sin(theta)**2 + \
            np.sin(theta)**4
        inside1 = np.array(inside1,dtype=np.float64)
        inside2 = 0.5*(-np.imag(s_n)**2 + np.real(s_n)**2 + np.sin(theta)**2 + \
            np.sqrt(inside1))
        inside2 = np.array(inside2,dtype=np.float64)
        sn_eff = np.sqrt(inside2)

        # calculate the absorption coefficient for multiple reflections
        alpha = (4 * np.pi * np.imag(s_n)) / np.array(wv_range)
        alpha = alpha.astype(complex)

        # Calculate the path length through the substrate
        path_len = ((sub_thick * (10**6))
                / np.sqrt(1 - (np.abs(i_n)**2
                * (np.sin(theta)**2) / sn_eff**2)))
        path_len = path_len.astype(complex)

        # Calculation of Fresnel Amplitudes & Intensities for
        # incident medium / substrate interface
        fb_out = fresnel_bare(i_n, s_n, theta)

        # the reflection originates from the incident medium
        admit_a = admit_delta(i_n, s_n, f_n, theta)
        cmat_a = c_mat(admit_a['ns_film'], admit_a['np_film'], admit_a['delta'])
        ff_a = fresnel_film(admit_a, cmat_a)

        # the reflection originates from the substrate medium
        theta_b = np.arcsin(i_n / s_n * np.sin(theta))
        admit_b = admit_delta(sn_eff, i_n, np.flipud(f_n), theta_b)
        cmat_b = c_mat(admit_b['ns_film'], admit_b['np_film'], admit_b['delta'])
        ff_1b = fresnel_film(admit_b, cmat_b)

        # calculation of the filter transmission and reflection
        big_r_s = ff_a['Rs'] + (((ff_a['Ts']**2) * fb_out['Rs'] * np.exp(-2 * alpha * path_len))
            / (1 - (ff_1b['Rs'] * fb_out['Rs'] * np.exp(-2 * alpha * path_len))))
        big_r_p = ff_a['Rp'] + (((ff_a['Tp']**2) * fb_out['Rp'] * np.exp(-2 * alpha * path_len))
            / (1 - (ff_1b['Rp'] * fb_out['Rp'] * np.exp(-2 * alpha * path_len))))
        big_r = (big_r_s + big_r_p) / 2
        big_t_s = ((ff_a['Ts'] * fb_out['Ts'] * np.exp(-alpha * path_len))
            / (1 - (ff_1b['Rs'] * fb_out['Rs'] * np.exp(-2 * alpha * path_len))))
        big_t_p = ((ff_a['Tp'] * fb_out['Tp'] * np.exp(-alpha * path_len))
            / (1 - (ff_1b['Rp'] * fb_out['Rp'] * np.exp(-2 * alpha * path_len))))
        big_t = (big_t_s + big_t_p) / 2

        return {'T':big_t, 'Ts':big_t_s, 'Tp':big_t_p, 'R':big_r, 'Rs':big_r_s, 'Rp':big_r_p}

    @staticmethod
    def reg_vec(opt_comp, big_t, big_r, wv_range):
        """
        Calculates the optical computation regression vector.

        Parameters
        ----------
        big_t (array):\n
        big_r (array):\n

        Returns
        ----------
        computed regression vector as an array

        """

        # initialize r_vec
        r_vec = []

        # calculation based on 'opt_comp' design setting
        if opt_comp == 0:
            r_vec = big_t - big_r
        elif opt_comp == 1:
            r_vec = (big_t - big_r) / (big_t + big_r)
        elif opt_comp == 2:
            r_vec = big_t - .5 * big_r
        elif opt_comp == 3:
            r_vec = 2 * big_t - np.ones(np.shape(wv_range)[1])
        elif opt_comp == 4:
            r_vec = big_t
        elif opt_comp == 5:
            r_vec = big_t
        elif opt_comp == 6:
            r_vec = big_t - big_r
        elif opt_comp == 7:
            r_vec = (big_t - big_r) / (big_t + big_r)
        elif opt_comp == 8:
            r_vec = big_t - .5 * big_r
        elif opt_comp == 9:
            r_vec = 2 * big_t - np.ones(np.shape(wv_range)[1])
        elif opt_comp == 10:
            r_vec = big_t
        elif opt_comp == 11:
            r_vec = big_t

        # calculation result
        return r_vec

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
