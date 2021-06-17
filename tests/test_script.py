#!user/bin/python
# -*- coding: utf-8 -*-

"""
This module 'test_script.py' contains a test script
demonstrating the common usage of the tff_lib package.

Example Usage
-------------
python -m tests.test_script

python test_script.py
"""

# import tff_lib for testing
from tff_lib.tff_lib import ThinFilmFilter as tff

# import pandas to read from csv
import pandas as pd

# import numpy
import numpy as np

# import matplotlib
from matplotlib import pyplot as plt


def convert_to_complex(arr):
    """
    Converts arrays from spreadsheet into
    complex type and row-based shape.

    Parameters
    ---------
    arr (list_like): input array from spreadsheet.

    Returns
    ---------
    Row-based, complex type numpy.ndarray.
    """
    # convert to strings
    arr = [str(x) for x in arr]
    # remove spaces from string values
    arr = [x.replace(' ', '') for x in arr]
    # replace i's with j's for python compatibility
    arr = [x.replace('i','j') for x in arr]
    # convert to complex type numpy array
    arr = np.array(arr).astype(complex)
    # reshape to row-based array
    arr = np.reshape(arr, (1, len(arr)))
    # return array
    return arr


def main():
    """
    Main script method, called when file is executed.
    """

    # define a path to a file
    file_path = r'E:\source_code\devops\designSuite\data\finished\test.xls'

    # read first sheet into dataframe
    df1 = pd.read_excel(file_path, sheet_name='DesignSuite Parameters')

    # read second sheet into dataframe
    df2 = pd.read_excel(file_path, sheet_name='MOE Design Data')

    # read in substrate thickness from first dataframe
    sub_thick = float(df1.iloc[7, 1])
    # define a value for incident angle theta
    theta = 0

    # define layers and materials
    layers = [2796.3, 470.8, 480.3, 1099.7, 1659, 369.4, 1601.6, 1585.9, 2271.7]
    materials = ["H", "L", "H", "L", "H", "L", "H", "L", "H"]

    # read wavelength from df2
    wv = df2['Wavelength (nm)'].dropna().values
    # reshape wv to match row-based shape
    wv = np.reshape(wv, (1, len(wv)))

    # read substrate, high/low material, and environmental interference
    # arrays from df2 and convert to row-based, complex typed numpy arrays
    substrate = convert_to_complex(df2['Substrate (n+ik)'].dropna().values)
    high_mat = convert_to_complex(df2['High Material (n + ik)'].dropna().values)
    low_mat = convert_to_complex(df2['Low Material (n+ik)'].dropna().values)
    env_int = df2['Env. Interference (20 g/m3 Abs. Humidity)'].dropna().values

    #---------- examples of using the thin film library ------------#

    # define incident medium refractive index (air)
    i_n = np.ones(np.shape(wv)).astype(complex)

    # make call to fresnel_bare() method (using default 'units' kwarg)
    fb_output = tff.fresnel_bare(i_n, substrate, theta)


    # create the film refractive index array
    f_n = np.zeros((len(layers), np.shape(wv)[1])).astype(complex)
    for i, val in enumerate(materials):
        if val == "H":
            f_n[i, :] = high_mat
        else:
            f_n[i, :] = low_mat


    # call admit_delta() method
    ad_output = tff.admit_delta(wv, layers, theta, i_n, substrate, f_n)

    # call the c_mat() method using results from ad_output
    ns_film = ad_output['ns_film']
    np_film = ad_output['np_film']
    delta = ad_output['delta']

    cmat_output = tff.c_mat(ns_film, np_film, delta)

    # call the fil_spec() method
    filspec_output = tff.fil_spec(wv, substrate,
                                high_mat, low_mat,
                                layers, materials,
                                theta, sub_thick)

    # remove dimensions of size 1 for plotting
    x_vals = np.squeeze(wv)
    y_vals = np.squeeze(np.real(filspec_output['T']))

    # plot the results
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)
    ax.set(xlabel='Wavelength (nm)', ylabel='Transmission',
            title='FilSpec Transmission')
    ax.grid()
    plt.show()

if __name__=='__main__':
    main()
