"""
utils.py contains helper functions used in the
tff_lib package.
"""

# import dependencies
import numpy as np
from typing import Union

def convert_to_numpy(data:dict, is_complex:bool=False) -> dict:
    """
    Convert lists in input dict to numpy arrays.
    """

    # validate input data
    if not isinstance(data, dict):
        raise TypeError(f'"data" expects <dict>, received {type(data)}.')
    if not isinstance(is_complex, bool):
        raise TypeError(f'"is_complex" expects <bool>, received {type(is_complex)}.')

    output = {}
    for k, v in data.items():
        if isinstance(v, list):
            output[k] = np.asarray(v).astype(np.complex) if is_complex else np.asarray(v)
        else:
            output[k] = v

    return output

def film_matrix(layers:list, high:np.ndarray, low:np.ndarray) -> np.ndarray:
    """
    Constructs a matrix of the thin film materials
    using their respective refractive index arrays.

    Parameters
    ----------
    layers = list of tuple pairs of material and thickness. ie: ("H", 1.2345)\n
    high = array of refractive indices of high index material.\n
    low = array of refractive indices of low index material.\n

    Returns
    -----------
    (np.ndarray) of shape (N-layers, len(high))
    """

    # validate the input data
    if not isinstance(high, np.ndarray):
        raise TypeError(f'"high" expects <np.ndarray>, received {type(high)}.')
    if not isinstance(low, np.ndarray):
        raise TypeError(f'"low" expects <np.ndarray>, received {type(low)}.')
    if not isinstance(layers, list):
        raise TypeError(f'"layers" expects <list>, received {type(layers)}.')
    if not high.shape == low.shape:
        raise ValueError(f'shape mismatch -----> {high.shape} != {low.shape}.')

    # film refractive indices
    films = np.zeros((len(layers), len(high))).astype(complex)

    # get measured substrate & thin film optical constant data
    for i, val in enumerate(layers):
        if val[0] == "H":
            films[i, :] = high
        elif val[0] == "L":
            films[i, :] = low

    return films


def path_length(
    sub_thick:Union[int, float], med:np.ndarray, sub_n_eff:np.ndarray,
    theta:Union[int, float], units:str='rad') -> np.ndarray:
    """
    Calculates the estimated optical path length
    through a given substrate.

    Parameters
    -------------
    sub_thick = thickness of substrate in (mm).\n
    med = refractive index of incident medium.\n
    sub_n_eff = effective refractive index of substrate.\n
    theta = angle of incidence.\n
    units = 'rad' or 'deg'.

    Raises
    -------------
    TypeError, ValueError

    Returns
    ------------
    (np.ndarray) Length of the path through the substrate.
    """

    # convert incident angle from degrees to radians
    theta = theta * (np.pi / 180) if units == 'deg' else theta

    # calculate the numerator and denominator
    num = sub_thick * (10**6)
    den = np.sqrt(1 - (np.abs(med)**2 * (np.sin(theta)**2) / sub_n_eff**2))

    # return the path length
    return num / den

