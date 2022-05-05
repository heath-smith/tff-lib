"""
utils.py contains helper functions used in the
tff_lib package.
"""

# import dependencies
import numpy as np

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

