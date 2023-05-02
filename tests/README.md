# Testing
The test data can be found in the data folder in the top level package
directory inside the test_data.json file.

The data file is structured in the following way:

```json
{
    "input": {
        "sub_thick": float,
        "layers": Iterable[Tuple(int, float)],
        "wv": Iterable[float],
        "substrate": Iterable[float],
        "high_mat": Iterable[float],
        "low_mat": Iterable[float],
        "env_int": Iterable[float]
    },
    "output": {
        "effective_index": Iterable[float],
        "path_length": Iterable[float],
        "char_matrix": {
            "S11": Iterable[float],
            "S12": Iterable[float],
            "S21": Iterable[float],
            "S22": Iterable[float],
            "P11": Iterable[float],
            "P12": Iterable[float],
            "P21": Iterable[float],
            "P22": Iterable[float]
        },
        "admit_delta": {
            "ns_inc": Iterable[float],
            "np_inc": Iterable[float],
            "ns_sub": Iterable[float],
            "np_sub": Iterable[float],
            "ns_film": Iterable[float],
            "np_film": Iterable[float],
            "delta": Iterable[float]
        },
        "filspec": {
            "T": Iterable[float],
            "Ts": Iterable[float],
            "Tp": Iterable[float],
            "R": Iterable[float],
            "Rs": Iterable[float],
            "Rp": Iterable[float]
        },
        "fresnel_bare": {
            "Ts": Iterable[float],
            "Tp": Iterable[float],
            "Rs": Iterable[float],
            "Rp": Iterable[float],
            "Fs": Iterable[float],
            "Fp": Iterable[float]
        },
        "fresnel_film": {
            "Ts": Iterable[float],
            "Tp": Iterable[float],
            "Rs": Iterable[float],
            "Rp": Iterable[float],
            "ts": Iterable[float],
            "tp": Iterable[float],
            "rs": Iterable[float],
            "rp": Iterable[float]
        }
    }
}
```