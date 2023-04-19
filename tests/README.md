# Testing
The test data can be found in the data folder in the top level package
directory inside the test_data.json file.

The data file is structured in the following way:

```json
{
    "input": {
        "sub_thick": float,
        "layers": Iterable[Tuple(str, float)],
        "wv": Iterable[float],
        "substrate": Iterable[float],
        "high_mat": Iterable[float],
        "low_mat": Iterable[float],
        "env_int": Iterable[float]
    },
    "output": {
        "effective_index": Iterable[float],
        "path_length": Iterable[float],
        "char_matrix": Dict[str, Iterable[complex]],
        "admit_delta": Dict[str, Iterable[float]],
        "filspec": Dict[str, Iterable[float]],
        "fresnel_bare": Dict[str, Iterable[float]],
        "fresnel_film": Dict[str, Iterable[float]]
    }
}
```