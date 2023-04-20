# Introduction
The Thin Film Filter library contains a collection of physical abstractions
of real-world optical materials used in making a thin-film optical filter.
A thin-film optical filter typically consists of a substrate material,
a stack of thin film layers, and an incident medium. The objects found in this
library offer a software API which mimics real-world behavior of optical
materials used in thin-film production.

# Installation
1. In a directory of your choice, create and a new python environment
```console
$ python -m venv env
```

2. Activate the environment with the following command:
```console
Linux/Mac:
$ source ./env/bin/activate

Windows:
$ ./env/Scripts/activate
```

2.	install tff_lib in the newly activated python environment:
```console
(env)$ python -m pip install git+https://ThorlabsSpectralWorks@dev.azure.com/ThorlabsSpectralWorks/Python%20Packages/_git/tff-lib
```
3.	tff_lib requires python >= 3.10

# Getting Started
To begin using tff_lib, import tff_lib just like any other python library, and start exploring the features. Below are some examples of common use cases.
```python
from tff_lib import ThinFilm, FilmStack

thick = 500                         # thickness in nanometers
mat = 'H'                           # high index material type
wvls = [400, 500, 600, 700, 800]    # wavelenghts in nanometers
idx = [1, 1, 1, 1, 1]               # refractive indices at each wvl

# create a ThinFilm object
film1 = ThinFilm(mat, thick, wvls, idx)

# create a low-index film
film2 = ThinFilm('L', thick, wvls, idx*0.5)

# create a stack of thin films
# which alternate 'H' and 'L'
stack = FilmStack([film1, film2])
```

# Design Principles
The diagram below shows the basic structure of tff_lib. tff_lib provides interfaces
to most of the common elements used in a thin-film optical filter design, including
ThinFilm, FilmStack, ThinFilmFilter, Substrate, and OpticalMedium. Additionally, tff_lib supports a RandomFilmStack which can be used in some exhaustive search
optimization applications. There is also a custom error class, WritePropertyError which is used to safeguard read-only class properties.

![Diagram](./class_diagram.png)

# Reference
[UML Wiki](https://en.wikipedia.org/wiki/Unified_Modeling_Language)

[UML Class Diagrams](https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-class-diagram-tutorial/)

[NumPy](https://numpy.org/)

[SciPy](https://scipy.org/)

[Thin-film Optics](https://en.wikipedia.org/wiki/Thin-film_optics)

[Thin-film interference](https://en.wikipedia.org/wiki/Thin-film_interference)