# QuOCS: The optimization library

[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/unit_testing_linux.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/unit_testing_windows.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)

QuOCS (Quantum Optimal Control Suite) is a python software package for theoretical and experimental optimizations with optimal control.
It uses the excellent Numpy and Scipy packages as numerical backends. 
QuOCS aims to provide a user-friendly interface to solve optimization problems. A wide variety of efficient optimal control algorithms are available:
* AD (Automatic Differentiation) Algorithm
* Direct Search Algorithm
* GRAPE (GRadient Ascent Pulse Engineering) Algorithm
* dCRAB (dressed Chopped RAndom Basis) Algorithm
* dCRAB Noisy Algorithm

QuOCS is open source and its interface structure allows for user-friendly customizability (see [customization](#customization) section). It can be used on all Unix-based platforms and on Windows.

## Installation

[![Pip Package](pypi_badge.svg)](https://pypi.org/project/quocs-lib/)
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/python_publish_PyPI.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)

QuOCS is available on `pip`. You can install QuOCS by doing

```bash
pip install quocs-lib
```

The requirements are:
* setuptools >= 44.0.0
* numpy >= 1.19.1
* scipy >= 1.5.1

If you want to customize the algortihm and basis inside QuOCS (see [customization](#customization)), the package has to be installed in the editable mode. You can easily do that with the following commands:

```bash
git clone https://github.com/Quantum-OCS/QuOCS.git
cd QuOCS
pip install -e .
```

## Documentation

You can find the latest development documentation [here](https://quantum-ocs.github.io/QuOCS).

A selection of demonstration notebooks is available, which demonstrate some of the many features of QuOCS. These are stored in the [QuOCS/QuOCS-jupyternotebooks repository](https://github.com/Quantum-OCS/QuOCS-jupyternotebooks) here on GitHub.

## Tests
Now you are able to use the tests scripts in the tests folder :)

## Customization
change the files and adapt algorithms and search methods, you can install it in a folder you like and in the editable mode by downloading from Git and running

## Authors
M. Rossignolo, A. Marshall, T. Reisser, P. Vetter, P. Rhembold, A. Pagano, R. Said, M. Müller, T. Calarco, S. Montangero, F. Jelezko