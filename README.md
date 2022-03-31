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
* If you want to use the AD Algorithm, the installation of [JAX](https://github.com/google/jax) (Autograd and XLA) is required.

### Editable mode
If you want to customize the algortihm and basis inside QuOCS (see [customization](#customization)), the package has to be installed in the editable mode. You can easily do that with the following commands:

```bash
git clone https://github.com/Quantum-OCS/QuOCS.git
cd QuOCS
pip install -e .
```

## Documentation

You can find the latest development documentation [here](https://quantum-ocs.github.io/QuOCS).

A selection of demonstration notebooks is available, which demonstrate some of the many features of QuOCS. These are stored in the [QuOCS/QuOCS-jupyternotebooks repository](https://github.com/Quantum-OCS/QuOCS-jupyternotebooks) here on GitHub.


## Example of usage

Using QuOCS is intuitive and simple. The main steps are:

1. Create and load the optimization dictionary. This json file contains all the optimization settings.
    ```python
    from quocslib.utils.inputoutput import readjson
    optimization_dictionary = readjson("opt_dictionary.json"))[1]
    ```
2. Create Figure of Merit object. This is an instance of a class that contains the physical problem to be optimized. In the following, you can see an example of how to define this class. The input and output of `get_FoM` should not be changed.

    ```python
    # Define problem class
    class OneQubit(AbstractFom):

        def __init__(self, args_dict:dict = None):
            """ Initialize the dynamics variables"""
            if args_dict is None:
                args_dict = {}
            ...

        def get_FoM(self, pulses: list = [],
                    parameters: list = [],
                    timegrids: list = []
            ) -> dict:
            # Compute the dynamics and FoM
            ...

            return {"FoM": fidelity}

    # Create Figure of Merit object
    FoM_object = OneQubit()
    ```
3. Define the optimizer by initializing it with the uploaded optimization dictionary and FoM object. After that the execution can be run.
    ```python
    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary,
                                 FoM_object)
    # Execute the optimization
    optimization_obj.execute()
    ```

Complete examples are provided in [QuOCS/QuOCS-jupyternotebooks repository](https://github.com/Quantum-OCS/QuOCS-jupyternotebooks) or in the [tests](https://github.com/Quantum-OCS/QuOCS/tree/main/tests) folders.

## Customization
The interface structure of QuOCS easily allows for user customization. 

### Add a new algorithm
To add a new algorithm in the QuOCS library, you need to install the software in the editable mode as explained in [installation](#editable-mode).
Then, use the [algorithm template](https://github.com/Quantum-OCS/QuOCS/blob/main/src/quocslib/optimizationalgorithms/AlgorithmTemplate.py) to implement your own algorithm.

*To easily use the name of your new custom algorithm in the optimization dictionary with the `opti_algorithm_name` key, you have to update the `module_name` and `class_name` keys in [this file](https://github.com/Quantum-OCS/QuOCS/blob/main/src/quocslib/utils/map_dictionary.json)*.

You can test your new algorithm, by using the examples in the [tests](https://github.com/Quantum-OCS/QuOCS/tree/main/tests) folder or by creating your own test. 

### Add a new basis
The same procedure applies for the creation of new basis.
A detailed guide for this procedure is also available [here](https://github.com/Quantum-OCS/QuOCS/wiki/Create-new-basis).


## Contribute

Would you like to implement a new algorithm or do you have in mind some new feature it would be cool to have in QuOCS?
You are most welcome to contribute to QuOCS development! You can do it by forking this repository and sending pull requests, or filing bug reports at the [issues page](https://github.com/Quantum-OCS/QuOCS/issues).
All code contributions are acknowledged in the [contributors]() section in the documentation. Thank you for your cooperation!

## Citing QuOCS
If you use QuOCS in your research, please cite the original QuOCS papers that are available [here]().

## Authors and contributors
* [Marco Rossignolo](https://github.com/marcorossignolo)
* [Alastair Marshall](https://github.com/alastair-marshall)
* [Thomas Reisser](https://github.com/ThomasReisser90)
* [Phila Rhembold](https://github.com/phila-rembold)
* [Alice Pagano](https://github.com/AlicePagano)