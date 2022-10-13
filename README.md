<p  align='center'> <img src="./logo/logo_quocs_color.png" width="400" /></p>
<h1 align='center'>The optimization library</h1>
     
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/unit_testing_linux.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/unit_testing_windows.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/unit_testing_macOS.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)

    
QuOCS (Quantum Optimal Control Suite) is a python software package for model- and experiment-based optimizations of quantum processes.
It uses the excellent Numpy and Scipy packages as numerical backends.
QuOCS aims to provide a user-friendly interface to solve optimization problems. A variety of popular optimal control algorithms are available:
* GRAPE (GRadient Ascent Pulse Engineering) Algorithm
* dCRAB (dressed Chopped RAndom Basis) Algorithm
* AD-GRAPE (Automatic Differentiation) Algorithm
* Direct Search Algorithm, i.e. Nelder Mead, CMA-ES...


QuOCS is open source and its interface structure allows for user-friendly customizability. It can be used on all Unix-based platforms and on Windows.

## Installation

[![Pip Package](pypi_badge.svg)](https://pypi.org/project/quocs-lib/)
[![Build Status](https://github.com/Quantum-OCS/QuOCS/actions/workflows/python_publish_PyPI.yml/badge.svg)](https://github.com/Quantum-OCS/QuOCS/actions)

QuOCS is available on `pip`. You can install QuOCS by doing

~~~bash
pip install quocs-lib
~~~

The requirements are:
* setuptools >= 44.0.0
* numpy >= 1.19.1
* scipy >= 1.5.1
* If you want to use the AD Algorithm, the installation of [JAX](https://github.com/google/jax) (Autograd and XLA) is required.

### Editable mode
If you want to customize the algortihm and basis inside QuOCS, the package has to be installed in the editable mode. You can easily do that with the following commands:

~~~bash
git clone https://github.com/Quantum-OCS/QuOCS.git
cd QuOCS
pip install -e .
~~~

## Documentation

The possible [settings](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Settings_in_Optimization_Dict.md) for the JSON file can be found [here](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Settings_in_Optimization_Dict.md).

You can find the latest development documentation [here](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation).

A selection of demonstration notebooks is available, which demonstrate some of the many features of QuOCS. These are stored in the [QuOCS/QuOCS-jupyternotebooks repository](https://github.com/Quantum-OCS/QuOCS-jupyternotebooks) here on GitHub.


## Example of usage

Using QuOCS is intuitive and simple. The main steps are:

1. Create and load the optimization dictionary. This json file contains all the optimization settings (as an example see [this file](https://github.com/Quantum-OCS/QuOCS/blob/main/tests/dCRAB_Fourier_NM_OneQubit.json)).
    ~~~python
    from quocslib.utils.inputoutput import readjson
    optimization_dictionary = readjson("opt_dictionary.json"))
    ~~~
2. Create Figure of Merit object. This is an instance of a class that contains the physical problem to be optimized. In the following, you can see an example of how to define this class. The input and output of `get_FoM` should not be changed.

    ~~~python
    from quocslib.utils.AbstractFoM import AbstractFoM
    # Define problem class
    class OneQubit(AbstractFoM):

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
    ~~~
3. Define the optimizer by initializing it with the uploaded optimization dictionary and FoM object. After that the execution can be run.
    ~~~python
    from quocslib.Optimizer import Optimizer
    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary,
                                 FoM_object)
    # Execute the optimization
    optimization_obj.execute()
    ~~~

Complete examples are provided in [QuOCS/QuOCS-jupyternotebooks repository](https://github.com/Quantum-OCS/QuOCS-jupyternotebooks) or in the [tests](https://github.com/Quantum-OCS/QuOCS/tree/main/tests) folders.

## Usage with Qudi

If you want to use QuOCS in combination with Qudi, please have a look at [this repository](https://github.com/Quantum-OCS/Qudi-plugin) with additional files, information and a tutorial.

## Contribute

Would you like to implement a new algorithm or do you have in mind some new feature it would be cool to have in QuOCS?
You are most welcome to contribute to QuOCS development! You can do it by forking this repository and sending pull requests, or filing bug reports at the [issues page](https://github.com/Quantum-OCS/QuOCS/issues).
All code contributions are acknowledged in the [contributors]() section in the documentation. Thank you for your cooperation!

## Citing QuOCS
If you use QuOCS in your research, please cite the original QuOCS papers that are available [here]().

## Authors and contributors
* [Marco Rossignolo](https://github.com/marcorossignolo)
* [Thomas Reisser](https://github.com/ThomasReisser90)
* [Alastair Marshall](https://github.com/alastair-marshall)
* [Phila Rembold](https://github.com/phila-rembold)
* [Alice Pagano](https://github.com/AlicePagano)
