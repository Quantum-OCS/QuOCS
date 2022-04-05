# Feature List


## Quantum Optimal Control Algorithms
### dCRAB
_Description_:

The dressed CHopped RAndom Basis algorithm (dCRAB) avoids local optima in optimisations with a 
low number of parameters through super-parameters [1]. Superparameters specify the basis-elements used 
during one sub-optimisation, called super-iteration (SI). They describe f.e. frequencies of the
Fourier basis or times at which to position step functions for the sigmoid basis.


_json_: 
~~~json
"algorithm_settings": {
    "opti_algorithm_name":      "dCRAB",
    "super_iteration_number":   5,
    "max_eval_total":          100
    "is_compensated_drift":     true,
    "random_number_generator": {
        "seed_number": 42
    },
    "re_evaluation": {
        "re_evaluation_steps": [0.3, 0.5, 0.51]
    }
    }
~~~

_Settings_:
- `"super_iteration_number"` The maximum number of superiterations, i.w. sub-optimisations, determines how many set of super-parameters are used 
to sequentially optimise the problem
- `"max_eval_per_SI"` Number of iterations per super-iteration.
- `"is_compensated_drift"`
- `"random_number_generator"`>`"seed_number"`
- `"re_evaluation"`>`"re_evaluation_steps"`


_References_:

[1] N. Rach, M. M. Müller, T. Calarco, and S. Montangero, “Dressing the chopped-random-basis optimization: A bandwidth-limited access to the trap-free landscape,” Phys. Rev. A - At. Mol. Opt. Phys., vol. 92, no. 6, p. 62343, 2015, doi: 10.1103/PhysRevA.92.062343.


### Direct Search
_Description_:

lalala [1]


_json_: 
~~~json

~~~

_Settings_:
- `"option"` lalala


_References_:
[1]

### GRAPE
_Description_:

lalala [1]


_json_: 
~~~json

~~~

_Settings_:
- `"option"` lalala


_References_:
[1]

### Automatic Differentiation
_Description_:

lalala [1]


_json_: 
~~~json

~~~

_Settings_:
- `"option"` lalala


_References_:
[1]


## Updating Algorithms
~~~json
"dsm_settings": {
    "general_settings": {
        "dsm_algorithm_name": "NelderMead",
        "is_adaptive": true
    },
    "stopping_criteria": {
        "xatol": 1e-2,
        "frtol": 1e-2
    }
~~~
### Direct search Methods
- Nelder-Mead
- CMAES

### Gradient-Based Optimisation
- BFGS
- L-BFGS-B


### Gradient-based specific features
- Time Evolution
- Noise Processes

### Stopping Criteria
- Convergence
- Goal
- Time-out
- Step-Size


## Pulses
### Features
~~~json
"pulses": [{"pulse_name": "Pulse_1",
            "upper_limit": 15.0,
            "lower_limit": -15.0,
            "bins_number": 101,
            "time_name": "time_1",
            "amplitude_variation": 0.3,
            "scaling_function": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: 1.0 + 0.0*t"
            },
            "initial_guess": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: np.pi/3.0 + 0.0*t"
            }
            "basis": {
                ...
            },
}],
~~~
### Basis
~~~json
"basis": {
    "basis_class": "Fourier",
    "basis_module": "quocslib.pulses.basis.Fourier",
    "basis_attribute": null,
    "basis_vector_number": 2,
    "random_super_parameter_distribution": {
        "distribution_class": "Uniform",
        "distribution_module": "quocslib.pulses.superparameter.Uniform",
        "distribution_attribute": null,
        "lower_limit": 0.1,
        "upper_limit": 5.0
    }
~~~
### Times
~~~json
"times": [{
    "time_name": "time_1",
    "initial_value": 3.0
}]
~~~

## Parameters
~~~json
"parameters": [{
            "parameter_name": "delta1",
            "lower_limit": -2.0,
            "upper_limit": 2.0,
            "initial_value": 0.01,
            "amplitude_variation": 0.5
        }],
~~~


## Test Problems
### OneQubitProblem.py
### OneQubitProblem_2fields.py
### RosenbrockProblem.py
### su2.py