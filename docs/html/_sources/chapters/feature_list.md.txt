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
    "opti_algorithm_name": "dCRAB",
    "super_iteration_number": 5,
    "max_eval_total": 100,
    "total_time_lim": 30,
    "FoM_goal": 0.00001,
    "optimization_direction": "minimization",
    "compensate_drift": {
                "compensate_after_SI": True,
                "compensate_after_minutes": 0.01
            },
    "random_number_generator": {
        "seed_number": 42
    },
    "re_evaluation": {
        "re_evaluation_steps": [0.3, 0.5, 0.51]
    }
}
~~~

_Settings_:

* `"opti_algorithm_name"`: The algorithm used for optimization.
* `"super_iteration_number"`: The maximum number of superiterations, i.e. sub-optimisations, are used to sequentially optimize the problem
* `"max_eval_total"`: Maximum number of function evaluations.
* `"total_time_lim"`: Maximum total time spent on the optimization in minutes.
* `"FoM_goal"`: this cancels the optimization after the goal FoM has been reached
* `"optimization_direction"`: define if it is a "maximization" or a "minimization".
* `"compensate_drift"`: Dictionary containing additional options for drift compensation.
	- `"compensate_after_SI"`: re-evaluates the current best pulse and updates the FoM at the beginning of each SI.
	- `"compensate_after_minutes"`: periodaically updates the current best FoM after a given time in minutes-
* `"random_number_generator"`
	- `"seed_number"`: specify a seed for reproducible optimization for benchmarking, debugging and comparison of certain parameters.
* `"re_evaluation"`
	- `"re_evaluation_steps"`: specify a list of probabilities with potential best controls are re-evaluated to make sure they have better performance. This option is useful for measurements with a known standard deviation. Please contact the developers for more information or try with the default values (no `"re_evaluation_steps"` key) or use the one given here as an example.


_References_:

[1] N. Rach, M. M. Müller, T. Calarco, and S. Montangero, “Dressing the chopped-random-basis optimization: A bandwidth-limited access to the trap-free landscape,” Phys. Rev. A - At. Mol. Opt. Phys., vol. 92, no. 6, p. 62343, 2015, doi: 10.1103/PhysRevA.92.062343.


### Direct Search
_Description_:

lalala [1]


_json_: 
~~~json
"dsm_settings": {
	"general_settings": {
		"dsm_algorithm_name": "NelderMead",
		"is_adaptive": True
	}, 
	"stopping_criteria": {
		"max_eval": 100,
		"time_lim": 5,
		"xatol": 1e-2,
		"frtol": 1e-2
	}
}
~~~

_Settings_:

* `"general_settings"`: General settings for the direct search
	-  `"dsm_algorithm_name"`: for the search algorithm (currently `"NelderMead"`and `"CMAES"`)
	- `"is_adaptive"`: to make use of the adaptive version of Nelder Mead (no effect or CMA-ES)
* `"stopping_criteria"`: Determines when a direct search is cancelled to proceed with the next SI (if the optimization is run within an enclosing algorithm) or when to stop the direct search
	- `"max_eval"`: maximum evaluations within a direct search, i.e. a sub-iteration for dCRAB
	- `"time_lim"`: maximum time spent in a direct search, i.e. a sub-iteration for dCRAB
	- `"xatol"`: criterion to cancel a direct search based on the simplex size (for Nelder Mead) or equivalent for e.g. CMA-ES
	- `"frtol"`: criterion to cancel the direct search based on the relative differences of the FoM in the simplex (for Nelder Mead) or equivalent for e.g. CMA-ES


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