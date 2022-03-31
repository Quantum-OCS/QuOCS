# Feature List

## Test Problems

## Quantum Optimal Control Algorithms
### dCRAB
_Description_:

The dressed CHopped RAndom Basis algorithm (dCRAB) avoids local optima in optimisations with a 
low number of parameters through super-parameters [1]. Superparameters specify the basis-elements used 
during one sub-optimisation, called super-iteration (SI). They describe f.e. frequencies of the
Fourier basis or times at which to position step functions for the sigmoid basis.


_json_: 
~~~
"algorithm_settings": {
    "opti_algorithm_name":"dCRAB",
    "super_iteration_number": 5,
    "max_eval_per_SI": 50
    "is_compensated_drift": true,
    "random_number_generator": {
        "seed_number": 420
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


_References_:

[1] N. Rach, M. M. Müller, T. Calarco, and S. Montangero, “Dressing the chopped-random-basis optimization: A bandwidth-limited access to the trap-free landscape,” Phys. Rev. A - At. Mol. Opt. Phys., vol. 92, no. 6, p. 62343, 2015, doi: 10.1103/PhysRevA.92.062343.

### GRAPE
### Automatik Differentiation


## Updating Algorithms
### Nelder-Mead
### CMAES
### BFGS
### L-BFGS-B


## Stopping Criteria
### Convergence
### Goal
### Time-out
### Step-Size


## Noise and Drift Compensation
### Re-evaluate steps
### Re-measure previous best


## Pulse features
### Amplitude limits
### Duration
### Variation scale
### Bases
- Fourier
- Sigmoid

## Parameter features
### Amplitude limits


## Gradient-based specific features
### Time Evolution
### Noise Processes