# Feature List of the possible Settings in the Optimization Dictionary

The optimizer of QuOCS needs the settings for the optimization to be provided in the form of a dictionary. This `optimization_dictionary` can either be defined in the Python code itself or as a JSON file and be read in with the provided `readjson` function. The settings (keys) in that dictionary are listed and explained here.


## General form of the JSON file

Assuming you define the settings in the form of a .json file, the general structure should look like this:

~~~yaml
{
    "optimization_client_name": "Name_of_your_Optimization",
    "create_logfile": true,  # determines if you want to save the log-file
    "console_info": true, # determines if you want the optimization output printed to the console
    "use_microsecond_name": false, # if true uses microseconds in name to avoid duplicates in name (e.g. if executed many times in parallel)
    "dump_format": "npz",  # format of the results file
    "algorithm_settings": {...},  # settings related to the algorithm
    "pulses": [{...}, {...}, ...],  # list of pulses and their settings
    "parameters": [{...}, {...}, ...],  # list of parameters and their settings
    "times": [{...}, {...}, ...]  # list of times and their settings
}
~~~

These entries are keywords but you can add more entries to the dictionary if you want to add a comment for example.

The `optimization_client_name` determines the name of the folder where the optimization results are saved. A timestamp is prepended to the name so that each optimization run generates a separate output folder. You can find the results inside of a "QuOCS_Results" folder created in the directory from where you execute the optimization.

The `create_logfile` key determines if the full output of the Python terminal is saved to a log-file. If you have an optimization with many iterations this can become quite large, so you might want to  disable it once you know your optimizations are running reliably.

The `console_info` key determines if the optimization output is shown in the terminal during the execution of an optimization. Indeed, if you want to run several optimizations in parallel, the standard output in the terminal gets mixed up and is not helpful anymore. In this case you might want to disable it once you know your optimizations are running reliably. The normal log-file is still created with the correct information for that specific run (if not disabled by the `create_logfile` option).

The `dump_format` key specifies the format of the results file (best controls and some meta data). Currently you can choose between "npz" and "json". The default (if you do not give this key) is "npz".

**Tip:** You can also change specific entries in the code after reading in the .json file if you, e.g., want to sweep certain parameters or have the name of the optimization defined on runtime.


## Algorithm Settings

We focus on an example of an optimization using the dCRAB algorithm in combination with Nelder-Mead. Examples for GRAPE and a pure parameter optimization are given below.


### Example: dCRAB with Nelder-Mead

We have included any possible key in this example. Note that the values defined might not be realistic and depend on your specific optimization!

To fully understand some of the settings discussed here you might want to have a look at the following references:

* [T. Caneva, T. Calarco, and S. Montangero, *Chopped random-basis quantum optimization*, Phys. Rev. A 84, 022326, (2011)](https://doi.org/10.1103/PhysRevA.84.022326)
* [N. Rach et al., *Dressing the chopped-random-basis optimization: A bandwidth-limited access to the trap-free landscape*, Phys. Rev. A 92, 062343, (2015)](https://doi.org/10.1103/PhysRevA.92.062343)
* [M. M. Müller et al, *One decade of quantum optimal control in the chopped random basis*, Rep. Prog. Phys. 85, 076001, (2022)](https://doi.org/10.1088/1361-6633/ac723c)

There are settings that are connected immediately to the QOC algorithm and are specified as entries directly in the "algorithm_settings" sub-dictionary. Settings for the direct search method (dsm) that is used to find the optimal values of the basis expansion in dCRAB for each super-iteration (SI) are set in the sub-dictionary under "dsm_settings". Here, you can also find the stopping criteria for single SIs.

~~~yaml
"algorithm_settings": {
    "algorithm_name": "dCRAB",
    "optimization_direction": "minimization",
    "super_iteration_number": 10,
    "max_eval_total": 20000,
    "total_time_lim": 2,
    "FoM_goal": 0.00001,
    "compensate_drift": {
        "compensate_after_SI": true,
        "compensate_after_minutes": 0.01,
        "num_average": 1,
    },
    "random_number_generator": {
        "seed_number": 420
    },
    "re_evaluation": {
        "re_evaluation_steps": [0.3, 0.5, 0.51]
    },
    "dsm_settings": {
        "general_settings": {
            "dsm_algorithm_name": "NelderMead",
            "is_adaptive": true
        },
        "stopping_criteria": {
            "max_eval": 20,
            "time_lim": 0.01,
            "xatol": 1e-14,
            "fatol": 1e-14,
            "change_based_stop": {
                "cbs_funct_evals": 300,
                "cbs_change": 0.00001
            }
        }
    }
}
~~~


#### Main Settings

| Setting | Type | Explanation |
| --- | --- | --- |
|**"algorithm_name"** |*string*| The name of the QOC algorithm. So far contains "dCRAB", "GRAPE", "AD" (automatic differentiation) and "DirectSearch". Can be extended (see [Customization](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Customization.md)). |
|**"optimization_direction"** *(optional)* |*string*| "minimization" or "maximization". *(Default: "minimization")* |
|**"super_iteration_number"** |*Int*| Maximum number of super-iterations to perform |
|**"max_eval_total"** *(optional)* |*Int*| Maximum number of function evaluations (in total) to perform *(Default: 10e10)* |
|**"total_time_lim"** *(optional)* |*float*| Time limit in minutes for the total optimization run |
|**"FoM_goal"** *(optional)* |float| Stop the optimization when this FoM value is reached |
|**"compensate_drift"** *(optional)* |*dict*| Compensation for drifting FoM in an experiment. Only works for Nelder-Mead due to the nature of the search method. "compensate_after_SI" re-calibrates the current best FoM after each SI. "compensate_after_minutes" periodically re-calibrates the current best FoM after the given value in minutes. "num_average" (optional) specifies the number of repeated measurements to perform and average for drift compensation (only for periodic compensation after XX minutes). The default is 1. |
|**"random_number_generator"** *(optional)* |*dict*| "seed_number" specifies a seed for the selection of randomized values during the optimization. Useful for generating reproducible runs during finding phase for optimization parameters. |
|**"re_evaluation"** *(optional)* |*dict*| Re-evaluate the measured pulse (in the case of a noisy measurement) to improve measurement uncertainty. Under "re_evaluation_steps" one defines a list of probabilities. Given the standard deviation of a measurement and the measured FoM the evaluation is repeated if the probability is that a set of parameters might actually be a new optimum considering the measurement uncertainty. The measurement is repeated maximally as often as the length of the given list. |


#### Direct Search Method Settings

In the general settings one can define the search method to be used and related, specific parameters.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"dsm_algorithm_name"** |*string*| Name of the dsm. So far one can pick from "NelderMead" and "CMAES". Can be extended (see [Customization](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Customization.md)). |

Specific to Nelder-Mead:

| Setting | Type | Explanation |
| --- | --- | --- |
|**"is_adaptive"** *(optional)* |*bool*| If you want to use an adaptive version of Nelder-Mead based on the simplex size according to [Gao, F., Han, L., *Implementing the Nelder-Mead simplex algorithm with adaptive parameters*, Comput. Optim. Appl. 51, (2012)](https://doi.org/10.1007/s10589-010-9329-3). *(Default: false)* |

Specific to CMA-ES:

| Setting | Type | Explanation |
| --- | --- | --- |
|**"population"** *(optional)* |*float*| Number of candidates per generation (iteration) of new individuals *(Default: 4 + 3 log(N))* (see [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES)). |


##### Stopping Criteria (dsm)

The stopping criteria for the dsm depend on the chosen search method.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"max_eval"** *(optional)* |*Int*| Maximum number of function evaluations per SI. *(Default: 10e10)* |
|**"time_lim"** *(optional)* |*float*| Time limit in minutes for singe SI. |
|**"xatol"** *(optional)* |*float*| Absolute change of simplex between iterations that is acceptable for convergence (see [here](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)). *(Default: 1e-14)* |
|**"fatol"** *(optional)* |*float*| Standard deviation of the points in the simplex (NM) or one population (CMA-ES) below which the optimization is considered as converged. *(Default: 1e-100)* |
|**"change_based_stop"** *(optional)* |*dict*| Stop the search if the FoM value does not change more than defined in "cbs_change" (on average) over a number of function evaluations given by "cbs_funct_evals". |


### Example: GRAPE

~~~yaml
"algorithm_settings": {
    "algorithm_name": "GRAPE"
}
~~~

For a GRAPE optimization you only need to define the algorithm name as "GRAPE". No specification of the dsm is needed. However, more details about the Hamiltonian need to be provided (see this [example](https://github.com/Quantum-OCS/QuOCS/blob/main/tests/test_GRAPE_Ising_Model.py) using this [FoM class](https://github.com/Quantum-OCS/QuOCS/blob/main/src/quocslib/optimalcontrolproblems/IsingModelProblem.py))


### Example: pure parameter optimization

~~~yaml
"algorithm_settings": {
    "algorithm_name": "DirectSearch",
    "dsm_settings": {
        "general_settings": {
            "dsm_algorithm_name": "NelderMead"
        },
        "stopping_criteria": {
            "xatol": 1e-14,
            "fatol": 1e-4
        }
    }
}
~~~

If you only want to optimize parameters and no pulses you can use QuOCS for its advanced stopping criteria and results management.

## Pulses

~~~yaml
"pulses": [{
    "pulse_name": "Pulse_1",
    "upper_limit": 15.0,
    "lower_limit": -15.0,
    "bins_number": 101,
    "time_name": "time_1",
    "amplitude_variation": 0.3,
    "shrink_ampl_lim": True,
    "scaling_function": {
        "function_type": "lambda_function",
        "lambda_function": "lambda t: 1.0 + 0.0*t"
    },
    "initial_guess": {
        "function_type": "lambda_function",
        "lambda_function": "lambda t: np.pi/3.0 + 0.0*t"
    }
    "basis": {
        "basis_name": "Fourier",
        "basis_vector_number": 6,
        "random_super_parameter_distribution": {
            "distribution_name": "Uniform",
            "lower_limit": 0.1,
            "upper_limit": 5.0
        }
    },
    "shaping_options": ["add_initial_guess",
                    	"add_base_pulse",
                    	"add_new_update_pulse",
                    	"scale_pulse",
                    	"limit_pulse"]
}]
~~~

A pulse in QuOCS is any time-dependent function that you want to vary and optimize. They are provided to the optimization dictionary under the key "pulses" as a list of pulse-settings for each pulse. 

### Main Pulses Settings

| Setting | Type | Explanation |
| --- | --- | --- |
|**"pulse_name"** |*string*| Some name to describe your pulse. *(Default: "pulse")* |
|**"upper_limit"** |*float*| Upper limit of the amplitude in the units your pulse is interpreted on the experiment or simulation. *(Default: 1.0)* |
|**"lower_limit"** |*float*| Lower limit of the pulse amplitude. *(Default: 0.0)* |
|**"bins_number"** |*Int*| Number of time steps the pulse is discretized in when sent for FoM evaluation. *(Default: 101)* |
|**"time_name"** |*string*| Time object connected to the pulse. Different pulses can have different times (i.e. lengths) associated with them. *(Default: "time")* |
|**"amplitude_variation"** |*float*| Initial amplitude variation of the pulse amplitude in the units of the pulse. Depends on the used basis and direct search algorithm. Typically, this should be a reasonable estimate of how strongly the pulse is to be varied. Larger values result in a more global search, while small values can be used for a local search and fine-tuning. *(Default: 1.0)* |
|**"shrink_ampl_lim"** *(optional)* |*bool*| If this option is set to *true*, the pulse is shrunk down in such a way to conserve most of its features / shape while obeying the amplitude restrictions. If this is turned off, the amplitudes are cut off at the limits. *(Default: false)* |
|**"scaling_function"** *(optional)* |*dict*| Scaling of the pulse, i.e. post-processing. Can be used to force a pulse to start and end a 0. The pulse is multiplied by this function before it is fed back to FoM evaluation. This function can be specified as a Python lambda function ("function_type": "lambda_function"). Then the key "lambda_function" should contain a lambda function depending on t and numpy constants / functions using the shortcut "np.". A scaling function can also be provided in the form of a list ("function_type": "list_function"). Then a key named "list_function" can contain a list of values that describe the scaling function which should have the same length as the "bins_number". It is recommended to read in (or create) such a list in the code and add it manually to the optimization_dictionary before the optimization object is created and executed. *(Default: no scaling)* |
|**"initial_guess"** *(optional)* |*dict*| Initial pulse from where to start the optimization. This function can be specified as a Python lambda function ("function_type": "lambda_function"). Then the key "lambda_function" should contain a lambda function depending on t and numpy constants / functions using the shortcut "np.". A guess pulse can also be provided in the form of a list ("function_type": "list_function"). Then a key named "list_function" can contain a list of values that describe the guess pulse which should have the same length as the "bins_number". It is recommended to read in (or create) such a list in the code and add it manually to the optimization_dictionary before the optimization object is created and executed. *(Default: 0 for all times)* |
|**"shaping_options"** *(optional)* |*list*| Specify the order in which the pulse is build up. "add_initial_guess" adds the initial guess pulse. "add_base_pulse" adds the update pulse from previous SIs (in the case of dCRAB). "add_new_update_pulse" adds the current update pulse from the algorithm. "scale_pulse" scales the pulse with the given scaling function. "limit_pulse" enforces the pulse limits if necessary. *(Default: ["add_initial_guess", "add_base_pulse", "add_new_update_pulse", "scale_pulse", "limit_pulse"])* |

#### Initial Guess and Scaling with custom Python file

The initial guess, as well as the scaling function can be linked to custom functions stored in a Python file. The entries should look like this:

~~~yaml
"scaling_function": {
    "function_type": "python_file",
    "file_path": "my_file_with_functions",
    "function_name": "scaling_function",
    "path_mode": "relative"
},
"initial_guess": {
    "function_type": "python_file",
    "file_path": "my_file_with_functions",
    "function_name": "guess_pulse_function",
    "path_mode": "relative"
},
~~~

where the "file_path" option describes the absolute or relative path to the Python file. Wether you want to provide an absolute or relative path can be set with "path_mode". The option "function_name" is for the name of the function inside the Python file. It should either only take the time as a parameter (first argument) or the time and the pulse (in that order).

#### Example 1:

~~~python
import numpy as np


def scaling_function(t):
	return -15 * (t - 0.5)**4 + 1


def scaling_function_with_pulse(t, pulse):
	return np.abs(pulse)


def guess_pulse_function(t):
	return np.pi/3.0 + 0.0*t
~~~

#### Example 2:

A useful scaling function is the following:

~~~python
import numpy as np

T = 10
steepness = 30

def shape_function(t):
    return np.tanh(np.sin(np.pi*t/(2*T)) * steepness) * np.tanh(-np.sin(np.pi*(t-T)/(2*T)) * steepness)
~~~

where the time T should be your own pulse time and the steepness can be adjusted to reflect the rise time of the pulse in your experiment. The shape of the example looks like ths:

<img src="https://github.com/Quantum-OCS/QuOCS/assets/44496709/a8d1bbaf-e805-4e47-b106-726a86a33b3e" width="50%" height="50%">

In this case the pulse will start and end at zero if multiplied with the scaling function and is not changed in between except for the rise and fall time specified by the steepness.

#### Example 3:

Having fixed non-zero start and end points for the pulse.

Please see [this example](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Enforcing_Pulse_Boundaries_at_Beginning_and_End.md)

### Basis Settings

| Setting | Type | Explanation |
| --- | --- | --- |
|**"basis_name"** |*string*| Name of the basis in which to expand the pulse. So far on can select from "Fourier", "Chebyshev", "PiecewiseBasis", "Sinc", and "Sigmoid". Depending on the basis, the randomized super-parameter differs. It is recommended to start with the Fourier basis. In that case the randomly selected parameter is the frequency of the trigonometrical function updating the pulse during dCRAB iterations. For more information please contact the developers. |
|**"basis_vector_number"** |*Int*| Number of vectors used for basis expansion i each super-iteration of the dCRAB algorithm. Typically, one vector corresponds to 1-2 parameters to optimize (e.g. amplitude and phase of a sine function). The higher this value is set, the longer each SI takes to converge. For more information please contact the developers. *(Default: 1)* |
|**"random_super_parameter_distribution"** |*dict*| Distribution from which to sample the randomized super-parameter. So far, the only option is "distribution_name": "Uniform" where one can set to upper and lower limits of the parameter. In the case of the Fourier basis, this corresponds to the number of oscillations allowed during the pulse time. Therefore, depending on the length of the pulse, this enforces bandwidth constraints. For more information please contact the developers. |
|**"sigma"** *(optional)* |*float*| The Sigmoid basis is designed to keep the pulse spectrum envelope-limited. Sigma specifies the steepness of the individual rises, which will in turn reduce high-frequency excitations. A good point to start is final_time/100. *(Default: 0.1)* |
|**"offset"** *(optional)* |*float*| The Sigmoid basis is designed to keep the pulse spectrum envelope-limited and the pulse zero at beginning and end. The offset determines how far from the edges it will go down. This parameter should depend on sigma and the maximum amplitude. A good point to start is sigma x (upper_limit - loweer_limit)/10. *(Default: 0.1)* |
| **"basis_max"** *(optional)*              |*float*| The Sinc basis keeps a spectrum of consistent width defined by this parameter. The units are defined in the same way as the Fourier frequencies.  *(Default: 1.)*|

## Bases Overview
| Basis name | Implementation | Visualization |
| --- | --- | --- |
| Fourier | $A_i \sin{(2 \pi \omega_i t/t_f)} + B_i \cos{(2 \pi \omega_i t/t_f)}$ |<img src="https://user-images.githubusercontent.com/47388967/233591288-8bf7a6be-59de-4ea5-b7c6-c7e67e11e1be.png" width="50%" height="50%">|
| Sigmoid | $A_i \frac{1}{2}\left(1+\text{erf}\left(\frac{t-\tau_i}{\sqrt{2}\sigma}\right)\right)$ |<img src="https://user-images.githubusercontent.com/47388967/233591361-058d96b3-9500-4808-a5da-a717cdc36f11.png" width="50%" height="50%"> |
| Piecewise | $A_i \text{  for  } t_{i-1} < t \le t_i$ |<img src="https://user-images.githubusercontent.com/47388967/233593775-709fcf3a-50d9-48c7-a038-c094126a9ccb.png" width="50%" height="50%">|
| Chebychev | $A_i \cos{\left(\omega_i \cos^{-1}(-1 + (2 t/t_f))+ B_i\right)}$ |<img src="https://user-images.githubusercontent.com/47388967/233595052-cea0edfb-127c-47bb-9659-6e61b8ceb24d.png" width="50%" height="50%"> <br/>*(image: By Glosser.ca - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=52799132)*|
| Sinc | $A_i  \text{sinc}(2 \pi \omega_\text{max} \frac{t-t_i}{t_f})= A_i \frac{\sin(2 \pi \omega_\text{max} (t-t_i)/t_f)}{2 \pi \omega_\text{max} (t-t_i)/t_f}$ |<img src="https://github.com/Quantum-OCS/QuOCS/assets/47388967/19369567-e1c2-400d-906b-834c2bc5af78" width="50%" height="50%"> <br/>*(image: By Chou https://www.semanticscholar.org/paper/Orthogonal-Basis-Functions-in-Discrete-Time-and-to-Chou/2a1a70fa3e64f17feabf575ef71dc48caa1081b3)*|


## Parameters

~~~yaml
"parameters": [
    {
        "parameter_name": "Parameter1",
        "lower_limit": -2.0,
        "upper_limit": 2.0,
        "initial_value": 0.4,
        "amplitude_variation": 0.5
    },
    {
        "parameter_name": "Parameter2",
        "lower_limit": -2.0,
        "upper_limit": 2.0,
        "initial_value": 0.4,
        "amplitude_variation": 0.5
    }
]
~~~

Parameters are values that are constant during the application of a pulse (or set of pulses) but varied and optimized during an optimization run. Examples are an offset voltage, a free evolution time or variables for a simulation.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"parameter_name"** |*string*| A name that can be assigned to the parameter. *(Default: "parameter")* |
|**"lower_limit"** |*float*| Lower limit of the parameter. *(Default: 0.1)* |
|**"upper_limit"** |*float*| Upper limit of the parameter. *(Default: 1.1)* |
|**"initial_value"** |*float*| Initial value of the parameter. *(Default: 0.1)* |
|**"amplitude_variation"** *(optional)* |*float*| Reasonable initial variation of the parameter during optimization. *(Default: 0.1)* |



## Times

~~~yaml
"times": [{
    "time_name": "time_1",
    "initial_value": 3.0
}]
~~~

Times are also specified as a list. Each pulse has to have a time assigned to it so that it effectively describes its duration. There can be several pulses where each has another time object assigned to it. Currently, the length of the pulse time can not be optimized here, but it is planned as an addition in future updates. If the pulse time hast to be optimized it is advised to just define a time here and link it but use the pulse length in the form of a parameter.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"time_name"** |*string*| A name that can be assigned to the time. *(Default: "time")* |
|**"initial_value"** |*float*| (Initial) value of the time. Used for creation of the returned time-array. *(Default: 1.0)* |

**Further options for time (pulse-length) optimization to be added in future updates.**
