# Feature List of the possible Settings in the Optimization Dictionary

The optimizer of QuOCS need the settings for the optimization to be provided in the form of a dictionary. This `optimization_dictionary` can either be defined in the python code itself or as a JSON file and be read in with the provided `readjson` function. The settings (keys) in that dictionary are listed and explained here.


## General form of the .json file

Assuming you define the settings in the form of a .json file, the general structure should look like this:

~~~yaml
{
    "optimization_client_name": "Name_of_your_Optimization",
    "create_logfile": true,  # determines if you want to save the log-file
    "algorithm_settings": {...},  # settings related to the algorithm
    "pulses": [{...}, {...}, ...],  # list of pulses and their settings
    "parameters": [{...}, {...}, ...],  # list of parameters and their settings
    "times": [{...}, {...}, ...]  # list of times and their settings
}
~~~

These entries are keywords but you can add more entries to the dictionary if you want to add a comment for example.

The `optimization_client_name` determines the name of the folder where the optimization results are saved. A timestamp is prepended to the name so that each optimization run generates a separate output folder. You can find the results inside of a "QuOCS_Results" folder created in the directory from where you execute the optimization.

The `create_logfile` key determines if the full output of the Python terminal is saved to a log-file. If you have an optimization with many iterations this can become quite large, so you might want to  disable it once you know your optimizations are running reliably.

**Tipp:** You can also change specific entries in the code after reading in the .json file if you, e.g., want to sweep certain parameters or have the name of the optimization defined on runtime.


## Algorithm Settings

We focus on an example of an optimization using the dCRAB algorithm in combination with Nelder-Mead. Examples for GRAPE and a pure parameter optimization are given below.


### Example: dCRAB with Nelder-Mead

We have included any possible key in this example. Note that the values defined might not be realistic and depend on your specific optimization!

To fully understand some of the settings discussed here you might want to have a look at the following references:

* [T. Caneva, T. Calarco, and S. Montangero, *Chopped random-basis quantum optimization*, Phys. Rev. A 84, 022326, (2011)](https://doi.org/10.1103/PhysRevA.84.022326)
* [N. Rach et al., *Dressing the chopped-random-basis optimization: A bandwidth-limited access to the trap-free landscape*, Phys. Rev. A 92, 062343, (2015)](https://doi.org/10.1103/PhysRevA.92.062343)
* [Matthias M Müller et al, *One decade of quantum optimal control in the chopped random basis*, Rep. Prog. Phys. 85, 076001, (2022)](https://doi.org/10.1088/1361-6633/ac723c)

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
        "compensate_after_minutes": 0.01
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
            "frtol": 1e-14,
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
|**"algorithm_name"** |*string*| The name of the QOC algorithm. So far contains "dCRAB", "GRAPE" and "AD" (automatic differentiation). Can bee extended (see [Customization](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Customization.md)). |
|**"optimization_direction"** |*string*| "minimization" or "maximization" |
|**"super_iteration_number"** |*Int*| Maximum number of super-iterations to perform |
|**"max_eval_total"** |*Int*| Maximum number of function evaluations (in total) to perform |
|**"total_time_lim"** |*float*| Time limit in minutes for the total optimizaiton run |
|**"FoM_goal"** |float| Stop the optimization when this FoM value is reached |
|**"compensate_drift"** |*dict*| Compensation for drifting FoM in an experiment. "compensate_after_SI" re-calibrates the current best FoM after each SI. "compensate_after_minutes" periodically re-calibrates the current best FoM after the given value in minutes. |
|**"random_number_generator"** |*dict*| "seed_number" specifies a seed for the selection of randomized values during the optimization. Useful for generating reproducible runs during finding phase for optimization parameters. |
|**"re_evaluation"** |*dict*| Re-evaluate the measured pulse (in the case of a noisy measurement) to improve measurement uncertainty. Under "re_evaluation_steps" one defines a list of probabilites. Given the standard deviation of a measurement and the measured FoM the evaluation is repeated if the probability is that a set of parameters might actually be a new optimum considering the measurement uncertainty. The measurement is repeated maximally as often as the length of the given list. |


#### Direct Search Method Settings

In the general settings one can define the search method to be used and related, specific parameters.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"dsm_algorithm_name"** |*string*| Name of the dsm. So far one can pick from "NelderMead" and "CMAES". Can bee extended (see [Customization](https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/Customization.md)). |

Specific to Nelder-Mead:

| Setting | Type | Explanation |
| --- | --- | --- |
|**"is_adaptive"** |*bool*| If you want to use an adaptive version of Nelder-Mead based on the simplex size according to [Gao, F., Han, L., *Implementing the Nelder-Mead simplex algorithm with adaptive parameters*, Comput. Optim. Appl. 51, (2012)](https://doi.org/10.1007/s10589-010-9329-3). |

Specific to CMA-ES:

| Setting | Type | Explanation |
| --- | --- | --- |
|**"population"** |*float*| Number of candidates per generation (iteration) of new individuals (see [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES). |


##### Stopping Criteria (dsm)

The stopping criteria for the dsm depend on the chosen search method.

| Setting | Type | Explanation |
| --- | --- | --- |
|**"max_eval"** |*Int*| Maximum number of function evaluations per SI. |
|**"time_lim"** |*float*| Time limit in minutes for singe SI. |
|**"xatol"** |*float*| Absolute change of simplex between iterations that is acceptable for convergence (see [here](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)). |
|**"frtol"** |*float*| Absolute change in FoM between iterations that is acceptable for convergence (see [here](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)). |
|**"change_based_stop"** |*dict*| Stop the search if the FoM value does not change more than defined in "cbs_change" (on average) over a number of function evaluations given by "cbs_funct_evals". |


### Example: GRAPE

~~~yaml
"algorithm_settings": {
    "algorithm_name": "GRAPE"
}
~~~

For a GRAPE optimization you only need to define the algorithm name as "GRAPE".


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
            "frtol": 1e-4
        }
    }
}
~~~

