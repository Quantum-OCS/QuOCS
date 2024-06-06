# Example: How to enforce pulse boundaries at beginning and end of pulse

Often one wants to ensure the pulse (e.g. for a voltage ramp) starts and stops at defined values. Say we want to have an initial guess of a linear ramp that starts at -20 mV and ends at 50 mV. The following pulse configuration will make sure that after the optimization these boundaries are still in tact:

~~~yaml
"pulses": [
    {
        "pulse_name": "Pulse_1",
        "bins_number": 2000,
        "upper_limit": 1000.0,
        "lower_limit": -1000.0,
        "time_name": "time_1",
        "amplitude_variation": 30.0,
        "scaling_function": {
            "function_type": "python_file",
            "file_path": "my_file_with_functions",
            "function_name": "shape_function",
            "path_mode": "relative"
        },
        "initial_guess": {
            "function_type": "lambda_function",
            "lambda_function": "lambda t: -20 + 70*t"
        },
        "basis": {
            "basis_name": "Fourier",
            "basis_vector_number": 5,
            "random_super_parameter_distribution": {
                "distribution_name": "Uniform",
                "lower_limit": 0.01,
                "upper_limit": 10.0
            }
        },
        "shaping_options": [
            "add_base_pulse",
            "add_new_update_pulse",
            "scale_pulse",
            "add_initial_guess",
            "limit_pulse"
        ]
    }
]
~~~

The pulse duration was set to 1 and we use the following scaling function:

```my_file_with_functions.py```:

~~~python
import numpy as np

T = 1
steepness = 30

def shape_function(t):
    return np.tanh(np.sin(np.pi*t/(2*T)) * steepness) * np.tanh(-np.sin(np.pi*(t-T)/(2*T)) * steepness)

~~~


Note the changed order in the shaping options! First we take the base pulse (prev SIs) and add the current OC pulse of the sub-iteration. The the pulse is scaled, putting it to 0 at the start and end. Then the guess pulse is added. The guess pulse should obey the boundaries and can be of any shape. In this case we chose a linear ramp. Finally, the pulse is limited, enforcing the amplitude limits defined.

This method only keeps the boundaries if the pulse is not rescaled but the amplitudes exceeding these limits are cut. If you want to smoothly rescale the pulse to stay inside the limits you can switch the "add_initial_guess" and "limit_pulse" settings inside of the "shaping_options". If you do this, however, you have to think about the new amplitude limits, i.e. before addition of the guess pulse! In this case you can also set "shrink_ampl_lim": True.



