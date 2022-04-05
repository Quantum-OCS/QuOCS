# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

from quocslib.Controls import Controls
from quocslib.pulses.basis.Fourier import Fourier
from quocslib.pulses.super_parameter.Uniform import Uniform
"""
Test for controls initialization using an external basis and control distribution
"""


def main(controls_dict):
    # Modify the controls with the attribute field
    controls_dict["pulses"][0]["basis"]["basis_attribute"] = Fourier
    controls_dict["pulses"][0]["basis"]["random_super_parameter_distribution"]["distribution_attribute"] = Uniform
    # Initialize controls
    controls_obj = Controls(controls_dict["pulses"], controls_dict["times"], controls_dict["parameters"])
    # Set random super_parameters
    controls_obj.select_basis()
    # Sigma variation
    print("sigma_variation = {0}".format(controls_obj.get_sigma_variation()))
    # Mean value
    print("mean_value = {0}".format(controls_obj.get_mean_value()))
    # Get control lists
    controls_list = [pulses_list, time_grids_list, parameters_list] = \
        controls_obj.get_controls_lists(controls_obj.get_mean_value())
    for control in controls_list:
        print("Control: {0}".format(control))
    controls_obj.update_base_controls(controls_obj.get_mean_value())
    print("The initialization version 2 is concluded")


if __name__ == '__main__':
    control_dict = {
        "Comment":
        "This is test dictionary for the controls: dCRAB, Fourier, Uniform Distribution. 1 pulse, 1 time, 1 parameter",
        "Disclaimer":
        "Do not use this json file for optimization",
        "pulses": [{
            "pulse_name": "Pulse111",
            "upper_limit": 1.0,
            "lower_limit": -1.0,
            "bins_number": 101,
            "time_name": "time111",
            "amplitude_variation": 0.12,
            "basis": {
                "basis_class": None,
                "basis_module": None,
                "basis_attribute": None,
                "basis_vector_number": 2,
                "random_super_parameter_distribution": {
                    "distribution_class": None,
                    "distribution_module": None,
                    "distribution_attribute": None,
                    "lower_limit": 0.1,
                    "upper_limit": 5.0
                }
            },
            "scaling_function": {
                "function_type": "list_function",
                "list_function": [1.0 for _ in range(101)]
            },
            "initial_guess": {
                "function_type": "list_function",
                "list_function": [0.0 for _ in range(101)]
            }
        }],
        "parameters": [{
            "parameter_name": "Parameter111",
            "lower_limit": -2.0,
            "upper_limit": 2.0,
            "initial_value": 0.0,
            "amplitude_variation": 0.5
        }],
        "times": [{
            "time_name": "time111",
            "initial_value": 5.0
        }]
    }
    main(control_dict)
