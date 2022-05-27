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

from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
from quocslib.Optimizer import Optimizer

import numpy as np


def test_dCRAB_with_maximization():

    optimization_dictionary = {
        "Comment":
        "This is a test dictionary for the controls: dCRAB, Fourier, Uniform Distribution.",
        "Disclaimer":
        "Do not use this json file for optimization",
        "optimization_client_name":
        "Optimization_dCRAB_test_maximization",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 1000,
            "optimization_direction": "maximization",
            "FoM_goal": 0.99999,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "xatol": 1e-2,
                    "frtol": 1e-2
                }
            }
        },
        "pulses": [{
            "pulse_name": "Pulse_1",
            "upper_limit": 15.0,
            "lower_limit": -15.0,
            "bins_number": 101,
            "time_name": "time_1",
            "amplitude_variation": 0.3,
            "basis": {
                "basis_name": "Fourier",
                "basis_vector_number": 2,
                "random_super_parameter_distribution": {
                    "distribution_name": "Uniform",
                    "lower_limit": 0.1,
                    "upper_limit": 5.0
                }
            },
            "scaling_function": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: 1.0 + 0.0*t"
            },
            "initial_guess": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: np.pi/3.0 + 0.0*t"
            }
        }],
        "parameters": [],
        "times": [{
            "time_name": "time_1",
            "initial_value": 3.0
        }]
    }

    # define some parameters for the optimization
    args_dict = {
        "initial_state": "[1.0 , 0.0]",
        "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]",
        "optimization_factor": -1.0
    }

    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)
    optimization_obj.execute()

    opt_alg_obj = optimization_obj.get_optimization_algorithm()
    # Get the final results
    FoM = (opt_alg_obj._get_final_results())["Figure of merit"]
    # Get the best controls and check if they correspond to the best FoM
    opt_alg_obj = optimization_obj.get_optimization_algorithm()
    controls = opt_alg_obj.get_best_controls()
    FoM_check = FoM_object.get_FoM(**controls)["FoM"]
    # Check if the FoM calculated during the optimization is consistent with the one calculated after the optimization
    # using the best controls
    assert (np.abs(FoM - FoM_check) < 10**(-8))


def test_dCRAB_with_minimization():

    optimization_dictionary = {
        "Comment":
        "This is a test dictionary for the controls: dCRAB, Fourier, Uniform Distribution.",
        "Disclaimer":
        "Do not use this json file for optimization",
        "optimization_client_name":
        "Optimization_dCRAB_test_minimization",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 1000,
            "optimization_direction": "minimization",
            "FoM_goal": 0.00001,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "xatol": 1e-2,
                    "frtol": 1e-2
                }
            }
        },
        "pulses": [{
            "pulse_name": "Pulse_1",
            "upper_limit": 15.0,
            "lower_limit": -15.0,
            "bins_number": 101,
            "time_name": "time_1",
            "amplitude_variation": 0.3,
            "basis": {
                "basis_name": "Fourier",
                "basis_vector_number": 2,
                "random_super_parameter_distribution": {
                    "distribution_name": "Uniform",
                    "lower_limit": 0.1,
                    "upper_limit": 5.0
                }
            },
            "scaling_function": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: 1.0 + 0.0*t"
            },
            "initial_guess": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: np.pi/3.0 + 0.0*t"
            }
        }],
        "parameters": [],
        "times": [{
            "time_name": "time_1",
            "initial_value": 3.0
        }]
    }

    # define some parameters for the optimization
    args_dict = {
        "initial_state": "[1.0 , 0.0]",
        "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]",
        "optimization_factor": -1.0
    }

    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)
    optimization_obj.execute()

    opt_alg_obj = optimization_obj.get_optimization_algorithm()
    # Get the final results
    FoM = (opt_alg_obj._get_final_results())["Figure of merit"]
    # Get the best controls and check if they correspond to the best FoM
    opt_alg_obj = optimization_obj.get_optimization_algorithm()
    controls = opt_alg_obj.get_best_controls()
    FoM_check = FoM_object.get_FoM(**controls)["FoM"]
    # Check if the FoM calculated during the optimization is consistent with the one calculated after the optimization
    # using the best controls
    assert (np.abs(FoM - FoM_check) < 10**(-8))
