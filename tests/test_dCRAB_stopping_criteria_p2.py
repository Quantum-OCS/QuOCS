from quocslib.Optimizer import Optimizer
from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
import pytest


def test_stopping_after_total_max_eval_with_few_iterations():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_total_max_eval_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 10,
            "FoM_goal": 0.0000001,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 20,
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
                "basis_vector_number": 6,
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

    optimization_dictionary.setdefault("optimization_direction", "minimization")
    # define some parameters for the optimization
    args_dict = {
        "initial_state": "[1.0 , 0.0]",
        "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]",
        "optimization_factor": -1.0
    }
    # Create FoM object
    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)
    optimization_obj.execute()

    optimization_obj.results_path