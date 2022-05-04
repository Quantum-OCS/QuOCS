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
import pytest


def test_stopping_after_FoM_goal():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_FoM_goal_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 100,
            "FoM_goal": 0.1,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 50,
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


def test_stopping_after_SI_max_eval():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_SI_max_eval_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 10000,
            "FoM_goal": 0.0000001,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 50,
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


def test_stopping_after_total_max_eval():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_total_max_eval_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 100,
            "FoM_goal": 0.0000001,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 500,
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


def test_stopping_after_direct_search_time_limit():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_direct_search_time_limit_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 1000,
            "total_time_lim": 2,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "time_lim": 0.01,
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


def test_stopping_after_total_time_limit():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_total_time_limit_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 1000,
            "total_time_lim": 0.05,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "time_lim": 2,
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


def test_stopping_with_change_based_stop():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_change_based_stopping_criterion",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 1000,
            "total_time_lim": 0.05,
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "change_based_stop": {
                        "cbs_funct_evals": 30,
                        "cbs_change": 0.01
                    }
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
