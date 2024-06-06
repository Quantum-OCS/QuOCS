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

import os, platform
import matplotlib.pyplot as plt
import numpy as np
from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
from quocslib.optimalcontrolproblems.IsingModelProblem import IsingModel
from quocslib.Optimizer import Optimizer
import pytest


def plot_FoM(result_path, FoM):

    save_name = "FoM_plot" 

    iterations = range(1, len(FoM) + 1)
    min_FoM = min(FoM)
    max_FoM = max(FoM)
    difference = abs(max_FoM - min_FoM)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(iterations, FoM, color='darkblue', linewidth=1.5, zorder=10)

    plt.grid(True, which="both")
    plt.ylim(min_FoM - 0.05 * difference, max_FoM + 0.05 * difference)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('FoM', fontsize=20)
    plt.savefig(os.path.join(result_path, save_name + '.png'))

def run_dCRAB_opti(optimization_dictionary):
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

    initial_guess = optimization_obj.get_optimization_algorithm().controls.pulse_objs_list[0].initial_guess_pulse
    initial_param = optimization_obj.get_optimization_algorithm().controls.parameter_objs_list[0].value
    
    # Run optimization

    optimization_obj.execute()

    optimal_pulse = optimization_obj.get_optimization_algorithm().get_best_controls()["pulses"][0]
    optimal_param = optimization_obj.get_optimization_algorithm().get_best_controls()["parameters"][0]
    
    fomlist = optimization_obj.get_optimization_algorithm().FoM_list
    
    res_path = optimization_obj.results_path
    datetime = optimization_obj.communication_obj.date_time

    return res_path, datetime, initial_guess, optimal_pulse, initial_param, optimal_param, fomlist


 
def test_dCRAB_continuation():

    optimization_dictionary = {
    "optimization_client_name": "continuation_test",
    "optimization_direction": "minimization",
    "continuation_datetime": "no",
    "dump_format": "json",     
    "algorithm_settings": {
        "algorithm_name": "dCRAB",
        "super_iteration_number": 2,   
        "max_eval_total": 100,         
        "FoM_goal": 0.00001,
        "dsm_settings": {
            "general_settings": {
                "dsm_algorithm_name": "NelderMead",
                "is_adaptive": True
            },
            "stopping_criteria": {
                "max_eval": 50,
            }
        },
        "random_number_generator": {
            "seed_number": 42
        }
    },
    "pulses": [{
        "pulse_name": "Pulse_1",
        "upper_limit": 5.0,
        "lower_limit": -5.0,
        "bins_number": 101,
        "time_name": "time_1",
        "amplitude_variation": 5.0,
        "basis": {
            "basis_name": "Fourier",
            "basis_vector_number": 2,
            "random_super_parameter_distribution": {
                "distribution_name": "Uniform",
                "lower_limit": 0.1,
                "upper_limit": 5.0
            }
        },
        "initial_guess": {
            "function_type": "lambda_function",
            "lambda_function": "lambda t: np.pi/3.0 + 0.0*t"
        }
    }],
    "parameters": [{"parameter_name": "Parameter0",
                       "lower_limit": -2.0,
                       "upper_limit": 2.0,
                       "initial_value": 0.4,
                       "amplitude_variation": 0.5}],
    "times": [{
        "time_name": "time_1",
        "initial_value": 3.0
    }]
    }
    res_path1, datetime1, _ , optimal_pulse1, _, optimal_param1, fomlist1  = run_dCRAB_opti(optimization_dictionary)

    optimization_dictionary["continuation_datetime"] = datetime1

    # optimization_dictionary["pulses"][0]["pulse_name"] = "new_name"

    res_path2, datetime2, inital_guess2, optimal_pulse2, initial_param2, optimal_param2, fomlist2 = run_dCRAB_opti(optimization_dictionary)

    res_path3, datetime3, inital_guess3 , _ ,initial_param3, _ , fomlist3  = run_dCRAB_opti(optimization_dictionary)

    plot_FoM(res_path2, fomlist1 + fomlist2 + fomlist3)


    assert res_path1 == res_path2 == res_path3           # test, if similar result path for both optimizations
    assert datetime1 == datetime2 == datetime3           # test for similar datetime
    assert np.array_equal(inital_guess2, optimal_pulse1) # test, if optimal pulse is given imported as initial guess
    assert np.array_equal(inital_guess3, optimal_pulse2) 
    assert initial_param2 == optimal_param1
    assert initial_param3 == optimal_param2
    assert min(fomlist3) <= min(fomlist2) <= min(fomlist1)  # test, if second optimization improved the results




def run_GRAPE_opti(optimization_dictionary):

    FoM_object = IsingModel(args_dict={})

    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    initial_guess = optimization_obj.get_optimization_algorithm().controls.pulse_objs_list[0].initial_guess_pulse    
    # Run optimization

    optimization_obj.execute()

    optimal_pulse = optimization_obj.get_optimization_algorithm().get_best_controls()["pulses"][0]    
    fomlist = optimization_obj.get_optimization_algorithm().FoM_list
    
    res_path = optimization_obj.results_path
    datetime = optimization_obj.communication_obj.date_time

    return res_path, datetime, initial_guess, optimal_pulse, fomlist


def test_GRAPE_continuation():
    optimization_dictionary = {
        "optimization_client_name": "continuation_test",
        "optimization_direction": "minimization",
        "continuation_datetime": "no",
        "dump_format": "json",   
        "algorithm_settings": {
            "algorithm_name": "GRAPE",
            "stopping_criteria": {"max_eval_total": 150}
        },
        "pulses": [{
            "pulse_name": "Pulse_1",
            "upper_limit": 100.0,
            "lower_limit": -100.0,
            "bins_number": 100,
            "amplitude_variation": 20.0,
            "time_name": "time_1",
            "basis": {
                "basis_name": "PiecewiseBasis",
                "bins_number": 100
            },
            "initial_guess": {
                "function_type": "lambda_function",
                "lambda_function": "lambda t: 0.0 + 0.0*t"
            }
        }],
        "parameters": [],
        "times": [{
            "time_name": "time_1",
            "initial_value": 1.0
        }]
    }

    res_path1, datetime1, _ , optimal_pulse1, fomlist1  = run_GRAPE_opti(optimization_dictionary)

    optimization_dictionary["continuation_datetime"] = datetime1

    # optimization_dictionary["pulses"][0]["pulse_name"] = "new_name"

    res_path2, datetime2, inital_guess2, optimal_pulse2, fomlist2 = run_GRAPE_opti(optimization_dictionary)

    res_path3, datetime3, inital_guess3 , _ , fomlist3  = run_GRAPE_opti(optimization_dictionary)

    plot_FoM(res_path2, fomlist1 + fomlist2 + fomlist3)


    assert res_path1 == res_path2 == res_path3           # test, if similar result path for both optimizations
    assert datetime1 == datetime2 == datetime3           # test for similar datetime
    assert np.array_equal(inital_guess2, optimal_pulse1) # test, if optimal pulse is given imported as initial guess
    assert np.array_equal(inital_guess3, optimal_pulse2) 
    assert min(fomlist3) <= min(fomlist2) <= min(fomlist1)  # test, if second optimization improved the results

test_dCRAB_continuation()