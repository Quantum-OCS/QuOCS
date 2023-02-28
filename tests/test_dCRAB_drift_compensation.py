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
from quocslib.Optimizer import Optimizer
import pytest


def plot_FoM(result_path, FoM_filename):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    file_path = os.path.join(result_path, FoM_filename)
    save_name = "FoM_" + opt_name

    FoM = [line.rstrip('\n') for line in open(file_path)]
    FoM = [float(f) for f in FoM]
    iterations = range(1, len(FoM) + 1)
    # print('\nInitial FoM: %.4f' % FoM[0])
    # print('Final FoM: %.4f \n' % FoM[-1])
    min_FoM = min(FoM)
    max_FoM = max(FoM)
    difference = abs(max_FoM - min_FoM)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(iterations, FoM, color='darkblue', linewidth=1.5, zorder=10)
    # plt.scatter(x, y, color='k', s=15)

    plt.grid(True, which="both")
    plt.ylim(min_FoM - 0.05 * difference, max_FoM + 0.05 * difference)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('FoM', fontsize=20)
    # plt.savefig(os.path.join(folder, save_name + '.pdf'))
    plt.savefig(os.path.join(result_path, save_name + '.png'))


def plot_controls(result_path):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    for file in os.listdir(result_path):
        if file.endswith('best_controls.npz'):
            file_path = os.path.join(result_path, file)

    save_name = "Controls_" + opt_name

    controls = np.load(file_path)

    time_grid = []
    pulse = []

    for data_name in controls.files:
        if "time_grid_for_Pulse_1" in data_name:
            time_grid = controls[data_name]
        elif "Pulse_1" in data_name:
            pulse = controls[data_name]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(time_grid, pulse, color='darkgreen', linewidth=1.5, zorder=10)
    plt.grid(True, which="both")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    # plt.savefig(os.path.join(folder, save_name + '.pdf'))
    plt.savefig(os.path.join(result_path, save_name + '.png'))


def test_drift_compensation_average_3_FoM():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_drift_compensation_average_3_FoM",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 200,
            "FoM_goal": 0.00001,
            "compensate_drift": {
                "compensate_after_SI": True,
                "compensate_after_minutes": 0.01,
                "num_average": 3
            },
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 100,
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
        "optimization_factor": -1.0,
        "include_drift": True
    }
    # Create FoM object
    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    FoM_object.set_save_path(optimization_obj.results_path)

    optimization_obj.execute()

    FoM_object.save_FoM()

    # Plot the results
    plot_FoM(FoM_object.save_path, FoM_object.FoM_save_name)
    plot_controls(FoM_object.save_path)

def test_drift_compensation_single_average():

    optimization_dictionary = {
        "optimization_client_name":
        "Check_drift_compensation_single_average",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 3,
            "max_eval_total": 200,
            "FoM_goal": 0.00001,
            "compensate_drift": {
                "compensate_after_SI": True,
                "compensate_after_minutes": 0.01,
            },
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead",
                    "is_adaptive": True
                },
                "stopping_criteria": {
                    "max_eval": 100,
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
        "optimization_factor": -1.0,
        "include_drift": True
    }
    # Create FoM object
    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    FoM_object.set_save_path(optimization_obj.results_path)

    optimization_obj.execute()

    FoM_object.save_FoM()

    # Plot the results
    plot_FoM(FoM_object.save_path, FoM_object.FoM_save_name)
    plot_controls(FoM_object.save_path)
