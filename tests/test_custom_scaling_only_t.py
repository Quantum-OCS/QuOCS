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
from quocslib.optimalcontrolproblems.IsingModelProblem import IsingModel
from quocslib.Optimizer import Optimizer
import matplotlib.pyplot as plt
import numpy as np
import sys, os, platform


def plot_FoM(result_path, FoM_filename):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    file_path = os.path.join(result_path, FoM_filename)
    save_name = "FoM_" + opt_name

    FoM = [line.rstrip('\n') for line in open(file_path)]
    FoM = [float(f) for f in FoM]
    num_eval = range(1, len(FoM) + 1)
    # print('\nInitial FoM: %.4f' % FoM[0])
    # print('Final FoM: %.4f \n' % FoM[-1])
    min_FoM = min(FoM)
    max_FoM = max(FoM)
    difference = abs(max_FoM - min_FoM)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    plt.plot(num_eval, FoM, color='darkblue', linewidth=1.5, zorder=10)
    # plt.scatter(x, y, color='k', s=15)

    plt.grid(True, which="both")
    plt.ylim(min_FoM - 0.05 * difference, max_FoM + 0.05 * difference)
    plt.xlabel('Function Evaluation', fontsize=20)
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

    plt.step(time_grid, pulse, color='darkgreen', linewidth=1.5, zorder=10)
    plt.grid(True, which="both")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    # plt.savefig(os.path.join(folder, save_name + '.pdf'))
    plt.savefig(os.path.join(result_path, save_name + '.png'))


def test_custom_scaling_only_t():
    optimization_dictionary = {
        "Disclaimer":
        "Do not use this json file for optimization",
        "optimization_client_name":
        "Optimization_test_custom_scaling_only_t",
        "algorithm_settings": {
            "algorithm_name": "dCRAB",
            "super_iteration_number": 2,
            "max_eval_total": 1000,
            "optimization_direction": "minimization",
            "dsm_settings": {
                "general_settings": {
                    "dsm_algorithm_name": "NelderMead"
                },
                "stopping_criteria": {
                    "xatol": 1e-4,
                    "fatol": 1e-6,
                    "change_based_stop": {
                        "cbs_funct_evals": 200,
                        "cbs_change": 0.05
                    },
                    "max_eval": 500
                }
            }
        },
        "pulses": [
            {
                "pulse_name": "Pulse_1",
                "bins_number": 2000,
                "upper_limit": 1000.0,
                "lower_limit": -1000.0,
                "time_name": "time_1",
                "amplitude_variation": 30.0,
                # "initial_guess": {
                #     "function_type": "lambda_function",
                #     "lambda_function": "lambda t: 0.0 * t"
                # },
                "initial_guess": {
                    "function_type": "python_file",
                    "file_path": "my_file_with_functions",
                    "function_name": "guess_pulse_function",
                    "path_mode": "relative"
                },
                "scaling_function": {
                    "function_type": "python_file",
                    "file_path": "my_file_with_functions",
                    "function_name": "scaling_function",
                    "path_mode": "relative"
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
                "shaping_options": ["add_initial_guess",
                                    "add_base_pulse",
                                    "add_new_update_pulse",
                                    "scale_pulse",
                                    "limit_pulse"]
            }
        ],
        "times": [
            {
                "time_name": "time_1",
                "initial_value": 1.0
            }
        ],
        "parameters": []
    }

    optimization_dictionary.setdefault("optimization_direction", "minimization")
    # define some parameters for the optimization
    args_dict = {}
    main(optimization_dictionary, args_dict)


def main(optimization_dictionary: dict, args_dict: dict):
    # Create FoM object
    FoM_object = IsingModel(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)
    optimization_obj.execute()

    fomlist = [element for element in FoM_object.FoM_list]
    np.savetxt(os.path.join(optimization_obj.results_path, "fom.txt"), fomlist)

    plot_FoM(optimization_obj.results_path, "fom.txt")
    plot_controls(optimization_obj.results_path)


if __name__ == "__main__":
    test_custom_pulse_build_rel_path()
