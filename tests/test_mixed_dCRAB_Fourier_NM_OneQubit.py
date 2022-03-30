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
from quocslib.utils.inputoutput import readjson
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
        if "time" in data_name:
            time_grid = controls[data_name]
        elif "pulse" in data_name:
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


def main(optimization_dictionary: dict, args_dict: dict):

    # Create FoM object
    FoM_object = OneQubit(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    # Execute the optimization and save the FoM
    FoM_object.set_save_path(optimization_obj.results_path)
    optimization_obj.execute()
    FoM_object.save_FoM()

    # Plot the results
    plot_FoM(FoM_object.save_path, FoM_object.FoM_save_name)
    plot_controls(FoM_object.save_path)


# --------------------------------------------------------------------

# if __name__ == '__main__':
#     # get the optimization settings from the json dictionary
#     optimization_dictionary = readjson(os.path.join(os.getcwd(), "dCRAB_Fourier_NM_OneQubit.json"))[1]
#     # define some parameters for the optimization
#     args_dict = {"initial_state": "[1.0 , 0.0]",
#                  "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]"}
#     main(optimization_dictionary, args_dict)


def test_mixed_dCRAB_Fourier_NM_OneQubit():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "mixed_dCRAB_Fourier_NM_OneQubit.json"))[1]
    # define some parameters for the optimization
    args_dict = {"initial_state": "[1.0 , 0.0]",
                 "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]"}
    main(optimization_dictionary, args_dict)

# --------------------------------------------------------------------

# if __name__ == '__main__':
#     # get the optimization settings from the json dictionary
#     optimization_dictionary = readjson(os.path.join(os.getcwd(), "dCRAB_Fourier_NM_OneQubit_Noisy.json"))[1]
#     # define some parameters for the optimization
#     args_dict = {"initial_state": "[1.0 , 0.0]",
#                  "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]",
#                  "is_noisy": True}
#     main(optimization_dictionary, args_dict)


