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

import os, sys, platform
import matplotlib.pyplot as plt
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
#from IsingModelProblem_noise import IsingModel
from IsingModelProblem import IsingModel
import numpy as np
import time
import statistics


def plot_FoM(result_path, FoM_filename):

    if 'Windows' in platform.platform():
        opt_name = result_path.split('\\')[-1]
    else:
        opt_name = result_path.split('/')[-1]

    file_path = os.path.join(result_path, FoM_filename)
    save_name = "FoM_" + opt_name

    FoM = [line.rstrip('\n') for line in open(file_path)]
    FoM = [1-float(f) for f in FoM]
    #FoM = FoM
    num_eval = range(1, len(FoM) + 1)
    # print('\nInitial FoM: %.4f' % FoM[0])
    # print('Final FoM: %.4f \n' % FoM[-1])
    min_FoM = min(FoM)
    max_FoM = max(FoM)
    difference = abs(max_FoM - min_FoM)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)

    # Compute cumulative maximum for the overlayed plot
    best_FoM = np.minimum.accumulate(FoM)
    # Overlay the cumulative maximum (monotonically increasing) plot
    plt.plot(num_eval, best_FoM, color='red', linestyle='--', linewidth=1.5, label='Best FoM', zorder=11)

    plt.plot(num_eval, FoM, color='darkblue', linewidth=1.5, zorder=10)
    # plt.scatter(x, y, color='k', s=15)

    plt.grid(True, which="both")
    plt.yscale('log')
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


def main(optimization_dictionary: dict):

    args_dict = {"n_qubits": 5, "J": 1, "g": 2, "N_slices": 2000, "T": 1.0,
                 "g_seed": 123, "g_variation": 0.3, "stdev": 0.01}

    optimization_dictionary["pulses"][0]["bins_number"] = args_dict["N_slices"]
    optimization_dictionary["times"][0]["initial_value"] = args_dict["T"]

    if args_dict["g_seed"] != 0:
        optimization_dictionary["algorithm_settings"]["re_evaluation"] = {"re_evaluation_steps": [0.3, 0.5, 0.501]}

    # Create FoM object
    FoM_object = IsingModel(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    t1 = time.time()

    optimization_obj.execute()

    t2 = time.time()

    optimization_time = t2 - t1

    with open(os.path.join(optimization_obj.results_path, "optimization_time.txt"), "w") as f:
        f.write("# Time for optimization in seconds:\n")
        f.write(str(optimization_time))

    # fomlist = [element * (-1) for element in optimization_obj.fom_list]
    fomlist = [element for element in FoM_object.FoM_list]
    np.savetxt(os.path.join(optimization_obj.results_path, "fom.txt"), fomlist)

    plot_FoM(optimization_obj.results_path, "fom.txt")
    plot_controls(optimization_obj.results_path)

    opt_controls = optimization_obj.opt_alg_obj.get_best_controls()

    if args_dict["g_seed"] != 0:
        statistics_fom_list = []
        num_for_average = 50
        for i in range(num_for_average):
            statistics_fom_list.append(FoM_object.get_FoM(**opt_controls)["FoM"]*(-1))

        mittel = statistics.mean(statistics_fom_list)
        deviation = statistics.stdev(statistics_fom_list)

        with open(os.path.join(optimization_obj.results_path, "statistics.txt"), 'w') as f:
            f.write('averaged over {} evals\n'.format(num_for_average))
            f.write('mean:{}\n'.format(mittel))
            f.write('stdev: {}\n'.format(deviation))

    print("\nBest FoM: {}".format(optimization_obj.opt_alg_obj.best_FoM))


if __name__ == "__main__":
    main(readjson(os.path.join(os.getcwd(), "settings_dCRAB.json")))
