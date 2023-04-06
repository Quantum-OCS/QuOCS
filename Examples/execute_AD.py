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

import os, sys

from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
from IsingModelProblem_AD import IsingModel
import numpy as np
import time
import matplotlib.pyplot as plt


def main(optimization_dictionary: dict):

    args_dict = {"n_qubits": 5, "J": 1, "g": 2, "n_slices": 100, "T": 1.0, 
                 "g_seed": 0, "g_variation": 1, "stdev": 0.1}

    optimization_dictionary["optimization_client_name"] = "Optimization_AD_IsingModel_{}_bins".format(args_dict['n_slices'])

    optimization_dictionary['pulses'][0]['basis']['bins_number'] = args_dict['n_slices']
    optimization_dictionary['pulses'][0]['bins_number'] = args_dict['n_slices']


    # Create FoM object
    FoM_object = IsingModel(args_dict=args_dict)

    # Define Optimizer
    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    t1 = time.time()

    optimization_obj.execute()

    t2 = time.time()

    optimization_time = t2 - t1

    with open(
        os.path.join(optimization_obj.results_path, "optimization_time.txt"), "w"
    ) as f:
        f.write("# Time for optimization in seconds:\n")
        f.write(str(optimization_time))

    # fomlist = [element * (-1) for element in optimization_obj.fom_list]
    fomlist = [element for element in FoM_object.FoM_list]
    np.savetxt(os.path.join(optimization_obj.results_path, "fom.txt"), fomlist)

    #######################################################################
    ### Optional: Visualization of the FoM evolution and optimized controls
    #######################################################################

    ### Get the optimization algorithm object from the optimization object
    opt_alg_obj = optimization_obj.get_optimization_algorithm()

    ### The FoM values for each function evaluation can be founf under FoM_list in the optimization algorithm object
    fomlist = opt_alg_obj.FoM_list

    ### Plot the FoM over the number of evaluations
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    num_eval = range(1, len(fomlist)+1)
    ax.plot(num_eval, np.asarray(fomlist), color='darkblue', linewidth=1.5, zorder=10)
    ax.scatter(num_eval, np.asarray(fomlist), color='k', s=15)
    plt.grid(True, which="both")
    plt.xlabel('Function Evaluation', fontsize=20)
    plt.ylabel('FoM', fontsize=20)
    plt.savefig(os.path.join(optimization_obj.results_path, 'FoM.png'))

    ### The optimized controls can be found via the function get_best_controls() called on 
    ### the optimization algorithm object
    controls = opt_alg_obj.get_best_controls()

    ### it contains the pulses and time grids under certain keys as a dictionary
    pulse, timegrid = controls["pulses"][0], controls["timegrids"][0]

    ### Plot the pulse over time
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.15, top=0.9, right=0.98, left=0.1)
    plt.step(timegrid, pulse, color='darkgreen', linewidth=1.5, zorder=10)
    plt.grid(True, which="both")
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)
    plt.savefig(os.path.join(optimization_obj.results_path, 'Controls.png'))
    # plt.show()


if __name__ == "__main__":
    main(readjson(os.path.join(os.getcwd(), "settings_AD.json")))
