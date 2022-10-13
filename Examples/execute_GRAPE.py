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
from IsingModelProblem import IsingModel
import numpy as np
import time


def main(optimization_dictionary: dict):

    args_dict = {"n_qubits": 5, "J": 1, "g": 2, "n_slices": 100, "T": 1.0, 
                 "g_seed": 0, "g_variation": 1, "stdev": 0.1}

    optimization_dictionary["optimization_client_name"] = "Optimization_GRAPE_IsingModel_{}_bins".format(args_dict['n_slices'])

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


if __name__ == "__main__":
    main(readjson(os.path.join(os.getcwd(), "settings_GRAPE.json")))
