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

import os
if os.name != 'nt':
    from quocslib.optimalcontrolproblems.IsingModelADProblem import IsingModel
from quocslib.Optimizer import Optimizer
import pytest
import numpy as np


def test_AD_Ising_Model():
    if os.name != 'nt':
        optimization_dictionary = {
            "Disclaimer":
            "Do not use this json file for optimization",
            "optimization_client_name":
            "Optimization_AD_IsingModel",
            "algorithm_settings": {
                "algorithm_name": "AD",
                "optimization_direction": "maximization",
                "stopping_criteria": {
                    "max_eval_total": 100,
                    "ftol": 1e-4,
                    "gtol": 1e-5
                }
            },
            "pulses": [{
                "pulse_name": "Pulse_1",
                "upper_limit": 100.0,
                "lower_limit": -100.0,
                "bins_number": 100,
                "amplitude_variation": 30.0,
                "time_name": "time_1",
                "basis": {
                    "basis_name": "PiecewiseBasis",
                    "bins_number": 100
                }
            }],
            "parameters": [],
            "times": [{
                "time_name": "time_1",
                "initial_value": 1.0
            }]
        }

        optimization_dictionary.setdefault("optimization_direction", "maximization")
        # define some parameters for the optimization
        args_dict = {}
        main(optimization_dictionary, args_dict)


def main(optimization_dictionary: dict, args_dict: dict):
    # Create FoM object
    FoM_object = IsingModel(args_dict=args_dict)

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
    assert (np.abs(FoM - FoM_check) < 5 * 10**(-5))


if __name__ == "__main__":
    test_AD_Ising_Model()
