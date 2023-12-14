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
import numpy as np
from quocslib.utils.inputoutput import readjson


def update_opti_dict(optimization_dict: dict, comm_obj: object) -> dict:
    """
    Load optimal results of previous run and initial guess into an optimization dictionary
    :param dict optimization_dictionary: optimization_dictionary to be updated
    :comm_obj: communication object of optimization
    :return dict: updated optimization dictionary
    """

    try: 
        if optimization_dict["dump_format"] == "json":
            best_res_path = os.path.join(comm_obj.results_path, comm_obj.date_time + "_best_controls.json")
            best_res = readjson(best_res_path)
        else:
            best_res_path = os.path.join(comm_obj.results_path, comm_obj.date_time + "_best_controls.npz")
            best_res = np.load(best_res_path)
        
        for pulse in optimization_dict["pulses"]:   # use same optimization dictionary as in previous optimization
            pulse_name = pulse["pulse_name"]        # make sure to use the same pulse names in opti_dict as in best_controls
            prev_opt_pulse = best_res[pulse_name]
            initial_guess = {"function_type": "list_function", "list_function": prev_opt_pulse}
            pulse["initial_guess"] = initial_guess
            comm_obj.logger.info(f"Initial guess for pulse {pulse_name} imported from previous results")

        for param in optimization_dict["parameters"]:
            param_name = param["parameter_name"]
            prev_opt_param = best_res[param_name]
            param["initial_value"] = prev_opt_param
            comm_obj.logger.info(f"Initial guess for parameter {param_name} imported from previous results")

    except:
        comm_obj.logger.warn("Previous optimal controls could not be imported for continuation."
                                    " Check if ..._best_controls...-file exists and pulse/paremeter names coincide with optimization dictionary")
        
    return optimization_dict


