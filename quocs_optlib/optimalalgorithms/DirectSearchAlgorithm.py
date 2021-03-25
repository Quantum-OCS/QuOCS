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

from quocs_optlib.Optimizer import Optimizer
from quocs_optlib.Controls import Controls
from quocs_optlib.freegradientmethods.NelderMead import NelderMead
from quocs_optlib.tools.linearalgebra import simplex_creation


class DirectSearchAlgorithm(Optimizer):
    """

    """
    initStatus = 0
    terminate_reason = "-1"

    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        :param optimization_dict:
        :param communication_obj:
        """
        super().__init__(communication_obj=communication_obj)
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["dsm_settings"]["stopping_criteria"]
        direct_search_method_settings = optimization_dict["dsm_settings"]["general_settings"]
        self.dsm_obj = NelderMead(direct_search_method_settings, stopping_criteria,
                                  callback=self.is_optimization_running)
        ###########################################################################################
        # Optimal algorithm variables ###########################################################
        ###########################################################################################
        # Starting fom
        self.best_fom = 1e10
        ###########################################################################################
        # Pulses, Parameters object ###########################################################
        ###########################################################################################
        self.controls = Controls(optimization_dict["pulses"], optimization_dict["times"],
                                 optimization_dict["parameters"])

    def _get_response_for_client(self):
        """
        Return True if a record is found
        Returns
        -------

        """
        is_record = False
        if self.fom_dict["FoM"] < self.best_fom:
            is_record = True
        response_dict = {"is_record": is_record, "FoM": self.fom_dict["FoM"], "iteration_number": self.iteration_number}
        return response_dict

    def run(self):
        """

        Returns
        -------

        """
        # Direct search method
        self._dsm_build()

    def _dsm_build(self):
        """

        Returns
        -------

        """
        start_simplex = simplex_creation(self.controls.get_mean_value(), self.controls.get_sigma_variation())
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call, x0, initial_simplex=start_simplex)
        # Update the results
        [self.best_fom, self.xx, self.terminate_reason] = \
            [result_l['F_min_val'], result_l['X_opti_vec'], result_l["terminate_reason"]]

    def _get_controls(self, xx):
        # pulses_list, time_grids_list, parameters_list
        [], [], parameters_list = self.controls.get_controls_lists(xx)

        controls_dict = {"pulses": [], "parameters": parameters_list, "timegrids": []}
        return controls_dict

    def _get_final_results(self):
        final_dict = {"Figure of merit": self.best_fom, "parameters": self.xx,
                      "terminate_reason": self.terminate_reason}
        return final_dict
