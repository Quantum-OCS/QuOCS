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

from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm
from quocslib.Controls import Controls
from quocslib.gradientfreemethods.NelderMead import NelderMead
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.tools.linearalgebra import simplex_creation


class DirectSearchAlgorithm(OptimizationAlgorithm):
    """ """

    initStatus = 0
    terminate_reason = "-1"

    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        :param optimization_dict:
        :param communication_obj:
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["algorithm_settings"]["dsm_settings"]["stopping_criteria"]
        # put global time limit into stopping_criteria so we don't have to pass it through functions
        optimization_dict["algorithm_settings"].setdefault("total_time_lim", 10**10)
        stopping_criteria.setdefault("total_time_lim", optimization_dict["algorithm_settings"]["total_time_lim"])
        direct_search_method_settings = optimization_dict["algorithm_settings"]["dsm_settings"]["general_settings"]
        if "dsm_name" in direct_search_method_settings:
            print("dsm_name is used direct search methods. This option is deprecated. Use \n"
                  "dsm_algorithm_module: quocslib.freegradients.NelderMead\n"
                  "dsm_algorithm_class: NelderMead")
            self.dsm_obj = NelderMead(direct_search_method_settings,
                                      stopping_criteria,
                                      callback=self.is_optimization_running)
        else:
            dsm_attribute = dynamic_import(
                module_name=direct_search_method_settings.setdefault("dsm_algorithm_module", None),
                class_name=direct_search_method_settings.setdefault("dsm_algorithm_class", None),
                name=direct_search_method_settings.setdefault("dsm_algorithm_name", None),
                class_type='dsm_settings')

            self.dsm_obj = dsm_attribute(direct_search_method_settings,
                                         stopping_criteria,
                                         callback=self.is_optimization_running)
        ###########################################################################################
        # Optimal algorithm variables ###########################################################
        ###########################################################################################
        # Starting FoM
        self.best_FoM = 1e10
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
        if self.FoM_dict["FoM"] < self.best_FoM:
            is_record = True
        response_dict = {"is_record": is_record, "FoM": self.FoM_dict["FoM"], "iteration_number": self.iteration_number}
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
        # Initialize the best xx vector for this SI
        self.best_xx = self.controls.get_mean_value().copy()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call,
                                        x0,
                                        initial_simplex=start_simplex,
                                        sigma_v=self.controls.get_sigma_variation())
        # Update the results
        [self.best_FoM, self.xx,
         self.terminate_reason] = [result_l["F_min_val"], result_l["X_opti_vec"], result_l["terminate_reason"]]

    def _get_controls(self, xx):
        # pulses_list, time_grids_list, parameters_list
        [], [], parameters_list = self.controls.get_controls_lists(xx)

        controls_dict = {"pulses": [], "parameters": parameters_list, "timegrids": []}
        return controls_dict

    def _get_final_results(self):
        final_dict = {
            "Figure of merit": self.best_FoM,
            "parameters": self.xx,
            "terminate_reason": self.terminate_reason
        }
        return final_dict
