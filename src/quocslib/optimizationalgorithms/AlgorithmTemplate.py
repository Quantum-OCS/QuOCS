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
import numpy as np

from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm
from quocslib.Controls import Controls
from quocslib.utils.dynamicimport import dynamic_import

from quocslib.tools.linearalgebra import simplex_creation


class AlgorithmTemplate(OptimizationAlgorithm):
    """
    This is the template for an algorithm class. The three important function are:
    * the constructor with the optimization dictionary and the communication object as parameters
    * run : The main loop for optimal control
    * _get_response_for_client : return info about the goodness of the controls and errors if any
    * _get_controls : return the set of controls as a dictionary with pulses, parameters, and times as keys
    * _get_final_results: return the final result of the optimization algorithm
    """
    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the dCRAB algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm class except the optimization dictionary where the dCRAB settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj)
        ###########################################################################################
        # Inner free gradient method
        ###########################################################################################
        stopping_criteria = optimization_dict["algorithm_settings"]["dsm_settings"]["stopping_criteria"]
        # put global time limit into stopping_criteria so we don't have to pass it through functions
        optimization_dict["algorithm_settings"].setdefault("total_time_lim", 10**10)
        stopping_criteria.setdefault("total_time_lim", optimization_dict["algorithm_settings"]["total_time_lim"])
        direct_search_method_settings = optimization_dict["algorithm_settings"]["dsm_settings"]["general_settings"]
        dsm_attribute = dynamic_import(
            class_name="GradientFreeTemplate",
            module_name="quocslib.gradientfreemethods.GradientFreeTemplate",
        )
        self.dsm_obj = dsm_attribute(
            direct_search_method_settings,
            stopping_criteria,
            callback=self.is_optimization_running,
        )
        ###########################################################################################
        # Optimal algorithm variables if any
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Starting FoM
        self.best_FoM = 1e10
        ###########################################################################################
        # Pulses, Parameters, Times object
        ###########################################################################################
        # Initialize the control object
        self.controls = Controls(
            optimization_dict["pulses"],
            optimization_dict["times"],
            optimization_dict["parameters"],
        )

    def _get_response_for_client(self) -> dict:
        """Return useful information for th interface"""
        is_record = False
        FoM = self.FoM_dict["FoM"]
        if FoM < self.best_FoM:
            self.best_FoM = FoM
            is_record = True
        response_dict = {
            "is_record": is_record,
            "FoM": FoM,
            "iteration_number": self.iteration_number,
        }
        return response_dict

    def run(self) -> None:
        """Main loop of the optimization"""
        for super_it in range(1, 2):
            # Check if the optimization was stopped by the user
            if not self.is_optimization_running():
                return
            # Initialize the random super_parameters
            self.controls.select_basis()
            # Direct search method
            if super_it == 1:
                self._dsm_build(self.max_num_function_ev)
            else:
                self._dsm_build(self.max_num_function_ev2)
            # Update the base current pulses
            self._update_base_pulses()

    def _update_base_pulses(self) -> None:
        """Update the base dCRAB pulse"""
        self.controls.update_base_controls(self.xx)

    def _dsm_build(self, max_iteration_number: int) -> None:
        """Build the direct search method and run it"""
        start_simplex = simplex_creation(self.controls.get_mean_value(), self.controls.get_sigma_variation())
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call,
                                        x0,
                                        initial_simplex=start_simplex,
                                        max_eval_total=max_iteration_number)
        # Update the results
        [FoM, self.xx, self.terminate_reason] = [
            result_l["F_min_val"],
            result_l["X_opti_vec"],
            result_l["terminate_reason"],
        ]

    def _get_controls(self, xx: np.array) -> dict:
        """Get the controls dictionary from the optimized control parameters"""
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(xx)
        #
        controls_dict = {
            "pulses": pulses,
            "parameters": parameters,
            "timegrids": timegrids,
        }
        return controls_dict

    def _get_final_results(self) -> dict:
        """Return a dictionary with final results to put into a dictionary"""
        final_dict = {
            "Figure of merit": self.best_FoM,
            "total number of function evaluations": self.iteration_number,
        }
        return final_dict
