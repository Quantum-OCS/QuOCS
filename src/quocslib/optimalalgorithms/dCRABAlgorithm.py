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

from quocslib.Optimizer import Optimizer
from quocslib.Controls import Controls
from quocslib.freegradientmethods.NelderMead import NelderMead
from quocslib.tools.linearalgebra import simplex_creation


class DCrabAlgorithm(Optimizer):
    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the dCRAB algorithm. All the arguments in the constructor are passed to the
        Optimizer class except the optimization dictionary where the dCRAB settings and the controls are defined.
        """
        super().__init__(
            communication_obj=communication_obj, optimization_dict=optimization_dict
        )
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["dsm_settings"]["stopping_criteria"]
        direct_search_method_settings = optimization_dict["dsm_settings"][
            "general_settings"
        ]
        # TODO Use dynamic import here to define the inner free gradient method
        # The callback function is called once in a while in the inner direct search method to check
        #  if the optimization is still running
        self.dsm_obj = NelderMead(
            direct_search_method_settings,
            stopping_criteria,
            callback=self.is_optimization_running,
        )
        self.terminate_reason = ""
        ###########################################################################################
        # Optimal algorithm variables
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Max number of SI
        self.max_num_si = int(alg_parameters["super_iteration_number"])
        # TODO change evaluation number for the first and second super iteration
        # Max number of iterations at SI1
        self.max_num_function_ev = int(
            alg_parameters["maximum_function_evaluations_number"]
        )
        # Max number of iterations from SI2
        self.max_num_function_ev2 = int(
            alg_parameters["maximum_function_evaluations_number"]
        )
        # Starting fom
        self.best_fom = 1e10
        ###########################################################################################
        # Pulses, Parameters object
        ###########################################################################################
        # Initialize the control object
        self.controls = Controls(
            optimization_dict["pulses"],
            optimization_dict["times"],
            optimization_dict["parameters"],
        )
        ###########################################################################################
        # Other useful variables
        ###########################################################################################
        self.dcrab_parameters_list = []
        self.dcrab_super_parameter_list = []
        self.fom_list = []
        self.iteration_number_list = []

    def _get_response_for_client(self) -> dict:
        """Return useful information for the client interface"""
        is_record = False
        fom = self.fom_dict["FoM"]
        if fom < self.best_fom:
            self.best_fom = fom
            is_record = True
        status_code = self.fom_dict.setdefault("status_code", 0)
        response_dict = {
            "is_record": is_record,
            "FoM": fom,
            "iteration_number": self.iteration_number,
            "status_code": status_code
        }
        # Load the current parameters
        if status_code == 0:
            self.fom_list.append(fom)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict

    def run(self) -> None:
        """Main loop of the dCRAB method"""
        for super_it in range(1, self.max_num_si + 1):
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
        # Add the best parameters and dcrab super_parameters of the current super-iteration
        self.dcrab_parameters_list.append(self.xx)
        self.dcrab_super_parameter_list.append(
            self.controls.get_random_super_parameter()
        )

    def _dsm_build(self, max_iteration_number: int) -> None:
        """Build the direct search method and run it"""
        start_simplex = simplex_creation(
            self.controls.get_mean_value(), self.controls.get_sigma_variation()
        )
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(
            self._routine_call,
            x0,
            initial_simplex=start_simplex,
            max_iterations_number=max_iteration_number,
        )
        # Update the results
        [fom, self.xx, self.terminate_reason] = [
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
            "Figure of merit": self.best_fom,
            "total number of function evaluations": self.iteration_number,
            "dcrab_freq_list": self.dcrab_super_parameter_list,
            "dcrab_para_list": self.dcrab_parameters_list,
            "terminate_reason": self.terminate_reason
        }
        return final_dict

    def get_best_controls(self):
        """ Return the best pulses_list, time_grids_list, and parameters_list found so far"""
        return self.controls.get_controls_lists(self.controls.get_mean_value())
