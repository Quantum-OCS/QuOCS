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
from quocslib.utils.dynamicimport import dynamic_import


class DCrabAlgorithm(Optimizer):
    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the dCRAB algorithm. All the arguments in the constructor are passed to the
        Optimizer class except the optimization dictionary where the dCRAB settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["dsm_settings"]["stopping_criteria"]
        direct_search_method_settings = optimization_dict["dsm_settings"]["general_settings"]
        # TODO Use dynamic import here to define the inner free gradient method
        # The callback function is called once in a while in the inner direct search method to check
        #  if the optimization is still running
        if "dsm_name" in direct_search_method_settings:
            print("dsm_name is used direct search methods. This option is deprecated. Use \n"
                  "dsm_algorithm_module: quocslib.freegradients.NelderMead\n"
                  "dsm_algorithm_class: NelderMead")
            self.dsm_obj = NelderMead(direct_search_method_settings,
                                      stopping_criteria,
                                      callback=self.is_optimization_running)
        else:
            dsm_attribute = dynamic_import(module_name=direct_search_method_settings["dsm_algorithm_module"],
                                           class_name=direct_search_method_settings["dsm_algorithm_class"])
            self.dsm_obj = dsm_attribute(direct_search_method_settings,
                                         stopping_criteria,
                                         callback=self.is_optimization_running)
        self.terminate_reason = ""
        ###########################################################################################
        # Optimal algorithm variables
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Max number of SI
        self.max_num_si = int(alg_parameters["super_iteration_number"])
        # TODO change evaluation number for the first and second super iteration
        # Max number of iterations at SI1
        self.max_num_function_ev = int(alg_parameters["maximum_function_evaluations_number"])
        # Max number of iterations from SI2
        self.max_num_function_ev2 = int(alg_parameters["maximum_function_evaluations_number"])
        # Starting FoM
        self.best_FoM = 1e10
        ###########################################################################################
        # Pulses, Parameters object
        ###########################################################################################
        # Initialize the control object
        self.controls = Controls(optimization_dict["pulses"],
                                 optimization_dict["times"],
                                 optimization_dict["parameters"])
        # Initialize the optimized control vector
        self.best_xx = self.controls.get_mean_value()
        ###########################################################################################
        # Other useful variables
        ###########################################################################################
        self.super_it: int = 1
        self.dcrab_parameters_list = []
        self.dcrab_super_parameter_list = []
        self.FoM_list = []
        self.iteration_number_list = []

    def _get_response_for_client(self) -> dict:
        """Return useful information for the client interface"""
        is_record = False
        FoM = self.FoM_dict["FoM"]
        if FoM < self.best_FoM:
            message = ("New record achieved. Previous FoM: {FoM}, new best FoM: {best_FoM}".format(FoM=self.best_FoM,
                                                                                               best_FoM=FoM))
            self.comm_obj.print_logger(message=message, level=20)
            self.best_FoM = FoM
            self.best_xx = self.xx.copy()
            is_record = True
        status_code = self.FoM_dict.setdefault("status_code", 0)
        response_dict = {"is_record": is_record,
                         "FoM": FoM,
                         "iteration_number": self.iteration_number,
                         "status_code": status_code}
        # Load the current parameters
        if status_code == 0:
            self.FoM_list.append(FoM)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict

    def run(self) -> None:
        """Main loop of the dCRAB method"""
        for super_it in range(1, self.max_num_si + 1):
            self.super_it = super_it
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
        self.controls.update_base_controls(self.best_xx)
        # Add the best parameters and dcrab super_parameters of the current super-iteration
        self.dcrab_parameters_list.append(self.best_xx)
        self.dcrab_super_parameter_list.append(self.controls.get_random_super_parameter())

    def _dsm_build(self, max_iteration_number: int) -> None:
        """Build the direct search method and run it"""
        # TR: what if I don't run NM and do not have a simnplex?!?!?
        start_simplex = simplex_creation(self.controls.get_mean_value(), self.controls.get_sigma_variation())
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Initialize the best xx vector for this SI
        self.best_xx = self.controls.get_mean_value().copy()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._routine_call, x0, initial_simplex=start_simplex,
                                        max_iterations_number=max_iteration_number,
                                        sigma_v=self.controls.get_sigma_variation())
        # Update the results
        [FoM, xx, self.terminate_reason, NfunevalsUsed] = [result_l["F_min_val"],
                                                           result_l["X_opti_vec"],
                                                           result_l["terminate_reason"],
                                                           result_l["NfunevalsUsed"]]
        message = ("SI {super_it} finished - Number of evaluations: {NfunevalsUsed}, "
                   "Termination Reason: {termination_reason}, "
                   "Best FoM: {best_FoM}\n".format(super_it=self.super_it,
                                                         NfunevalsUsed=NfunevalsUsed,
                                                         termination_reason=self.terminate_reason,
                                                         best_FoM=self.best_FoM))

        self.comm_obj.print_logger(message=message, level=20)

    def _get_controls(self, xx: np.array) -> dict:
        """Get the controls dictionary from the optimized control parameters"""
        [pulses, timegrids, parameters] = self.controls.get_controls_lists(xx)
        controls_dict = {"pulses": pulses,
                         "parameters": parameters,
                         "timegrids": timegrids}
        return controls_dict

    def _get_final_results(self) -> dict:
        """Return a dictionary with final results to put into a dictionary"""
        final_dict = {"Figure of merit": self.best_FoM,
                      "total number of function evaluations": self.iteration_number,
                      "dcrab_freq_list": self.dcrab_super_parameter_list,
                      "dcrab_para_list": self.dcrab_parameters_list,
                      "terminate_reason": self.terminate_reason}
        return final_dict

    def get_best_controls(self) -> list:
        """Return the best pulses_list, time_grids_list, and parameters_list found so far"""
        return self.controls.get_controls_lists(self.controls.get_mean_value())
