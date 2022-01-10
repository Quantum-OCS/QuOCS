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
from scipy.stats import norm

from quocslib.Optimizer import Optimizer
from quocslib.Controls import Controls
from quocslib.freegradientmethods.NelderMead import NelderMead
from quocslib.tools.linearalgebra import simplex_creation


class DCrabNoisyAlgorithm(Optimizer):
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
        # TODO: Use dynamic import here to define the inner free gradient method
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
        # TODO: Change evaluation number for the first and second super iteration
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
        # Update the drift Hamiltonian
        self.is_compensate_drift = alg_parameters.setdefault("is_compensated_drift", True)
        # Re-evaluation steps option
        if "re_evaluation" in alg_parameters:
            re_evaluation_parameters = alg_parameters["re_evaluation"]
            if "re_evaluation_steps" in re_evaluation_parameters:
                self.re_evaluation_steps = np.asarray(re_evaluation_parameters["re_evaluation_steps"], dtype=float)
            else:
                self.re_evaluation_steps = np.asarray([0.3, 0.5, 0.6], dtype=float)
                message = "Steps not found. The default will be used in the optimization: {0}".format(
                    self.re_evaluation_steps)
                self.comm_obj.print_logger(message, level=30)
        else:
            self.re_evaluation_steps = np.array([0.5])
        # The fact a FoM is a record Fom is decided by the inner call
        self.is_record = False
        # Define the average figure fom merit and sigma
        self.average_fom = self.best_fom
        self.best_sigma = 0.0
        # Define the fom test and sigma test arrays, with the maximum number of steps
        self.fom_test = np.zeros((self.re_evaluation_steps.shape[0] + 1,), dtype=float)
        self.sigma_test = np.zeros_like(self.fom_test)
        # Initialize the step number used during the fom calculation
        self.step_number = 0
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
        """Return useful information for the client interface """
        # Get the average fom 
        fom, std = self._get_average_fom_std()
        status_code = self.fom_dict.setdefault("status_code", 0)
        response_dict = {
            "is_record": self.is_record,
            "FoM": fom,
            "iteration_number": self.iteration_number,
            "status_code": status_code
        }
        message = "Step number: {0}, fom: {1}, std: {2} ".\
            format(self.step_number, fom, std)
        self.comm_obj.print_logger(message, level=20)
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
            # Compensate the drift Hamiltonian
            if self.is_compensate_drift and super_it >= 2:
                self._update_fom()
            # Initialize the random super_parameters
            self.controls.select_basis()
            # Direct search method
            if super_it == 1:
                self._dsm_build(self.max_num_function_ev)
            else:
                self._dsm_build(self.max_num_function_ev2)
            # Update the base current pulses
            self._update_base_pulses()

    def _update_fom(self) -> None:
        """ Update the value of the best fom using the current best controls """
        # Info message
        message = "Update the best fom using the best controls found so far"
        self.comm_obj.print_logger(message=message, level=10)
        # Get the controls value
        x0 = self.controls.get_mean_value()
        # Evaluate the fom with the standard routine call
        iteration = 0
        self.is_record = True
        self.best_fom = self._routine_call(x0, iteration)
        # At this point is not necessary to set again is_record to False since is newly re-define at the beginning of
        # the _inner_routine_call function
        # TODO: Thinks if makes sense to update the sigma best value here

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
            self._inner_routine_call,
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

    def _inner_routine_call(self, optimized_control_parameters: np.array, iterations: int) -> float:
        """ This is an inner method for function evaluation. It is useful when the user wants to evaluate the FoM
        with the same controls multiple times to take into accounts noise in the system """
        ################################################################################################################
        # Implement the re-evaluation step method
        ################################################################################################################
        self.is_record = False
        # check mu-sig criterion by calculating probability of current pulses being new record
        # Re evaluation steps initialization e.g. [0.33, 0.5, 0.501, 0.51]
        re_evaluation_steps = self.re_evaluation_steps
        # First evaluation in whole optimization -> do not reevaluate
        if self.iteration_number == 0:
            re_evaluation_steps = np.array([0.5])
        # Initialize step number to 0
        self.step_number = 0
        # number of steps
        max_steps_number = re_evaluation_steps.shape[0]
        # fom_test = np.zeros(max_steps_number + 1)
        # sigma_test = np.zeros_like(fom_test)
        # Initialize to zero the fom_test and the sigma_test arrays
        self.fom_test = 0.0 * self.fom_test
        self.sigma_test = 0.0 * self.sigma_test
        # Get the figure of merit from the client
        self.fom_test[0] = self._routine_call(optimized_control_parameters, iterations)
        # TODO: Check if optimization_is_running is necessary here
        # Get the standard deviation
        self.sigma_test[0] = float(self.fom_dict.setdefault("std", 1.0))
        # Increase step number after function evaluation
        self.step_number += 1
        # p level test better than current record
        for ii in range(max_steps_number):
            p_level = re_evaluation_steps[ii]
            mu_1, sigma_1 = self._get_average_fom_std(mu_sum=np.sum(self.fom_test) * 1.0,
                                                      sigma_sum=np.sum(self.sigma_test) * 1.0)
            # mu_1 = np.mean(fom_test) * 1.0
            # sigma_1 = np.mean(sigma_test) / np.sqrt(ii + 1.0)
            mu_2, sigma_2 = self.best_fom, self.best_sigma
            probability = self._probabnormx1betterx2(mu_1, sigma_1, mu_2, sigma_2)
            # If probability is lower than the probability in the list return the
            if probability < p_level:
                return mu_1
            # else: go on with further re-evaluations
            self.fom_test[ii + 1] = self._routine_call(optimized_control_parameters, iterations)
            self.sigma_test[ii + 1] = float(self.fom_dict.setdefault("std", 1.0))
            # Increase step number after function evaluation
            self.step_number += 1

        # Update the step number to the last one
        # self.step_number += 1
        # check if last threshold (re_evaluation_steps[-1]) is surpassed -> new record
        mu_1, sigma_1 = self._get_average_fom_std(mu_sum=np.sum(self.fom_test) * 1.0,
                                                  sigma_sum=np.sum(self.sigma_test) * 1.0)
        # mu_1 = np.mean(fom_test) * 1.0
        # sigma_1 = np.mean(sigma_test) / np.sqrt(ii + 1.0)
        mu_2, sigma_2 = self.best_fom, self.best_sigma
        probability = self._probabnormx1betterx2(mu_1, sigma_1, mu_2, sigma_2)
        # TODO: Check what best fom means in this case
        if probability > re_evaluation_steps[-1]:  # or self.fund_updatebestfom:  # we've got a new record!!
            self.best_sigma = sigma_1
            self.best_fom = mu_1
            self.is_record = True
            message = "Found a record. fom: {0}, std: {1}".format(mu_1, sigma_1)
            self.comm_obj.print_logger(message, level=20)

        return mu_1

    def _get_average_fom_std(self, mu_sum: float = None,
                         sigma_sum: float = None) -> np.array:
        """ Calculate the average figure of merit and sigma """
        step_number = self.step_number
        # Call from the response for client function. Calculate the average fom based on all the previous fom stored in
        # the fom_test array and the current fom in the fom dict
        if mu_sum is None:
            curr_fom, curr_std = self.fom_dict["FoM"], self.fom_dict.setdefault("std", 1.0)
            mu_sum = np.mean(self.fom_test[:step_number + 1]) * step_number + curr_fom
            sigma_sum = np.mean(self.sigma_test[:step_number + 1]) * step_number + curr_std
        # If it is called
        average_fom, average_std = mu_sum / (step_number + 1), sigma_sum / (step_number + 1)
        return average_fom, average_std

    def _probabnormx1betterx2(self, mu_1: float, sigma_1: float, mu_2: float, sigma_2: float):
        """
        Calculates probability for normal distributed random variable x1 being greater or equal than x2
        x1 usually refers to the test pulse and
        x2 to the current record that is tried to be outperformed
        ----------------------------------------------------------------------------------------------------------------
        :param mu_1: = <x1>
        :param sigma_1: = std(x1)
        :param mu_2:  = <x2>
        :param sigma_2: = std(x2)
        :return: probability P(x1>=x2)
        """
        # Start by defining a new random variable z = x1 - x2
        # if mu_z > 0 the probability is > 0.5 , else: <0.5
        mu_z = mu_2 - mu_1
        std_comb = np.sqrt(sigma_1 ** 2 + sigma_2 ** 2)
        if np.abs(std_comb) < 10 ** (-14):
            # Warning message
            message = "Combined standard deviation std_comb = {0} < 10**(-14) . To avoid numerical instabilities " \
                      "std_comb will be set equal to 1.0".format(std_comb)
            self.comm_obj.print_logger(message, level=30)
            # Set std_com to 1.0
            std_comb = 1.0
        zz = mu_z / std_comb
        # Calculate the probability with the cumulative density function
        probability = norm.cdf(zz)
        return probability

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
            "Std": self.best_sigma,
            "total number of function evaluations": self.iteration_number,
            "dcrab_freq_list": self.dcrab_super_parameter_list,
            "dcrab_para_list": self.dcrab_parameters_list,
            "terminate_reason": self.terminate_reason
        }
        return final_dict

    def get_best_controls(self) -> list:
        """ Return the best pulses_list, time_grids_list, and parameters_list found so far"""
        return self.controls.get_controls_lists(self.controls.get_mean_value())
