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

from quocslib.optimizationalgorithms.OptimizationAlgorithm import OptimizationAlgorithm
from quocslib.Controls import Controls
from quocslib.gradientfreemethods.NelderMead import NelderMead
from quocslib.tools.linearalgebra import simplex_creation
from quocslib.tools.randomgenerator import RandomNumberGenerator
from quocslib.utils.dynamicimport import dynamic_import


class dCRABAlgorithm(OptimizationAlgorithm):
    def __init__(self, optimization_dict: dict = None, communication_obj=None):
        """
        This is the implementation of the dCRAB algorithm. All the arguments in the constructor are passed to the
        OptimizationAlgorithm class except the optimization dictionary where the dCRAB settings and the controls are defined.
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["algorithm_settings"]["dsm_settings"]["stopping_criteria"]
        # put global time limit into stopping_criteria so we don't have to pass it through functions
        optimization_dict["algorithm_settings"].setdefault("total_time_lim", 10**10)
        stopping_criteria.setdefault("total_time_lim", optimization_dict["algorithm_settings"]["total_time_lim"])
        optimization_dict["algorithm_settings"]["dsm_settings"]["stopping_criteria"].setdefault('k3', 3)
        direct_search_method_settings = optimization_dict["algorithm_settings"]["dsm_settings"]["general_settings"]
        # TODO: Use dynamic import here to define the inner free gradient method
        # The callback function is called once in a while in the inner direct search method to check
        #  if the optimization is still running

        dsm_attribute = dynamic_import(module_name=direct_search_method_settings.setdefault("dsm_algorithm_module", None),
                                       class_name=direct_search_method_settings.setdefault("dsm_algorithm_class", None),
                                       name=direct_search_method_settings.setdefault("dsm_algorithm_name", None),
                                       class_type='dsm_settings')

        self.dsm_obj = dsm_attribute(direct_search_method_settings,
                                     stopping_criteria,
                                     callback=self.is_optimization_running,
                                     stop_optimization_callback=self.stop_optimization)

        # self.dsm_obj = NelderMead(direct_search_method_settings,
        #                           stopping_criteria,
        #                           callback=self.is_optimization_running)
        self.terminate_reason = ""
        ###########################################################################################
        # Optimal algorithm variables
        ###########################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        # Max number of SI
        self.max_num_si = int(alg_parameters["super_iteration_number"])
        # TODO: old: Change evaluation number for the first and second super iteration... new: think of something
        #  else, e.g. adaption on how close one is to the desired FoM or so
        # Max number of iterations at SI1
        self.max_eval_total = int(alg_parameters["max_eval_total"])
        # Starting FoM and sigma
        self.best_FoM = 1e10
        self.best_sigma = 0.0
        # Update the drift Hamiltonian
        self.is_compensate_drift = alg_parameters.setdefault("is_compensated_drift", False)
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
            # Define the FoM test and sigma test arrays, with the maximum number of steps
            # FoM test and sigma test are arrays containing the FoM and sigma at every re-evaluation step
            self.FoM_test = np.zeros(self.re_evaluation_steps.shape[0] + 1, dtype=float)
            self.sigma_test = np.zeros_like(self.FoM_test)
        else:
            self.re_evaluation_steps = None
        # Seed for the random number generator
        if "random_number_generator" in alg_parameters:
            try:
                seed_number = alg_parameters["random_number_generator"]["seed_number"]
                self.rng = RandomNumberGenerator(seed_number=seed_number)
            except (TypeError, KeyError):
                default_seed_number = 2022
                message = "Seed number must be an integer value. Set {0} as a seed numer for this optimization".format(
                    default_seed_number)
                self.rng = RandomNumberGenerator(seed_number=default_seed_number)
                self.comm_obj.print_logger(message, level=30)
        # The fact a FoM is a record FoM is decided by the inner call
        self.is_record = False
        # Initialize the step number used during the FoM calculation
        self.step_number = 0
        ###########################################################################################
        # Pulses, Parameters object
        ###########################################################################################
        # Initialize the control object
        self.controls = Controls(optimization_dict["pulses"],
                                 optimization_dict["times"],
                                 optimization_dict["parameters"],
                                 rng=self.rng)
        ###########################################################################################
        # General Log message
        ###########################################################################################
        self.comm_obj.print_general_log = False
        ###########################################################################################
        # Other useful variables
        ###########################################################################################
        self.super_it: int = 0
        self.dcrab_parameters_list: list = []
        self.dcrab_super_parameter_list: list = []
        self.FoM_list: list = []
        self.iteration_number_list: list = []

    def _get_response_for_client(self) -> dict:
        """ Return useful information for the client interface and print message in the log """
        # Get the average FoM
        FoM, std = self._get_average_FoM_std()
        status_code = self.FoM_dict.setdefault("status_code", 0)
        # If re-evaluation steps is not used check for current best figure of merit
        if self.re_evaluation_steps is None:
            if self.get_is_record(FoM):
                message = "New record achieved. Previous FoM: {FoM}, new best FoM : {best_FoM}".format(
                    FoM=self.best_FoM, best_FoM=FoM)
                self.comm_obj.print_logger(message=message, level=20)
                self.best_FoM = FoM
                self.best_xx = self.xx.copy()
                self.is_record = True
        response_dict = {
            "is_record": self.is_record,
            "FoM": FoM,
            "iteration_number": self.iteration_number,
            "super_it": self.super_it,
            "status_code": status_code
        }
        ################################################################################################################
        # Print message in the log
        ################################################################################################################
        # Iterations
        message = ("Function evaluation number: {func_eval}, "
                   "SI: {super_it}, "
                   "Sub-iteration number: {sub_it}".format(func_eval=self.iteration_number,
                                                           super_it=self.super_it,
                                                           sub_it=self.alg_iteration_number))
        # Data
        if self.re_evaluation_steps is not None:
            message += ", Re-eval. number: {0}, FoM: {1}, std: {2}".format(self.step_number, FoM, std)
        else:
            message += ", FoM: {0}".format(FoM)
        self.comm_obj.print_logger(message, level=20)
        # Load the current figure of merit and iteration number in the summary list of dCRAB
        if status_code == 0:
            self.FoM_list.append(FoM)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict

    def run(self) -> None:
        """Main loop of the dCRAB method"""
        for super_it in range(1, self.max_num_si + 1):
            # Check if the optimization was stopped by the user
            if not self.is_optimization_running():
                return
            try:
                # reset the timeout of the dsm for each SI
                self.dsm_obj.sc_obj.reset_direct_search_start_time()
                message = "Direct search start time has been reset."
                self.comm_obj.print_logger(message, level=20)
            except:
                message = "Direct search start time could not be reset!"
                self.comm_obj.print_logger(message, level=30)
            # Set super iteration number
            self.super_it = super_it
            # Compensate the drift Hamiltonian
            if self.is_compensate_drift and super_it >= 2:
                self._update_FoM()
            # Initialize the random super_parameters
            self.controls.select_basis()
            # Direct search method
            self._dsm_build(self.max_eval_total)
            # Update the base current pulses
            self._update_base_pulses()

    def _update_FoM(self) -> None:
        """Update the value of the best FoM using the current best controls"""
        previous_best_FoM = self.best_FoM
        # Get the current best control optimization vector
        x0 = self.controls.get_mean_value()
        # Evaluate the FoM with the standard routine call and set the FoM as the current record
        iteration, self.is_record, self.step_number = 0, True, 0
        self.best_FoM = self._routine_call(x0, iteration)
        # Info message
        message = f"Previous best FoM: {previous_best_FoM} , Current best FoM after drift compensation: {self.best_FoM}"
        self.comm_obj.print_logger(message=message, level=20)
        # At this point is not necessary to set again is_record to False since is newly re-define at the beginning of
        # the _inner_routine_call function
        # TODO: Thinks if makes sense to update the sigma best value here

    def _update_base_pulses(self) -> None:
        """Update the base dCRAB pulse with the best controls found so far"""
        self.controls.update_base_controls(self.best_xx)
        # Add the best parameters and dcrab super_parameters of the current super-iteration
        self.dcrab_parameters_list.append(self.best_xx)
        self.dcrab_super_parameter_list.append(self.controls.get_random_super_parameter())

    def _dsm_build(self, max_iteration_number: int) -> None:
        """Build the direct search method and run it"""
        start_simplex = simplex_creation(self.controls.get_mean_value(),
                                         self.controls.get_sigma_variation(),
                                         rng=self.rng)
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Initialize the best xx vector for this SI
        self.best_xx = self.controls.get_mean_value().copy()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._inner_routine_call,
                                        x0,
                                        initial_simplex=start_simplex,
                                        max_eval_total=max_iteration_number)
        # Update the results
        [FoM, xx, self.terminate_reason, NfunevalsUsed] = [result_l["F_min_val"],
                                                           result_l["X_opti_vec"],
                                                           result_l["terminate_reason"],
                                                           result_l["NfunevalsUsed"]]
        # Message at the end of the SI
        message = ("SI {super_it} finished - Number of evaluations: {NfunevalsUsed}, "
                   "Best FoM: {best_FoM}, Terminate reason: {reason}\n".format(super_it=self.super_it,
                                                   NfunevalsUsed=NfunevalsUsed,
                                                   termination_reason=self.terminate_reason,
                                                   best_FoM=self.best_FoM,
                                                   reason=self.terminate_reason))
        self.comm_obj.print_logger(message=message, level=20)

    def _inner_routine_call(self, optimized_control_parameters: np.array, iterations: int) -> float:
        """This is an inner method for function evaluation. It is useful when the user wants to evaluate the FoM
        with the same controls multiple times to take into accounts noise in the system"""
        self.is_record = False
        # Initialize step number to 0
        self.step_number = 0
        FoM = -1.0 * self.optimization_factor * self._routine_call(optimized_control_parameters, iterations)
        ################################################################################################################
        # Standard function evaluation - dCRAB without re-evaluation steps
        ################################################################################################################
        if self.re_evaluation_steps is None:
            mu_1 = FoM
        else:
            ############################################################################################################
            # Implement the re-evaluation step method
            ############################################################################################################
            # check mu-sig criterion by calculating probability of current pulses being new record
            # Re evaluation steps initialization e.g. [0.33, 0.5, 0.501, 0.51]
            re_evaluation_steps = self.re_evaluation_steps
            # First evaluation in whole optimization -> do not reevaluate
            if self.iteration_number == 1:
                re_evaluation_steps = np.array([0.5])
            # number of steps
            max_steps_number = re_evaluation_steps.shape[0]
            # Initialize to zero the FoM_test and the sigma_test arrays
            self.FoM_test = 0.0 * self.FoM_test
            self.sigma_test = 0.0 * self.sigma_test
            # Get the figure of merit from the client
            self.FoM_test[0] = FoM
            # TODO: Check if optimization_is_running is necessary here
            # Get the standard deviation
            self.sigma_test[0] = float(self.FoM_dict.setdefault("std", 1.0))
            # Increase step number after function evaluation
            self.step_number += 1
            # p level test better than current record
            for ii in range(max_steps_number):
                p_level = re_evaluation_steps[ii]
                mu_1, sigma_1 = self._get_average_FoM_std(mu_sum=np.sum(self.FoM_test) * 1.0,
                                                          sigma_sum=np.sum(self.sigma_test) * 1.0)
                mu_2, sigma_2 = self.best_FoM, self.best_sigma
                probability = self._probabnormx1betterx2(mu_1, sigma_1, mu_2, sigma_2)
                # If probability is lower than the probability in the list return the
                if probability < p_level:
                    return mu_1
                # else: go on with further re-evaluations
                self.FoM_test[ii + 1] = -1.0 * self.optimization_factor * self._routine_call(
                    optimized_control_parameters, iterations)
                self.sigma_test[ii + 1] = float(self.FoM_dict.setdefault("std", 1.0))
                # Increase step number after function evaluation
                self.step_number += 1

            # check if last threshold (re_evaluation_steps[-1]) is surpassed -> new record
            mu_1, sigma_1 = self._get_average_FoM_std(mu_sum=np.sum(self.FoM_test) * 1.0,
                                                      sigma_sum=np.sum(self.sigma_test) * 1.0)
            mu_2, sigma_2 = self.best_FoM, self.best_sigma
            probability = self._probabnormx1betterx2(mu_1, sigma_1, mu_2, sigma_2)
            # TODO: Check what best FoM means in this case
            if probability > re_evaluation_steps[-1]:
                # We have a new record
                self.best_sigma, self.best_FoM = sigma_1, mu_1
                self.is_record = True
                message = "New record achieved. New best FoM: {0}, std: {1}".format(mu_1, sigma_1)
                self.comm_obj.print_logger(message, level=20)
                self.best_xx = self.xx.copy()
                self.comm_obj.update_controls(is_record=True,
                                              FoM=self.best_FoM,
                                              sigma=self.best_sigma,
                                              super_it=self.super_it)

        # Return the figure of merit to be minimized by the updating algorithm
        return -1.0 * self.optimization_factor * mu_1

    def _get_average_FoM_std(self, mu_sum: float = None, sigma_sum: float = None) -> np.array:
        """Calculate the average figure of merit and sigma"""
        step_number = self.step_number
        # For the first evaluation and in case no re-evaluation step is needed return directly
        if step_number == 0:
            return self.FoM_dict["FoM"], self.FoM_dict.setdefault("std", 1.0)
        # Call from the response for client function. Calculate the average FoM based on all the previous FoM stored in
        # the FoM_test array and the current FoM in the FoM dict
        if mu_sum is None:
            curr_FoM, curr_std = self.FoM_dict["FoM"], self.FoM_dict.setdefault("std", 1.0)
            mu_sum = np.mean(self.FoM_test[:step_number]) * (step_number - 1) + curr_FoM
            sigma_sum = (np.mean(self.sigma_test[:step_number]) * (step_number - 1) + curr_std)
        # If it is called inside the _inner_routine_call()
        average_FoM, average_std = mu_sum / step_number, sigma_sum / step_number
        return average_FoM, average_std

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
        std_comb = np.sqrt(sigma_1**2 + sigma_2**2)
        if np.abs(std_comb) < 10**(-14):
            # Warning message
            message = ("Combined standard deviation std_comb = {0} < 10**(-14) . To avoid numerical instabilities "
                       "std_comb will be set equal to 1.0".format(std_comb))
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

        controls_dict = {"pulses": pulses, "parameters": parameters, "timegrids": timegrids}
        return controls_dict

    def _get_final_results(self) -> dict:
        """Return a dictionary with final results to put into a dictionary"""
        final_dict = {
            "Figure of merit": self.best_FoM,
            "Std": self.best_sigma,
            "total number of function evaluations": self.iteration_number,
            "dcrab_freq_list": self.dcrab_super_parameter_list,
            "dcrab_para_list": self.dcrab_parameters_list,
            "terminate_reason": self.terminate_reason
        }
        return final_dict

    def get_best_controls(self) -> list:
        """Return the best pulses_list, time_grids_list, and parameters_list found so far"""
        return self.controls.get_controls_lists(self.controls.get_mean_value())
