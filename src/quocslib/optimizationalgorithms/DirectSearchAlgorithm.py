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
import numpy as np
from scipy.stats import norm


class DirectSearchAlgorithm(OptimizationAlgorithm):
    """
    This is an implementation of a direct search algorithm. It only performs a direct search based on the specified
    search method in the dsm_settings of the optimization_dict. The algorithm is stopped when the stopping criteria
    are met. The algorithm is implemented in the run function. The results are returned in the _get_final_results
    function.
    """

    initStatus = 0
    terminate_reason = "-1"

    def __init__(self, optimization_dict: dict = None, communication_obj=None, **kwargs):
        """
        Constructor for the direct search algorithm. The optimization dictionary is passed to the constructor of the
        OptimizationAlgorithm class. The direct search method is initialized here.
        :param optimization_dict: dictionary with the optimization settings
        :param communication_obj: communication object to communicate with the client
        """
        super().__init__(communication_obj=communication_obj, optimization_dict=optimization_dict)
        ###########################################################################################
        # Direct Search method
        ###########################################################################################
        stopping_criteria = optimization_dict["algorithm_settings"]["dsm_settings"]["stopping_criteria"]
        # put global time limit into stopping_criteria, so we don't have to pass it through functions
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

        ################################################################################################
        # Extract some Settings for the Search #########################################################
        ################################################################################################
        alg_parameters = optimization_dict["algorithm_settings"]
        drift_compensations_parameters = optimization_dict["algorithm_settings"].setdefault("compensate_drift", {})
        # Define sigma for re-evaluation steps
        self.best_sigma = 0.0
        # Update the FoM for drift
        self.compensate_drift_after_minutes = drift_compensations_parameters.setdefault("compensate_after_minutes", 0.0)
        self.compensate_drift_num_average = drift_compensations_parameters.setdefault("num_average", 1)
        # Re-evaluation steps option
        if "re_evaluation" in alg_parameters:
            re_evaluation_parameters = alg_parameters["re_evaluation"]
            if "re_evaluation_steps" in re_evaluation_parameters:
                self.re_evaluation_steps = np.asarray(re_evaluation_parameters["re_evaluation_steps"], dtype=float)
            else:
                self.re_evaluation_steps = np.asarray([0.3, 0.5, 0.501], dtype=float)
                message = "Steps not found. The default will be used in the optimization: {0}".format(
                    self.re_evaluation_steps)
                self.comm_obj.print_logger(message, level=30)
            # Define the FoM test and sigma test arrays, with the maximum number of steps
            # FoM test and sigma test are arrays containing the FoM and sigma at every re-evaluation step
            self.FoM_test = np.zeros(self.re_evaluation_steps.shape[0] + 1, dtype=float)
            self.sigma_test = np.zeros_like(self.FoM_test)
        else:
            self.re_evaluation_steps = None

        ###########################################################################################
        # Optimal algorithm variables ###########################################################
        ###########################################################################################
        # Starting FoM
        # self.best_FoM = 1e10  # defined in parent class
        ###########################################################################################
        # Pulses, Parameters object ###########################################################
        ###########################################################################################
        self.controls = Controls(optimization_dict["pulses"], optimization_dict["times"],
                                 optimization_dict["parameters"])
        self.FoM_list: list = []
        self.iteration_number_list: list = []

    def _inner_routine_call(self, optimized_control_parameters: np.array, iterations: int,
                            drift_comp_new_val=None) -> float:
        """
        Wrapper Function for the _routine_call of the OptimizationAlgorithm class.
        Used to perform drift compensation and re-evaluation steps.
        :param optimized_control_parameters: optimized control parameters
        :param iterations: number of iterations
        :param drift_comp_new_val: new FoM value in case a drift compensation happened
        :return float: FoM
        """
        if drift_comp_new_val is not None:
            mu_1 = drift_comp_new_val
            self.best_FoM = mu_1
            self.is_record = True
            message = "New record due to drift compensation. New best FoM: {0}".format(self.FoM_factor*mu_1)
            self.comm_obj.print_logger(message, level=20)
            self.best_xx = self.xx.copy()
            self.comm_obj.update_controls(is_record=True,
                                          FoM=self.best_FoM,
                                          # super_it=self.super_it,
                                          iteration_number=self.iteration_number)
        else:
            # set the is_record to False unless we have drift_comp call... then we want to update the current best FoM
            self.is_record = False
            # Initialize step number to 0
            self.step_number = 0
            FoM = self._routine_call(optimized_control_parameters, iterations)

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
                    self.FoM_test[ii + 1] = self._routine_call(optimized_control_parameters, iterations)
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
                    message = "New record achieved. New best FoM: {0}, std: {1}".format(self.FoM_factor*mu_1, sigma_1)
                    self.comm_obj.print_logger(message, level=20)
                    self.best_xx = self.xx.copy()
                    self.comm_obj.update_controls(is_record=True,
                                                  FoM=self.best_FoM,
                                                  sigma=self.best_sigma,
                                                  # super_it=self.super_it,
                                                  iteration_number=self.iteration_number)

        # Return the figure of merit to be minimized by the updating algorithm
        return mu_1

    def _get_average_FoM_std(self, mu_sum: float = None, sigma_sum: float = None) -> np.array:
        """
        Calculate the average figure of merit and standard deviation based on the passed arrays

        :param mu_sum: Sum of the figure of merit values
        :param sigma_sum: Sum of the standard deviation values
        :return float, float: Average figure of merit and standard deviation
        """
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

        :param mu_1: = <x1>
        :param sigma_1: = std(x1)
        :param mu_2:  = <x2>
        :param sigma_2: = std(x2)
        :return float: probability P(x1>=x2)
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

    def _get_response_for_client(self):
        """
        Return useful information for the client interface and print message in the log

        :return dict: Dictionary containing useful information for the client interface
        """
        FoM = self.FoM_dict["FoM"]
        status_code = self.FoM_dict.setdefault("status_code", 0)
        # If re-evaluation steps is not used check for current best figure of merit
        if self.re_evaluation_steps is None:
            if self.get_is_record(FoM):
                message = "New record achieved. Previous FoM: {FoM}, new best FoM : {best_FoM}".format(
                    FoM=self.FoM_factor*self.best_FoM, best_FoM=self.FoM_factor*FoM)
                self.comm_obj.print_logger(message=message, level=20)
                self.best_FoM = FoM
                self.best_xx = self.xx.copy()
                self.is_record = True
        response_dict = {
            "is_record": self.is_record,
            "FoM": FoM,
            "iteration_number": self.iteration_number,
            "status_code": status_code
        }

        if self.re_evaluation_steps is not None:
            FoM, std = self._get_average_FoM_std()
            message = "Function evaluation number: {func_eval}, ".format(func_eval=self.iteration_number)
            message += "Re-eval. number: {0}, FoM: {1}, std: {2}".format(self.step_number, self.FoM_factor*FoM, std)
            self.comm_obj.print_logger(message, level=20)

        if status_code == 0:
            self.FoM_list.append(self.FoM_factor*FoM)
            self.iteration_number_list.append(self.iteration_number)
        return response_dict

    def run(self):
        """
        Runs the optimization algorithm
        """
        # Direct search method
        self._dsm_build()

    def _dsm_build(self):
        """
        Runs the direct search
        """
        start_simplex = simplex_creation(self.controls.get_mean_value(), self.controls.get_sigma_variation())
        # Initial point for the Start Simplex
        x0 = self.controls.get_mean_value()
        # Initialize the best xx vector for this SI
        self.best_xx = self.controls.get_mean_value().copy()
        # Run the direct search algorithm
        result_l = self.dsm_obj.run_dsm(self._inner_routine_call,
                                        x0,
                                        initial_simplex=start_simplex,
                                        sigma_v=self.controls.get_sigma_variation(),
                                        drift_comp_minutes=self.compensate_drift_after_minutes,
                                        drift_comp_num_average=self.compensate_drift_num_average)
        # Update the results
        [FoM, xx,
         self.terminate_reason] = [result_l["F_min_val"], result_l["X_opti_vec"], result_l["terminate_reason"]]

    def _get_controls(self, xx):
        """
        Returns the controls dictionary for the given input vector
        :param xx:
        :return dict: Dictionary containing the controls
        """
        # pulses_list, time_grids_list, parameters_list
        [], [], parameters_list = self.controls.get_controls_lists(xx)

        controls_dict = {"pulses": [], "parameters": parameters_list, "timegrids": []}
        return controls_dict

    def _get_final_results(self):
        """
        Returns the final results of the optimization

        :return dict: Dictionary containing the final results
        """
        final_dict = {
            "Figure of merit": self.FoM_factor * self.best_FoM,
            "parameters": self.xx,
            "terminate_reason": self.terminate_reason
        }
        return final_dict
