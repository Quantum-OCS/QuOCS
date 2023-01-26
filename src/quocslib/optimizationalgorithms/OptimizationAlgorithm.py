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
from abc import abstractmethod
import numpy as np
import signal

from quocslib.Controls import Controls
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib import __VERSION__ as QUOCSLIB_VERSION
from datetime import datetime
import threading

from quocslib.gradientfreemethods.DirectSearchMethod import DirectSearchMethod

INITIAL_FOM: float = 10**10


class OptimizationAlgorithm:
    init_status: bool
    xx: np.array
    alg_iteration_number: int
    iteration_number: int
    FoM_dict: dict
    dsm_obj: DirectSearchMethod
    controls: Controls

    def __init__(self, communication_obj: AllInOneCommunication = None, optimization_dict: dict = None):
        """
        The constructor of the OptimizationAlgorithm class. All the algorithms has to inherit it. It provides all the basic
        modules an optimizer should have. All the arguments are passed to the communication object. Find all the info
        in that class.

        :param dict communication_obj: Object fo the communication class
        """
        self.comm_obj = communication_obj
        self.optimization_dict = optimization_dict
        # get the names of the controls
        self.controls_names = self._get_controls_names()
        # set the dictionary in the communication onject
        self.comm_obj.set_controls_names(self.controls_names)
        # Print optimization dictionary into a file
        self.comm_obj.print_optimization_dictionary(self.optimization_dict)
        # Get the algorithm settings
        alg_parameters = self.optimization_dict["algorithm_settings"]
        # Initialize the total iteration number, i.e. the total function evaluations of the algorithm
        self.iteration_number = 0
        # Update status
        self.init_status = True
        # Random number generator
        self.rng = None
        # define message for global stopping criterion
        self.higher_order_terminate_reason = ""
        # Max number of total function evaluations
        self.max_eval_total = int(alg_parameters.setdefault("max_eval_total", 10**10))
        # Max total time of evaluation
        self.total_time_lim = alg_parameters.setdefault("total_time_lim", 10**10)
        # Goal FoM
        self.FoM_goal = alg_parameters.setdefault("FoM_goal", None)
        # initialize the optimizaiton start time
        self.optimization_start_time = None
        # Maximization or minimization
        # optimization_direction
        self.optimization_direction = alg_parameters.setdefault("optimization_direction", "minimization")
        if self.optimization_direction == "minimization":
            self.optimization_factor = -1.0
        elif self.optimization_direction == "maximization":
            self.optimization_factor = 1.0
        else:
            message = "You can choose between maximization/minimization " \
                      "only, but {0} is provided".format(self.optimization_direction)
            self.comm_obj.print_logger(message=message, level=40)
            raise TypeError
        self.best_FoM = self.FoM_maximum = self.optimization_factor * (-1.0) * INITIAL_FOM
        message = "The optimization direction is {0}".format(self.optimization_direction)
        self.comm_obj.print_logger(message=message, level=20)
        # listener for ctrl + c cancellation
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.handle_user_cancellation)
        # signal.signal(signal.SIGINT, self.handle_user_cancellation)

        if "algorithm_settings" in self.optimization_dict:
            temp_dict = self.optimization_dict["algorithm_settings"]
            if "dsm_settings" in temp_dict:
                temp_dict = temp_dict["dsm_settings"]
                if "stopping_criteria" in temp_dict:
                    temp_dict = temp_dict["stopping_criteria"]
                    if "frtol" in temp_dict:
                        self.comm_obj.print_logger('The "frtol" stopping criterion has been replaced by "fatol". '
                                                   'For more information pleas have a look at the documentation '
                                                   '(https://github.com/Quantum-OCS/QuOCS/blob/develop/Documentation/'
                                                   'Settings_in_Optimization_Dict.md)', 30)

    def handle_user_cancellation(self, sig, frame):
        self.higher_order_terminate_reason = "User stopped the optimization"
        self.dsm_obj.sc_obj.is_converged = True
        self.stop_optimization()

    def begin(self) -> None:
        """Initialize the communication with the client"""
        # Open the log with the QuOCS version number
        self.comm_obj.print_logger("QuOCS version number: {0}".format(QUOCSLIB_VERSION))
        # Send starting message to the interface
        self.comm_obj.send_message("start")
        # Assign new job number to the client
        self.comm_obj.assign_job()
        # Send the new data for the communication to the client
        self.comm_obj.send_communication_data()
        # Notify it to the client
        self.comm_obj.update_init_msg_server(upd_name=self.comm_obj.client_job_name)
        # set start time of optimization
        self.optimization_start_time = datetime.now()

    def _routine_call(self, optimized_control_parameters: np.array, iterations: int) -> float:
        """
        General routine for any control algorithm. It has to be given as the argument of the inner free gradient control
        methods

        :param: np.array optimized_control_parameters: The vector with all the optimized control parameters
        :param: int iterations: Iteration number of the inner free gradient method
        :return: float: Return the figure of merit to the inner free gradient method
        """

        # check if total maximum number of evals has been reached
        if self.iteration_number >= self.max_eval_total:
            self.higher_order_terminate_reason = "Maximum number of total function evaluations reached"
            self.dsm_obj.sc_obj.is_converged = True
            self.stop_optimization()

        # check if FoM_goal has been reached
        if self.FoM_goal is not None:
            if self.optimization_direction == "maximization":
                if self.best_FoM >= self.FoM_goal:
                    self.higher_order_terminate_reason = "Goal FoM reached"
                    self.dsm_obj.sc_obj.is_converged = True
                    self.stop_optimization()
            else:
                if self.best_FoM <= self.FoM_goal:
                    self.higher_order_terminate_reason = "Goal FoM reached"
                    self.dsm_obj.sc_obj.is_converged = True
                    self.stop_optimization()

        # check total optimization time limit
        if self.total_time_lim < 10**10:
            curr_time = datetime.now()
            time_passed = (curr_time - self.optimization_start_time).total_seconds() / 60.0

            if time_passed >= self.total_time_lim:
                self.higher_order_terminate_reason = "Maximum optimization runtime reached"
                self.dsm_obj.sc_obj.is_converged = True
                self.stop_optimization()

        # self.optimization_start_time = datetime.now()
        # Check if the optimization is still running
        is_running = self.comm_obj.get_user_running()
        if not is_running:
            return self.FoM_maximum
        # Update parameter array and iteration number
        self.xx, self.alg_iteration_number = optimized_control_parameters, iterations
        # Update iteration number
        self.iteration_number += 1
        # General workflow
        # Check interface update
        self.comm_obj.check_msg_client()
        # Send the controls to the interface
        self.comm_obj.send_controls(self._get_controls(optimized_control_parameters))
        # Update the notification file for the interface
        self.comm_obj.update_msg_server()
        # The interface reads the controls, calculates the FoM and updates its notification
        # Check for interface response
        self.comm_obj.check_msg_client()
        # Check if the optimization is still running
        is_running = self.comm_obj.get_user_running()
        if not is_running:
            return self.FoM_maximum
        # Get the figure of merit and update it to the main algorithm
        self.FoM_dict = self.comm_obj.get_data()["FoM_values"]
        # Send the response for the interface
        self.comm_obj.send_FoM_response(self._get_response_for_client())
        # Update the notification file for the interface
        self.comm_obj.update_msg_server()
        # The interface reads the FoM response and updates its notification file

        # add FoM to FoM track for stopping criteria
        try:
            self.dsm_obj.sc_obj.add_to_FoM_track(self.FoM_dict["FoM"])
        except Exception as ex:
            ### example for exception logging
            # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            # message = template.format(type(ex).__name__, ex.args)

            ### Ignore this if we are using GRAPE
            if type(self).__name__ == "GRAPEAlgorithm":
                pass
            # elif type(ex).__name__ == "AttributeError":
            #     message = "FoM track for stopping criteria could not be updated because " \
            #               "there is no stopping criteria object!"
            #     self.comm_obj.print_logger(message, level=30)
            else:
                message = "FoM track for stopping criteria could not be updated!"
                self.comm_obj.print_logger(message, level=30)

        # check the advanced stopping criteria
        try:
            self.dsm_obj.sc_obj.check_advanced_stopping_criteria()
        except Exception as ex:
            ### example for exception logging
            # template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            # message = template.format(type(ex).__name__, ex.args)

            ### Ignore this if we are using GRAPE
            if type(self).__name__ == "GRAPEAlgorithm":
                pass
            # elif type(ex).__name__ == "AttributeError":
            #     message = "Advanced stopping criteria could not be checked because " \
            #               "there is no stopping criteria object!"
            #     self.comm_obj.print_logger(message, level=30)
            else:
                message = "Advanced stopping criteria could not be checked!"
                self.comm_obj.print_logger(message, level=30)

        # Return the figure of merit, i.e. a real number, to the optimal based algorithm
        return -1.0 * self.optimization_factor * self.FoM_dict["FoM"]

    def get_is_record(self, FoM: float) -> bool:
        """Check if the figure of merit provided is a new record

        :param: FoM  : figure of merit provided by the user
        """
        # Minimization
        if self.optimization_factor < 0.0:
            if FoM < self.best_FoM:
                return True
        else:
            # Maximization
            if FoM > self.best_FoM:
                return True
        return False

    @abstractmethod
    def run(self) -> None:
        """Run the optimization algorithm"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_response_for_client(self) -> dict:
        """Return a dictionary with useful info for the client interface. At least the dictionary
        has to provide "is_record": bool and "FoM": float"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_controls(self, optimized_control_parameters: np.array) -> [list, list, list]:
        """Given the optimized control parameters, the control object in the optimal algorithm builds the
        pulses, time grids, and parameters"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    # TODO Make the below function no more public !
    @abstractmethod
    def _get_final_results(self) -> dict:
        """The optimal algorithm gives back a dictionary with useful results"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    def is_optimization_running(self) -> bool:
        """Function to check if the optimization is still running"""
        return self.comm_obj.get_user_running()

    def stop_optimization(self) -> None:
        """Function to stop the optimization (inner direct search algorithm)"""
        self.comm_obj.set_is_running_state(value=False)

    def get_best_controls(self) -> dict:
        """Return the best pulses_list, time_grids_list, and parameters_list found so far"""
        pulses_list, time_grids_list, parameters_list = self.controls.get_controls_lists(self.controls.get_mean_value())
        return {"pulses": pulses_list, "parameters": parameters_list, "timegrids": time_grids_list}

    def get_best_bare_controls(self) -> dict:
        """Return the best bare pulses_list (without guess and scaling) found so far"""
        bare_pulses_list = self.controls.get_bare_controls_lists(self.controls.get_mean_value())
        return {"bare_pulses": bare_pulses_list}

    def _get_controls_names(self):
        """Return the names of the pulses, parameters and times"""
        pulse_name_list = []
        for pulse_dict in self.optimization_dict["pulses"]:
            pulse_name_list.append(pulse_dict["pulse_name"])
        parameter_name_list = []
        for param_dict in self.optimization_dict["parameters"]:
            parameter_name_list.append(param_dict["parameter_name"])
        time_name_list = []
        for time_dict in self.optimization_dict["times"]:
            time_name_list.append(time_dict["time_name"])
        return {"pulse_names": pulse_name_list, "parameter_names": parameter_name_list, "time_names": time_name_list}

    def end(self) -> None:
        """Finalize the transmission with  the client"""
        # Check client update
        self.comm_obj.check_msg_client()
        # End communication
        # ToDo: fix this... this is kind of a workaround for now
        end_comm_message_dict = self._get_final_results()
        self.comm_obj.dump_obj.dump_dict("optimized_parameters", end_comm_message_dict)
        if self.higher_order_terminate_reason != "":
            end_comm_message_dict["Termination Reason"] = self.higher_order_terminate_reason
        self.comm_obj.end_communication(end_comm_message_dict)
        # Update server message
        self.comm_obj.update_msg_server()
