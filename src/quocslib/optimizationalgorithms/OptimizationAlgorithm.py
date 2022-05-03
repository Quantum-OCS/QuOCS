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
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib import __VERSION__ as QUOCSLIB_VERSION

INITIAL_FOM: float = 10**10


class OptimizationAlgorithm:
    init_status: bool = False
    FoM_maximum: float = 1e10
    xx: np.array
    alg_iteration_number: int
    iteration_number: int
    FoM_dict: dict

    def __init__(self, communication_obj: AllInOneCommunication = None, optimization_dict: dict = None):
        """
        The constructor of the OptimizationAlgorithm class. All the algorithms has to inherit it. It provides all the basic
        modules an optimizer should have. All the arguments are passed to the communication object. Find all the info
        in that class.

        :param dict communication_obj: Object fo the communication class
        """
        self.comm_obj = communication_obj
        # Print optimization dictionary into a file
        self.comm_obj.print_optimization_dictionary(optimization_dict)
        # Initialize the total iteration number, i.e. the total function evaluations of the algorithm
        self.iteration_number = 0
        # Update status
        self.init_status = True
        # Random number generator
        self.rng = None
        # Maximization or minimization
        # optimization_direction
        optimization_direction = optimization_dict.setdefault("optimization_direction", "minimization")
        if optimization_direction == "minimization":
            self.optimization_factor = -1.0
        elif optimization_direction == "maximization":
            self.optimization_factor = 1.0
        else:
            message = "You can choose between maximization/minimization " \
                      "only, but {0} is provided".format(optimization_direction)
            self.comm_obj.print_logger(message=message, level=40)
            raise TypeError
        self.best_FoM = self.optimization_factor * (-1.0) * INITIAL_FOM
        message = "The optimization direction is {0}".format(optimization_direction)
        self.comm_obj.print_logger(message=message, level=20)

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

    def _routine_call(self, optimized_control_parameters: np.array, iterations: int) -> float:
        """
        General routine for any control algorithm. It has to be given as the argument of the inner free gradient control
        methods

        :param: np.array optimized_control_parameters: The vector with all the optimized control parameters
        :param: int iterations: Iteration number of the inner free gradient method
        :return: float: Return the figure of merit to the inner free gradient method
        """
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
        # The interface reads the FoM response and update its notification file
        #
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

    def end(self) -> None:
        """Finalize the transmission with  the client"""
        # Check client update
        self.comm_obj.check_msg_client()
        # End communication
        self.comm_obj.end_communication(self._get_final_results())
        # Update server message
        self.comm_obj.update_msg_server()
