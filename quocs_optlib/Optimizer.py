# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from abc import abstractmethod
import numpy as np

from quocs_optlib.communication.AllInOneCommunication import AllInOneCommunication
from quocs_optlib.figureofmeritevaluation.AbstractFom import AbstractFom
from quocs_optlib.handleexit.AbstractHandleExit import AbstractHandleExit


class Optimizer:
    init_status: bool = False
    fom_maximum: float = 1e10
    xx: np.array
    alg_iteration_number: int
    iteration_number: int
    fom_dict: dict

    def __init__(self, interface_job_name: str = None, fom_obj: AbstractFom = None,
                 handle_exit_obj: AbstractHandleExit = None,
                 comm_signals_list: [list, list, list] = None):
        """
        The constructor of the Optimizer class. All the algorithms has to inherit it. It provides all the basic
        modules an optimizer should have. All the arguments are passed to the communication object. Find all the info
        in that class.
        :param str interface_job_name: Job name from the interface
        :param AbstractFom fom_obj: Figure of merit object.
        :param AbstractHandleExit handle_exit_obj: Handle exit object
        :param [list, list, list] comm_signals_list: Signals for the communication with the main interface
        """
        # Create the communication object
        self.comm_obj = AllInOneCommunication(interface_job_name, fom_obj, handle_exit_obj, comm_signals_list)
        # Initialize the total iteration number
        self.iteration_number = 0
        # Update status
        self.init_status = True

    def begin(self) -> None:
        """ Initialize the communication with the client"""
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
        :param np.array optimized_control_parameters: The vector with all the optimized control parameters
        :param int iterations: Iteration number of the inner free gradient method
        :return: float: Return the figure of merit to the inner free gradient method
        """
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
        if not self.comm_obj.is_running:
            return self.fom_maximum
        # Get the figure of merit and update it to the main algorithm
        self.fom_dict = self.comm_obj.get_data()["fom_values"]
        # Send the response for the interface
        self.comm_obj.send_fom_response(self._get_response_for_client())
        # Update the notification file for the interface
        self.comm_obj.update_msg_server()
        # The interface reads the FoM response and update its notification file
        #
        # Return the figure of merit, i.e. a real number, to the optimal based algorithm
        return self.fom_dict["FoM"]

    @abstractmethod
    def run(self) -> None:
        """ Run the optimization algorithm """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_response_for_client(self) -> dict:
        """ Return a dictionary with useful info for the client interface. At least the dictionary
        has to provide "is_record": bool and "FoM": float """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_controls(self, optimized_control_parameters: np.array) -> [list, list, list]:
        """ Given the optimized control parameters, the control object in the optimal algorithm builds the
         the pulses, time grids, and parameters"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_final_results(self) -> dict:
        """ The optimal algorithm gives back a dictionary with useful results"""
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    def is_optimization_running(self) -> bool:
        """ Module to stop the inner direct search algorithm, or to handle a possible recovery or pause mode """
        return self.comm_obj.is_running

    def end(self) -> None:
        """ Finalize the transmission with  the client """
        # Check client update
        self.comm_obj.check_msg_client()
        # End communication
        self.comm_obj.end_communication(self._get_final_results())
        # Update server message
        self.comm_obj.update_msg_server()
