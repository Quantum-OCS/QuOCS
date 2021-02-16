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

from OptimizationCode.communication.ServerLocalCommunication import ServerLocalCommunication as SLC
from OptimizationCode.communication.AllInOneCommunication import AllInOneCommunication as AIOC


class Optimizer:
    """

    """
    init_status = False
    xx = None
    alg_iteration_number = 0
    iteration_number = 0
    further_args = {}
    fom_dict = {}

    def __init__(self, opti_dict=None, handle_exit_obj=None, comm_fom_dict=None, comm_signals_list=None):
        """

        Parameters
        ----------
        options_dict
        """

        if handle_exit_obj is None:
            comm_dict=opti_dict["comm_dict"]
            self.comm_obj = SLC(comm_dict)
        else:
            self.comm_obj = AIOC(handle_exit_obj, opti_dict["comm_dict"]["client_job_name"], comm_fom_dict["fom_dict"],
                                 comm_signals_list)
        # Update status
        self.init_status = True

    def begin(self):
        """
        Initialize the communication with the client
        Returns
        -------

        """
        # Assign new job number to the client
        self.comm_obj.assign_job()
        # Send the new data for the communication to the client
        self.comm_obj.send_communication_data()
        # Notify it to the client
        self.comm_obj.update_init_msg_server(upd_name=self.comm_obj.client_job_name)

    def _routine_call(self, xx, iterations, rmode, other):
        """
        General routine for any control algorithm. It has to be called in the optimal class.
        It can also be used for multiple evaluation, like in the CMAES algorithm.
        Parameters
        ----------
        xx (nd.array) parameters vector
        iterations (int) iteration number
        rmode (string) keyword for the figure of merit evaluation
        other (dict) any

        Returns
        -------

        """
        # Update parameter array and iteration number
        self.xx, self.alg_iteration_number = xx, iterations
        # Update iteration number
        self.iteration_number += 1
        # General workflow
        # Check client update
        self.comm_obj.check_msg_client()
        # Create the json file with the parameter
        self.comm_obj.send_controls(self._get_controls(xx))
        # Update the notification file for the Client
        self.comm_obj.update_msg_server()
        # The client reads the parameters, calculate the FoM and update its notification file
        # Check for client response
        self.comm_obj.check_msg_client()
        # Get the figure of merit and update it to the main algorithm
        self.fom_dict = self.comm_obj.get_data()["fom_values"]
        # Create the Figure of merit response
        self.comm_obj.send_fom_response(self._get_response_for_client())
        # Update the notification file for the Client
        self.comm_obj.update_msg_server()
        # The client reads the FoM response and update its notification file

        # Return the figure of merit, i.e. a real number, to the direct search algorithm
        return self.fom_dict["FoM"]

    @abstractmethod
    def run(self):
        """

        Returns
        -------

        """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_response_for_client(self):
        """

        Returns
        -------

        """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_controls(self, xx):
        """

        Returns
        -------

        """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    @abstractmethod
    def _get_final_results(self):
        """

        Returns
        -------

        """
        raise NotImplementedError("Must override method in the Optimal Algorithm class")

    def _check_client_status(self):
        """
        Module to stop the inner direct search algorithm, or to handle a possible recovery or pause mode
        Returns
        -------

        """
        pass

    def end(self):
        """
        Finalize the transmission with  the client
        Returns
        -------

        """
        # Check client update
        self.comm_obj.check_msg_client()
        # End communication
        self.comm_obj.end_communication(self._get_final_results())
        # Update server message
        self.comm_obj.update_msg_server()
