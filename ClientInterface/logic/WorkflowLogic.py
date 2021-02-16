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

from ClientInterface.logic.communication.LocalCommunication import LocalCommunication as LocComm
from ClientInterface.logic.communication.RemoteCommunication import RemoteCommunication as RemComm


class Workflow:
    """
    Workflow is the main class to handle the process of the optimization. It takes responsible for communication,
    figure of merit evaluation, dumping and plotting
    """
    # Current json data file
    opti_data = None

    def __init__(self, handle_exit_obj, client_job_name, comm_fom_dict, comm_signals_list):
        """
        Constructor for the workflow process
        @param handle_exit_obj: Object to handle the exit
        @param client_job_name: client job name
        @param comm_fom_dict: Dictionary with Communication and FoM evaluation settings
        @param comm_signals_list: List with Gui signals
        """
        # Communication dictionary
        comm_dict = comm_fom_dict["comm_dict"]
        # Steps in the workflow
        self.steps = [self.wait_data, self.send_data, self.wait_response]
        # Communication object
        comm_type_dict = {"LocalCommunication": LocComm, "RemoteCommunication": RemComm}
        for el in comm_type_dict:
            if comm_dict["type"] == el:
                self.comm_obj = comm_type_dict[el](handle_exit_obj, client_job_name, comm_fom_dict, comm_signals_list)
        # TODO Return an error message if comm_obj does not exist

    def initialize_communication(self, cfg_dict):
        """
        Initialize the communication with the Optimization code by sending the config dictionary and getting back the
        information regarding the transmission
        @param cfg_dict:
        @return: True or False
        """
        # Initialize the communication with the Optimization Code
        # Message
        self.comm_obj.client_message("Initialize the communication")
        # Send the config file to the communication folder
        self.comm_obj.send_config(cfg_dict)
        # Set CI number to 0
        self.comm_obj.set_client_number(0)
        # Wait for the update from the Server
        self.comm_obj.check_initial_msg_server(prefix=self.comm_obj.client_job_name)
        # Get the new data for the communication from OC
        self.comm_obj.initialize_comm_data()
        # Update CI message
        self.comm_obj.update_msg_client()
        # Set CI number to 0
        self.comm_obj.set_client_number(1)
        # Check if the communication was successful
        return self.comm_obj.check_communication(self.comm_obj.get_communication_numbers())

    # Main modules for the communications
    def wait_data(self):
        """
        Wait set of controls or the end of process notification
        @return: True or False
        """
        # Message in the CI window
        self.comm_obj.client_message("Waiting data from Server")
        # Check for OC update
        self.comm_obj.check_msg_server()
        # Read the opti data json file
        self.opti_data = self.comm_obj.get_data()
        communication_data = self.opti_data["communication"]
        # Return optimization status
        return self.comm_obj.check_communication(communication_data)

    def send_data(self):
        """
        FoM calculation and send the result to the Optimization code
        @return: True or False
        """
        # Send the controls to the gui
        self.comm_obj.controls_signal_update(**self.opti_data["controls"])
        # Calculate FoM
        fom_values_dict = self.comm_obj.get_FoM(self.opti_data["controls"])
        # Message
        self.comm_obj.client_message("Send data to the Server")
        # Set CI number
        self.comm_obj.set_client_number(2)
        # Put the fom data in the communication folder
        self.comm_obj.send_data(fom_values_dict)
        # Update client message
        self.comm_obj.update_msg_client()
        # Check if any error occurs in the client side during FoM calculation
        return self.comm_obj.check_communication(self.comm_obj.get_communication_numbers())

    def wait_response(self):
        """
        Get the fom response, i.e. it is an improvement or not by the Optimization Code
        @return: True or False
        """
        # Message
        self.comm_obj.client_message("Waiting response from Server")
        # check if json file was updated
        self.comm_obj.check_msg_server()
        # Read the opti data json file
        opti_data = self.comm_obj.get_data()
        # Dump Data
        self.comm_obj.print_data(opti_data)
        # Message
        self.comm_obj.client_message("Iteration_number: {0} , FoM: {1}".
                                     format(opti_data["response"]["iteration_number"], opti_data["response"]["FoM"]),
                                     is_log=True)
        # Update FoM data
        self.comm_obj.fom_plot_data(opti_data["response"]["iteration_number"], opti_data["response"]["FoM"])
        # Set client number
        self.comm_obj.set_client_number(1)
        # Give back the opti_data json file to the server with the communication data
        self.comm_obj.send_data()
        # Update client message
        self.comm_obj.update_msg_client()
        # Check if any error occurs in the Server side during the FoM validation
        return self.comm_obj.check_communication(self.comm_obj.get_communication_numbers())

    def end_communication(self):
        """ End the communication"""
        self.comm_obj.end(self.opti_data)
