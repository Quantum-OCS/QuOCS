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

import os
import time

from quocstools.AbstractFom import AbstractFom
from quocstools.DummyDump import DummyDump
from quocslib.handleexit.AbstractHandleExit import AbstractHandleExit
from quocslib.tools.logger import create_logger


class AllInOneCommunication:

    def __init__(self, interface_job_name: str = "OptimizationTest", fom_obj: AbstractFom = None,
                 handle_exit_obj: AbstractHandleExit = None, dump_attribute: callable = DummyDump,
                 comm_signals_list: [list, list, list] = None):
        """
        In case the user chooses to run the optimization in his device, this class is used by the Optimizer.
        The objects to dump the results, calculate the figure of merit, and the logger are created here.

        :param str interface_job_name: Name decided by the Client. It is change in the constructor adding the current
        time to ensure univocity
        :param AbstractFom fom_obj: object for the figure of merit evaluation. Have a look to the abstract class for
        for more info
        :param AbstractHandleExit handle_exit_obj: Collect any error during the optimization and check when the
        communication is finished to communicate with the client interface
        :param [list, list, list] comm_signals_list: List containing the signals to the gui
        """
        # Communication signals
        if comm_signals_list is None:
            self.message_signal, self.fom_plot_signal, self.controls_update_signal = None, None, None
        else:
            self.message_signal, self.fom_plot_signal, self.controls_update_signal = comm_signals_list
        # Pre job name
        pre_job_name = interface_job_name
        # Datetime for 1-1 association
        date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        # Client job name to send to the Server
        self.client_job_name = date_time + "_" + pre_job_name
        ###
        # Logging, Results, Figure of merit evaluation ...
        ###
        # Optimization folder
        optimization_folder = "QuOCS_Results"
        self.results_path = os.path.join(os.getcwd(), optimization_folder, self.client_job_name)
        if not os.path.isdir(os.path.join(os.getcwd(), optimization_folder)):
            os.makedirs(os.path.join(os.getcwd(), optimization_folder))
        # Create the folder for logging and results
        os.makedirs(self.results_path)
        # Create logging object
        self.logger = create_logger(self.results_path)
        # Figure of merit object
        self.fom_obj = fom_obj
        # TODO Thinks whether it is a good idea dumping the results
        # Dumping data object
        self.dump_obj = dump_attribute(self.results_path)
        # Handle exit object
        self.he_obj = handle_exit_obj
        # Initialize the control dictionary
        self.controls_dict = {}

    def get_user_running(self) -> bool:
        """ Check if the user stopped the optimization """
        return self.he_obj.is_user_running

    def send_controls(self, controls_dict: dict) -> None:
        """
        Set the controls for FoM calculation and notify the gui

        :param dict controls_dict:
        :return:
        """
        self.controls_dict = controls_dict
        if self.controls_update_signal is not None:
            self.controls_update_signal.emit(
                controls_dict["pulses"], controls_dict["timegrids"], controls_dict["parameters"])

    def get_data(self) -> dict:
        """
        Calculate the figure of merit and return a dictionary with all the arguments

        :return dict: {"fom_values": {"FoM": float, ...}}
        """
        fom_dict = self.fom_obj.get_FoM(**self.controls_dict)
        return {"fom_values": fom_dict}

    def send_fom_response(self, response_for_client: dict) -> None:
        """
        Emit signal to the Client Interface and dump the results in case any

        :param dict response_for_client: It is a dictionary defined in the optimal algorithm
        :return:
        """
        iteration_number, fom = response_for_client["iteration_number"], response_for_client["FoM"]
        self.logger.info("Iteration number: {0}, FoM: {1}".format(iteration_number, fom))
        self.dump_obj.dump_controls(**self.controls_dict, **response_for_client)
        if self.fom_plot_signal is not None:
            self.fom_plot_signal.emit(iteration_number, fom)

    def end_communication(self, results_dict: dict) -> None:
        """
        Report the final results

        :param dict results_dict: It is a dictionary defined in the optimal algorithm with all the data to display at
        the end of the optimization process
        :return:
        """
        # Print final results data
        for el in results_dict:
            self.logger.info("{0} : {1}".format(el, results_dict[el]))

    def assign_job(self) -> None:
        """Do nothing"""
        pass

    def send_communication_data(self) -> None:
        """Do nothing"""
        pass

    def update_init_msg_server(self, upd_name=None) -> None:
        """Do nothing"""
        pass

    def check_msg_client(self) -> None:
        """Do nothing"""
        pass

    def update_msg_server(self) -> None:
        """Do nothing"""
        pass
