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

import numpy as np

from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.utils.DummyDump import DummyDump
from quocslib.handleexit.AbstractHandleExit import AbstractHandleExit
from quocslib.tools.logger import create_logger
from quocslib.utils.inputoutput import writejsonfile
from quocslib import __VERSION__ as quocslib_version


class AllInOneCommunication:
    def __init__(self,
                 interface_job_name: str = "OptimizationTest",
                 FoM_obj: AbstractFoM = None,
                 handle_exit_obj: AbstractHandleExit = None,
                 dump_attribute: callable = DummyDump,
                 comm_signals_list: [list, list, list] = None):
        """
        In case the user chooses to run the optimization in his device, this class is used by the OptimizationAlgorithm.
        The objects to dump the results, calculate the figure of merit, and the logger are created here. 

        :param str interface_job_name: Name decided by the Client. It is change in the constructor adding the current
        time to ensure univocity
        :param AbstractFoM FoM_obj: object for the figure of merit evaluation. Have a look to the abstract class for
        more info
        :param AbstractHandleExit handle_exit_obj: Collect any error during the optimization and check when the
        communication is finished to communicate with the client interface
        :param [list, list, list] comm_signals_list: List containing the signals to the gui
        """
        # Communication signals
        if comm_signals_list is None:
            self.message_signal, self.FoM_plot_signal, self.controls_update_signal = (None, None, None)
        else:
            (self.message_signal, self.FoM_plot_signal, self.controls_update_signal) = comm_signals_list
        # Pre job name
        pre_job_name = interface_job_name
        # Datetime for 1-1 association
        self.date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        # Client job name to send to the Server
        self.client_job_name = self.date_time + "_" + pre_job_name
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
        # Write the current quocs lib version in the file
        with open(os.path.join(self.results_path, "quocs_version.txt"), "w") as version_file:
            version_file.write("QuOCS library version: {0}".format(quocslib_version))
        # Create logging object
        self.logger = create_logger(self.results_path, self.date_time)
        # Print function evaluation and figure of merit
        self.print_general_log = True
        # Figure of merit object
        self.FoM_obj = FoM_obj
        # TODO Thinks whether it is a good idea dumping the results
        # Dumping data object
        self.dump_obj = dump_attribute(self.results_path, self.date_time)
        # Handle exit object
        self.he_obj = handle_exit_obj
        # Initialize the control dictionary
        self.controls_dict = {}

    def print_logger(self, message: str = "", level: int = 20):
        """Print a message in the log"""
        if level <= 10:
            self.logger.debug(message)
        elif 10 < level <= 20:
            self.logger.info(message)
        elif 20 < level <= 30:
            self.logger.warning(message)
        else:
            self.logger.error(message)

    def send_message(self, message):
        """Send a message to the interface"""
        if self.message_signal is not None:
            self.message_signal.emit(message)

    def print_optimization_dictionary(self, optimization_dictionary: dict) -> None:
        """Print optimization dictionary into a file"""
        writejsonfile(os.path.join(self.results_path, self.date_time + "_" + "optimization_dictionary.json"),
                      optimization_dictionary)

    def get_user_running(self) -> bool:
        """Check if the user stopped the optimization"""
        self.logger.debug("User running: {0}".format(self.he_obj.is_user_running))
        return self.he_obj.is_user_running

    def set_is_running_state(self, value:bool) -> None:
        """Set if the optimization is running"""
        self.he_obj.is_user_running = value
        self.logger.info("Setting is_running state: {0}".format(self.he_obj.is_user_running))

    def send_controls(self, controls_dict: dict) -> None:
        """
        Set the controls for FoM calculation and notify the gui

        :param dict controls_dict:
        :return:
        """
        self.controls_dict = controls_dict
        if self.controls_update_signal is not None:
            self.controls_update_signal.emit(controls_dict["pulses"], controls_dict["timegrids"],
                                             controls_dict["parameters"])

    def get_data(self) -> dict:
        """
        Calculate the figure of merit and return a dictionary with all the arguments

        :return dict: {"FoM_values": {"FoM": float, ...}}
        """
        self.logger.debug("User running: {0}".format(self.he_obj.is_user_running))
        FoM_dict = self.FoM_obj.get_FoM(**self.controls_dict)
        # set the status of the FoM to 0 if it does not exist already
        status_code = FoM_dict.setdefault("status_code", 0)
        # if the user passes a different status code than zero, stop the optimization
        if status_code != 0:
            self.he_obj.is_user_running = False
        return {"FoM_values": FoM_dict}

    def send_FoM_response(self, response_for_client: dict) -> None:
        """
        Emit signal to the Client Interface and dump the results in case any

        :param dict response_for_client: It is a dictionary defined in the optimal algorithm
        :return:
        """
        iteration_number, FoM = (response_for_client["iteration_number"], response_for_client["FoM"])
        status_code = response_for_client.setdefault("status_code", 0)
        # Check for interrupting signals
        if status_code != 0:
            self.logger.info("The optimization was interrupted with status code: {0}"
                             " at iteration {1}".format(status_code, iteration_number))
            # Set the user running to False in order to not continue with the next iteration
            self.he_obj.is_user_running = False
            return
        self._print_general_log(iteration_number, FoM)
        self.update_controls(**response_for_client)
        if self.FoM_plot_signal is not None:
            self.FoM_plot_signal.emit(iteration_number, FoM)

    def _print_general_log(self, iteration_number: int, FoM: float):
        """Print the general log at each function evaluation"""
        if self.print_general_log:
            self.logger.info("Function evaluation number: {0}, FoM: {1}".format(iteration_number, FoM))

    def update_controls(self, **response_for_client) -> None:
        """External call to update the controls"""
        if response_for_client is None:
            response_for_client = {}
        self.dump_obj.dump_controls(**self.controls_dict, **response_for_client)

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
