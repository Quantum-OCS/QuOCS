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

import os
import time
from abc import abstractmethod
from shutil import rmtree
import logging
import sys

from ClientInterface.logic.utilities.writejsonfile import writejsonfile
from ClientInterface.logic.utilities.readjson import readjson
from ClientInterface.logic.ReceiveFom import ReceiveFom
from ClientInterface.logic.DumpResults import DumpResults


class Communication:
    """Parent Communication class. All the important methods for the communication between Client Interface and
    optimization code are defined here.
     """
    ####################################################################################################################
    # Workflow numbers and job names
    ####################################################################################################################
    # Main numbers for the communication between the Client and the Server
    server_job_name = ""
    server_number = 0

    client_job_name = ""
    client_number = 0

    # Exchange folder defined by the Client
    exchange_folder = ""

    # Initial communication folder
    initial_communication_folder = ""

    # Main communication folder
    main_communication_folder = ""

    # Paths for communication defined by the Server
    path_opti_data = ""
    opti_data_file = ""
    path_communication_folder = ""

    is_connected = False
    is_debug = False

    def __init__(self, handle_exit_obj, client_job_name, fom_dict, message_signal, plot_signal,
                 parameters_update_signal):
        """
        @param handle_exit_obj: Object to handle the exit
        @param client_job_name: Job name chose by the Client Interface
        @param fom_dict: Dictionary for Figure of Merit calculation
        @param message_signal: signal for Client Interface GUI
        @param plot_signal: signal for plotting in the Client Interface GUI
        @param parameters_update_signal: signal for updating the parameter in the GUI
        """
        ##################
        # Signals
        ##################
        # Define message signal
        self.message_signal = message_signal
        # Define plot signal
        self.plot_signal = plot_signal
        # Define parameters signal
        self.parameters_update_signal = parameters_update_signal
        # Pre job name
        pre_job_name = client_job_name
        # Datetime for 1-1 association
        date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        # Client job name to send to the Server
        self.client_job_name = date_time + "_" + pre_job_name
        # Shared folder for communication
        self.shared_folder = os.path.join(os.getcwd(), "shared_folder", self.client_job_name)
        # Create folder
        os.mkdir(self.shared_folder)
        # Results directory path
        self.res_path = os.path.join(os.getcwd(), "dCRAB", self.client_job_name)

        ########################################
        # Check folder of the control algorithm
        ########################################
        # TODO Change name here
        opti_method = "dCRAB"
        if not os.path.isdir(os.path.join(os.getcwd(), opti_method)):
            os.makedirs(os.path.join(os.getcwd(), opti_method))
        # Create the folder for logging and results
        os.makedirs(self.res_path)

        ########################################
        # Logging object
        ########################################
        # Create logging object
        self.logger = self._create_logger()

        # FoM object
        self.rf_obj = ReceiveFom(fom_dict)

        # Dumping data object
        self.dr_obj = DumpResults(self.client_job_name)

        # Initialization of the handle exit object
        self.he_obj = handle_exit_obj

    def _create_logger(self):
        """Logger creation for console, log file, and debug log file"""

        log_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        date_format = '%m/%d/%Y %I:%M:%S'
        print_format = '%(levelname)-8s %(name)-12s %(message)s'
        log_filename = os.path.join(self.res_path, "logging.log")
        log_debug_filename = os.path.join(self.res_path, "logging_debug.log")

        logger = logging.getLogger("oc_logger")
        # Default level for logger
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(print_format))

        # Log file handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # Debug
        if self.is_debug:
            # Log debug file handler
            debug_file_handler = logging.FileHandler(log_debug_filename)
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.setFormatter(logging.Formatter(log_format, date_format))

            logger.addHandler(debug_file_handler)

        return logger

    def get_FoM(self, controls_dict):
        """

        @param controls_dict: dictionary with pulses, parameters, and timegrids, i.e. {"pulses": [[]], "paras": [],
        "timegrids": [[]]}
        @return: {"FoM": float, "ErrorCode": int, "ErrorMessage": str}
        """
        # Get the figure of merit dictionary
        fom_values_dict = self.rf_obj.get_FoM(**controls_dict)
        # Update the
        if "ErrorCode" in fom_values_dict:
            self.set_client_number(fom_values_dict["ErrorCode"], error=True)

        return fom_values_dict

    def check_communication(self, communication_dict):
        """
        Update the client, server numbers and check the communiation status
        @param communication_dict:
        @return:
        """
        return self.he_obj.check_communication(communication_dict)

    def set_client_number(self, client_number, error=False):
        """
        Update the client number. It is updated only in the case no other error happened before
        @param client_number:
        @param error:
        @return:
        """
        if error:
            self.client_number = client_number
        else:
            if self.client_number >= 0:
                self.client_number = client_number

    def get_communication_numbers(self):
        """ Return communication numbers as a dictionary"""
        communication_dict = {"server_number": self.server_number, "client_number": self.client_number}
        return communication_dict

    def _create_communication_dict(self):
        """ Return the communication data for the Client-Server transmission """
        communication_dict = {"server_job_name": self.server_job_name, "server_number": self.server_number,
                              "client_job_name": self.client_job_name, "client_number": self.client_number,
                              "path_opti_data": self.path_opti_data}

        return communication_dict

    def client_message(self, message, is_log=False):
        """
        Send the message to the client interface or in the console/log file
        @param message:
        @param is_log:
        @return:
        """
        # TODO overwrite this module for AllInOne communication
        # TODO Create a log class and get in this class the logger of that and used it
        if self.message_signal is not None:
            self.message_signal.emit(message)
        if is_log:
            self.logger.info(message)

    def controls_signal_update(self, pulses, paras, timegrids):
        """
        Send a signal to the Client Interface GUI with the controls
        @param pulses: List of list [[]]
        @param paras: List []
        @param timegrids: List of list [[]]
        @return:
        """
        # TODO send all the controls to the gui
        if self.parameters_update_signal is not None:
            self.parameters_update_signal.emit(paras)

    def print_data(self, opti_data):
        """
        Print the data in the result file
        @param opti_data: dictionary with controls
        @return:
        """
        # TODO Save optimal pulses and so on
        self.dr_obj.print_data(opti_data)

    def fom_plot_data(self, iteration, fom):
        """
        Send a signal for the Figure of merit plot
        @param iteration: positive integer number
        @param fom: float number
        @return:
        """
        if self.plot_signal is not None:
            self.plot_signal.emit(iteration, fom)

    def send_config(self, cfg_dict):
        """
        Send the config file for initial communication
        @param cfg_dict: Dictionary containing the optimization settings
        @return:
        """
        # Initialize the client number and the filename
        self.client_number = 0
        initial_filename = self.client_job_name + ".json"
        # Send only the opti_dict
        opti_dict = {"opti_dict": cfg_dict["opti_dict"]}
        # Set the client job name
        opti_dict["opti_dict"]["comm_dict"]["client_job_name"] = self.client_job_name
        # Create the config file and put it in the communication folder
        filepath_shared = os.path.join(self.shared_folder, initial_filename)
        self.logger.debug("Send the cfg file to the initial communication folder: {0}"
                          .format(filepath_shared))
        writejsonfile(filepath_shared, opti_dict)
        # Move json file from shared to the communication folder
        filepath_communication = os.path.join(self.initial_communication_folder, initial_filename)
        self.move_file(filepath_shared, filepath_communication)

    def initialize_comm_data(self):
        """ ### """
        # Move file to shared folder
        opti_data_file = self.client_job_name + ".json"
        #
        filepath_communication = self.main_communication_folder + "/" + opti_data_file
        filepath_shared = os.path.join(self.shared_folder, opti_data_file)
        self.get_file(filepath_communication, filepath_shared)
        # Read the initial communication file from the Server
        err_stat, opti_data = readjson(filepath_shared)
        # Set the communication data
        self.logger.debug("Set the communication Data {0}".format(opti_data["communication"]))
        self._set_comm_data(opti_data)
        # Update Client number
        self.client_number = 1

    def _set_comm_data(self, opti_data):
        """
        Get the data for the communication from the Server after the Server had accepted the Client job
        Returns
        -------

        """
        comm_data = opti_data["communication"]
        self.server_job_name = comm_data["server_job_name"]
        self.server_number = comm_data["server_number"]
        # Full path for the opti data communication file
        self.path_opti_data = comm_data["path_opti_data"]
        # Opti data file
        self.opti_data_file = comm_data["opti_data_file"]
        # Path of the communication folder to be used during transmission
        self.path_communication_folder = comm_data["path_communication_folder"]

    def update_msg_client(self):
        """
        Update client message
        Returns
        -------

        """
        upd_file = "upd_client.txt"
        filepath_shared = os.path.join(self.shared_folder, upd_file)
        self.logger.debug("Update the client message in {0}".format(filepath_shared))
        open(filepath_shared, "w").close()
        # Move file to communication folder
        filepath_communication = self.path_communication_folder + "/" + upd_file
        self.move_file(filepath_shared, filepath_communication, confirm=False)

    def check_initial_msg_server(self, prefix=""):
        """

        Parameters
        ----------
        prefix

        Returns
        -------

        """
        upd_file = self.main_communication_folder + "/" + prefix + "upd_server.txt"
        while not self.check_file(upd_file):
            # Do nothing, just wait
            time.sleep(0.01)
        self.remove_file(upd_file)

    def check_msg_server(self):
        """

        Returns
        -------

        """
        upd_file = self.path_communication_folder + "/" + "upd_server.txt"
        while not self.check_file(upd_file):
            # Do nothing, just wait
            time.sleep(0.01)
        self.remove_file(upd_file)

    def get_data(self):
        """
        This module is used to get any data from server program, written in the exchange json file.
        It is an internal function.
        Returns
        -------

        """
        filepath_communication = self.path_opti_data
        filepath_shared = os.path.join(self.shared_folder, self.opti_data_file)
        # Move file to shared folder
        self.get_file(filepath_communication, filepath_shared)
        err_stat, opti_data = readjson(filepath_shared)
        return opti_data

    def send_data(self, fom_dict=None):
        """

        Parameters
        ----------
        fom_dict

        Returns
        -------

        """
        # Get communication dict
        comm_dict = self._create_communication_dict()
        # Put together the dictionary
        data_dict = {"communication": comm_dict, "fom_values": fom_dict}
        # Create the json file in the shared folder
        filepath_shared = os.path.join(self.shared_folder, self.opti_data_file)
        writejsonfile(filepath_shared, data_dict)
        filepath_communication = self.path_opti_data
        self.move_file(filepath_shared, filepath_communication)

    def print_results(self, it_no, fom):
        self.logger.info("Iteration number: {0} , FoM: {1}".format(it_no, fom))

    def end(self, opti_data_dict):
        """Print final results and remove shared folder"""
        # Print final results data
        if "final_results" in opti_data_dict:
            results_dict = opti_data_dict["final_results"]
            for el in results_dict:
                self.logger.info("{0} : {1}".format(el, results_dict[el]))
        # Remove the shared folder
        rmtree(self.shared_folder)
        self.logger.debug("Shared folder removed")

    @abstractmethod
    def move_file(self, origin, destination, confirm=True):
        """ """
        raise NotImplementedError("Must override in the custom communication")

    @abstractmethod
    def get_file(self, origin, destination):
        """ """
        raise NotImplementedError("Must override in the custom communication")

    @abstractmethod
    def remove_file(self, file):
        """ """
        raise NotImplementedError("Must override in the custom communication")

    @abstractmethod
    def check_file(self, file):
        """ """
        raise NotImplementedError("Must override in the custom communication")
