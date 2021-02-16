import time
import os
import logging
import sys

from ClientInterface.logic.ReceiveFom import ReceiveFom as RF
from ClientInterface.logic.DumpResults import DumpResults as DR


class AllInOneCommunication:
    is_debug = False

    def __init__(self, handle_exit_obj, client_job_name, fom_dict, comm_signal_list):
        """

        Parameters
        ----------
        comm_dict
        """
        # Signals
        # Communication signals
        if comm_signal_list is None:
            self.message_signal, self.plot_signal, self.parameters_update_signal = None, None, None
        else:
            self.message_signal, self.plot_signal, self.parameters_update_signal = comm_signal_list
        # Pre job name
        pre_job_name = client_job_name
        # Datetime for 1-1 association
        date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        # Client job name to send to the Server
        self.client_job_name = date_time + "_" + pre_job_name
        ###
        # Logging, Results ...
        ###
        self.res_path = os.path.join(os.getcwd(), "dCRAB", self.client_job_name)
        date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        self.job_name = date_time + "_" + "test_job_optimal_code"
        ## Check folder of the control algorithm
        # TODO Change name here
        opti_method = "dCRAB"
        if not os.path.isdir(os.path.join(os.getcwd(), opti_method)):
            os.makedirs(os.path.join(os.getcwd(), opti_method))
        # Create the folder for logging and results
        os.makedirs(self.res_path)

        ## Logging object
        # Create logging object
        self.logger = self._create_logger()

        ## FoM object
        self.rf_obj = RF(fom_dict)

        ## Dumping data object
        self.dr_obj = DR(self.client_job_name)

        ## Initialization of the handle exit object
        self.he_obj = handle_exit_obj

        ## Controls dict
        self.controls_dict = {}


    def _create_logger(self):
        """Logger creation for console, log file, and debug log file"""
        log_format = '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s'
        date_format = '%m/%d/%Y %I:%M:%S'
        print_format = '%(levelname)-8s %(name)-12s %(message)s'
        log_filename = os.path.join(self.res_path, "logging.log")
        log_debug_filename=os.path.join(self.res_path, "logging_debug.log")

        logger = logging.getLogger("oc_logger")
        # Default level for logger
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler=logging.StreamHandler(sys.stdout)
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
            debug_file_handler=logging.FileHandler(log_debug_filename)
            debug_file_handler.setLevel(logging.DEBUG)
            debug_file_handler.setFormatter(logging.Formatter(log_format, date_format))

            logger.addHandler(debug_file_handler)

        return logger

    def get_client_job_name(self):
        """Get client job name"""
        return self.client_job_name

    def send_controls(self, controls):
        """Set the controls for FoM calculation and send to the client"""
        self.controls_dict = controls
        if self.parameters_update_signal is not None:
            self.parameters_update_signal.emit(controls["paras"])

    def get_data(self):
        """Calculate the figure of merit"""
        fom_dict = self.rf_obj.get_FoM(**self.controls_dict)
        return {"fom_values": fom_dict}

    def send_fom_response(self, response_for_client):
        """Emit signal to the Client Interface"""
        #
        iteration_number=response_for_client["iteration_number"]
        fom = response_for_client["FoM"]
        #
        self.logger.info("Iteration number: {0}, FoM: {1}".format(iteration_number, fom))
        if self.plot_signal is not None:
            self.plot_signal.emit(iteration_number, fom)

    def end_communication(self, results_dict):
        """Report the final results"""
        # Print final results data
        for el in results_dict:
            self.logger.info("{0} : {1}".format(el, results_dict[el]))

    def assign_job(self):
        """Do nothing"""
        pass

    def send_communication_data(self):
        """Do nothing"""
        pass

    def update_init_msg_server(self, upd_name=None):
        """Do nothing"""
        pass

    def check_msg_client(self):
        """Do nothing"""
        pass

    def update_msg_server(self):
        """Do nothing"""
        pass
