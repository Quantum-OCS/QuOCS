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

from ClientInterface.logic.utilities.writejsonfile import writejsonfile
from ClientInterface.logic.utilities.readjson import readjson


class ServerCommunication:
    """

    """
    ## Workflow number
    server_job_name = ""
    server_number = 0

    client_job_name = ""
    client_number = 0

    ## Paths for communication
    path_opti_data = ""
    opti_data_file = ""
    path_communication_folder = ""


    def __init__(self, comm_dict):
        """

        Parameters
        ----------
        comm_dict
        """
        print("Communication activated.")
        # Folder for incoming jobs
        self.incoming_communication_folder=os.path.join(os.getcwd(), "incoming_jobs")
        # Main folder for the communication
        self.main_communication_folder = os.path.join(os.getcwd(), "communication_jobs")
        # Read the client job name
        self.client_job_name = comm_dict["client_job_name"]

    def get_client_job_name(self):
        return self.client_job_name

    def assign_job(self):
        """
        The Server assigns a job number and a job folder for the communication
        Returns
        -------

        """
        print("I am going to assign a job")
        # Datetime for 1-1 association
        date_time = str(time.strftime("%Y%m%d_%H%M%S"))
        # Server job name
        self.server_job_name = date_time + self.client_job_name
        # Name opti data file
        self.opti_data_file = self.client_job_name + ".json"
        # Communication folder
        self.path_communication_folder = os.path.join(self.main_communication_folder, self.server_job_name)
        # Full path opti data json file
        self.path_opti_data = os.path.join(self.path_communication_folder, self.opti_data_file)
        # Create communication folder
        os.mkdir(self.path_communication_folder)
        # Update Server number
        self.server_number = 0

    def _get_communication_data(self):
        """
        Return the communication data for the Client-Server transmission
        Returns
        -------

        """
        communication_dict = {"server_job_name": self.server_job_name, "server_number": self.server_number,
                              "client_job_name": self.client_job_name, "client_number": self.client_number,
                              "path_communication_folder": self.path_communication_folder,
                              "path_opti_data": self.path_opti_data,
                              "opti_data_file": self.opti_data_file}

        return communication_dict

    def send_communication_data(self):
        """
        Send the initial communication dictionary to the client to enable the transmission
        Returns
        -------

        """
        # Initial dictionary
        print("Sending the communication dictionary to the Client to start the transmission")
        send_dict = {"communication": self._get_communication_data() }
        #
        writejsonfile(os.path.join(self.main_communication_folder, self.client_job_name + ".json"), send_dict)

    # TODO Check if this module is really compulsory
    def update_communication_folder(self):
        """
        Update the communication folder with the folder assigned by the Server for the communication
        Returns
        -------

        """
        self.main_communication_folder = self.path_communication_folder

    def send_controls(self, pars_dict):
        """
        Send the parameters computed by the Server to the client
        Parameters
        ----------
        pars_dict

        Returns
        -------

        """
        # Update server number
        self.server_number = 1
        # Communication dictionary and controls dictionary
        data_dict = {"controls": pars_dict, "communication": self._get_communication_data()}
        #
        pathfile=os.path.join(self.path_communication_folder, self.opti_data_file)
        print("I am sending the parameters file to the communication folder: {0}".
              format(pathfile))
        writejsonfile(pathfile, data_dict)


    # Probably this is a Client Module
    def send_config(self, cfg_dict):
        """

        Parameters
        ----------
        cfg_dict

        Returns
        -------

        """
        pathfile = os.path.join(self.main_communication_folder, self.opti_data_file)
        print("I am sending the cfg file to the communication folder: {0}"
              .format(pathfile))
        writejsonfile(pathfile, cfg_dict)

        pass

    def update_init_msg_server(self, upd_name):
        """
        Update client message for initial communication
        Returns
        -------

        """
        pathfile = os.path.join(self.main_communication_folder, upd_name + "upd_server.txt")
        print("Update the initial server message in {0}".format(pathfile))
        open(pathfile, "w").close()

    def update_msg_server(self):
        """
        Update client message
        Returns
        -------

        """
        pathfile = os.path.join(self.path_communication_folder, "upd_server.txt")
        print("Update the server message in {0}".format(pathfile))
        open(pathfile, "w").close()

    def check_msg_client(self):
        """

        Returns
        -------

        """
        # TODO Add limit wait time for the communication
        update_message = os.path.join(self.path_communication_folder, "upd_client.txt")
        print("Check the client message update in {0}".format(update_message))
        while not os.path.isfile(update_message):
            # Do nothing, just wait
            time.sleep(0.01)
        os.remove(update_message)

    def get_data(self):
        """

        Returns
        -------

        """
        err_stat, opti_data = readjson(os.path.join(self.path_communication_folder, self.opti_data_file))
        print(opti_data)
        return opti_data

    def send_fom_response(self, response_dict):
        """
        The Server sends the FoM response to the Client
        Parameters
        ----------
        opti_dict

        Returns
        -------

        """
        # Update server number
        self.server_number = 2
        pathfile = os.path.join(self.path_communication_folder, self.opti_data_file)
        comm_dict = self._get_communication_data()
        data_dict = {"communication": comm_dict, "response": response_dict}
        print("Send fom response in {0}".format(pathfile))
        writejsonfile(pathfile, data_dict)

    def end_communication(self, final_dict):
        """
        End the communication with the client
        Returns
        -------

        """
        self.server_number = 4
        print("Send the final signal to the client and end the communication")
        pathfile = os.path.join(self.path_communication_folder, self.opti_data_file)
        data_dict = {"communication": self._get_communication_data(), "final_results": final_dict}
        writejsonfile(pathfile, data_dict)

    def get_results(self):
        pass

