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

import paramiko
import logging
import os
import time

from ClientInterface.logic.communication.Communication import Communication


class RemoteCommunication(Communication):

    ssh_client = None
    sftp_client = None

    def __init__(self, handle_exit_obj, client_job_name, comm_fom_dict, comm_signals_list):
        super().__init__(handle_exit_obj, client_job_name, comm_fom_dict["fom_dict"], *comm_signals_list)


        ## Paramiko logging
        logging.getLogger("paramiko").setLevel(logging.INFO)
        ## Communication folder for the server, suppose it is a Linux server
        self.main_communication_folder="/home/mrossign/MEGA/PhDUlmPadua/Padua_Projects/dCRABClient/communication_jobs"
        self.initial_communication_folder="/home/mrossign/MEGA/PhDUlmPadua/Padua_Projects/dCRABClient/incoming_jobs/"

        ## Activation of the client server communication
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        test_connection = 1

        #
        server_name = ""
        username = ""
        password = ""
        while not self.is_connected and test_connection<10:
            try:
                ssh_client.connect(server_name, username=username, password=password, look_for_keys=False,
                                   allow_agent=False)
                self.ssh_client = ssh_client
                self.is_connected = True
                self.logger.info("Remote connection activated")
            except paramiko.AuthenticationException:
                self.logger.critical("Authentication Failed. Check your server, username, and password")
            except Exception as ex:
                self.logger.warning("Unhandled exception in {0}. Connection failed. Wait 2 seconds and retry. Try {0}"
                                    .format(self.__module__ + "constructor", test_connection))
                time.sleep(2)
                test_connection += 1

        ## Set the sftp connection
        if not self.is_connected:
            print("Connection do not work. Check your Internet connection or retry later")
            self.set_client_number(-1, error=True)
        else:
            self._sftp_connection()

        # Communication folder in the Server
        self.share_folder = os.path.join(os.getcwd(),"incoming_jobs")

    def _sftp_connection(self):
        """
        Open the sftp channel
        Returns
        -------

        """
        # Create channel for sftp connection
        self.sftp_client = self.ssh_client.open_sftp()
        # Set timeout
        self._set_ftp_channel_timeout()

    def _set_ftp_channel_timeout(self, channel_timeout=20.0):
        """
        Set the channel timeout

        Parameters
        ----------
        channel_timeout

        Returns
        -------

        """
        self.sftp_client.get_channel().settimeout(channel_timeout)

    def move_file(self, origin_path, destination_path, confirm=True):
        """
        Put the file in the remote server
        Returns
        -------

        """
        try:
            self.logger.debug("Remote Communication: Move file from {0} to {1}".format(origin_path, destination_path))
            if os.path.isfile(origin_path):
                self.sftp_client.put(origin_path, destination_path, confirm=confirm)
            else:
                self.logger.critical("Someone wants to joke. {0} is not a file".format(origin_path))
        except Exception as ex:
            self.logger.critical("An unhandled exception happened in move_file: {0}".format(ex.args))
            self.set_client_number(-1, error=True)

    def get_file(self, remote_path, local_path):
        """
        Get the file from the remote server
        Parameters
        ----------
        local_path
        remote_path

        Returns
        -------

        """
        try:
            self.sftp_client.get(remote_path, local_path)
        except Exception as ex:
            self.logger.critical("An unhandled exception happened in get_file: {0}".format(ex.args))
            self.set_client_number(-1, error=True)

    def check_file(self, remote_path):
        """

        Parameters
        ----------
        remote_path

        Returns
        -------

        """
        try:
            self.sftp_client.stat(remote_path)
            return True
        except IOError:
            return False
        except Exception as ex:
            self.logger.critical("An unhandled exception happened in exist_file: {0}".format(ex.args))
            self.set_client_number(-1, error=True)

    def remove_file(self, remote_path):
        """

        Parameters
        ----------
        remote_path

        Returns
        -------

        """
        try:
            self.logger.debug("Remote communication: Remove file {0}".format(remote_path))
            self.sftp_client.remove(remote_path)
        except Exception as ex:
            self.logger.critical("An unhandled exception happened in remove_file: {0}".format(ex.args))
            self.set_client_number(-1, error=True)

    def end(self, opti_data):
        super().end(opti_data)
        if self.is_connected:
            self.sftp_client.close()
            self.logger.debug("Close the sftp connection")
            self.ssh_client.close()
            self.logger.debug("Close the ssh connection")
            self.logger.info("Remote connection deactivated")
