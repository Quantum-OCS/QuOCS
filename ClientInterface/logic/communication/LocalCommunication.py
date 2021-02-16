"""
This is the class for the remote communication
"""
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

from ClientInterface.logic.communication.Communication import Communication


class LocalCommunication(Communication):
    """

    """

    def __init__(self, handle_exit_obj, client_job_name, comm_fom_dict, comm_signals_list):
        """

        Parameters
        ----------
        comm_dict
        """
        # Debug flag
        comm_dict = comm_fom_dict["comm_dict"]
        self.is_debug = comm_dict.setdefault("is_debug", False)
        super().__init__(handle_exit_obj, client_job_name, comm_fom_dict["fom_dict"], *comm_signals_list)
        # Initialize the folder for initial communication
        self.initial_communication_folder=comm_dict["communication_folder"]
        # This is a custom choice in the initial settings
        self.main_communication_folder = "communication_jobs"
        self.logger.info("Local communication activated")
        self.is_connected = True

    def move_file(self, origin_path, destination_path, confirm = True):
        os.rename(origin_path, destination_path)
        self.logger.debug("Local Communication: Move file from {0} to {1}".format(origin_path, destination_path))

    def get_file(self, origin, destination):
        self.move_file(origin, destination)

    def check_file(self, file):
        self.logger.debug("Local Communication: Check if {0} exists".format(file))
        return os.path.isfile(file)

    def remove_file(self, file):
        self.logger.debug("Local Communication: Remove {0}".format(file))
        os.remove(file)
