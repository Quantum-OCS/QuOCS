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

from OptimizationCode.communication.ServerCommunication import ServerCommunication


class ServerLocalCommunication(ServerCommunication):

    def __init__(self, comm_dict):
        super().__init__(comm_dict)
        print("Server Local communication activated")
        # Share folder
        # self.share_folder = os.path.join(os.getcwd(),"shared_folder")

    def send_cfg(self, cfg_dict):
        """

        Parameters
        ----------
        cfg_dict

        Returns
        -------

        """
        super().send_config(cfg_dict)
        print("Sent the communication file to the shared folder")
        pass

    # Set later as a static method
    def sftp_connection(self):
        """
        Open the sftp channel
        Returns
        -------

        """
        pass

    # Set later as a static method
    def set_ftp_channel_timeout(self, channel_timeout=20.0):
        """
        Set the channel timeout

        Parameters
        ----------
        channel_timeout

        Returns
        -------

        """
        pass

    # This is an important class module to use in the workflow. Different behaviour dependent on the connection
    def put_file(self, local_path, remote_path):
        """
        Put the file in the remote server
        Returns
        -------

        """
        pass

    def get_file(self, local_path, remote_path):
        """
        Get the file from the remote server
        Parameters
        ----------
        local_path
        remote_path

        Returns
        -------

        """
        pass

    def check_file(self):
        pass

    # Lists of static methods, i.e. for file transfers
