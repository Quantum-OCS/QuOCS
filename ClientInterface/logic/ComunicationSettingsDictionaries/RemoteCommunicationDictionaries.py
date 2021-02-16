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

class RemoteCommunicationDictionaries:

    def __init__(self, std_dictionary=None):
        # TODO Insert standard values ? , then stretch the dictionary into a line
        if std_dictionary is None:
            remote_communication_dictionary = {}
            remote_communication_dictionary["type"] = "RemoteCommunication"
            remote_communication_dictionary["credentials_file"] = ""
            remote_communication_dictionary["username"] = ""
            remote_communication_dictionary["password"] = ""
            remote_communication_dictionary["server"] = ""
            remote_communication_dictionary["username"] = ""
            remote_communication_dictionary["shared_folder"] = ""
            remote_communication_dictionary["results_folder"] = ""
        else:
            remote_communication_dictionary = std_dictionary

        self.remote_communication_dictionary = remote_communication_dictionary

    def get_credentials_file(self):
        return self.remote_communication_dictionary["credentials_file"]

    def get_shared_folder(self):
        return self.remote_communication_dictionary["shared_folder"]

    def get_results_folder(self):
        return self.remote_communication_dictionary["results_folder"]

    def set_credentials_file(self, credentials_file):
        self.remote_communication_dictionary["credentials_file"] = credentials_file

    def set_shared_folder(self, shared_folder):
        self.remote_communication_dictionary["shared_folder"] = shared_folder

    def set_results_folder(self, results_folder):
        self.remote_communication_dictionary["results_folder"] = results_folder

    def set_username(self, username):
        self.remote_communication_dictionary["username"]=username

    def set_password(self, password):
        self.remote_communication_dictionary["password"]=password

    def set_server(self, server):
        self.remote_communication_dictionary["server"] = server

    def get_dictionary(self):
        return self.remote_communication_dictionary

    def get_summary_list(self):
        summary_list = []
        summary_list.append("Remote Communication")
        for element in self.remote_communication_dictionary:
            if element == "password": pass
            else: summary_list.append(element+" : "+self.remote_communication_dictionary[element])
        return summary_list


