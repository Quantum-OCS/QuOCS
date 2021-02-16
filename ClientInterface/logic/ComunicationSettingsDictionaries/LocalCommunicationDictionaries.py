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

class LocalCommunicationDictionaries:

    def __init__(self, std_dictionary=None):
        # TODO Insert standard values ? , then stretch the dictionary into a line
        if std_dictionary is None:
            local_communication_dictionary = {}
            local_communication_dictionary["shared_folder"] = ""
            local_communication_dictionary["results_folder"] = ""
        else:
            local_communication_dictionary = std_dictionary

        self.local_communication_dictionary = local_communication_dictionary

    def get_shared_folder(self):
        return self.local_communication_dictionary["shared_folder"]

    def get_results_folder(self):
        return self.local_communication_dictionary["results_folder"]

    def set_shared_folder(self, shared_folder):
        self.local_communication_dictionary["shared_folder"] = shared_folder

    def set_results_folder(self, results_folder):
        self.local_communication_dictionary["results_folder"] = results_folder

    def get_dictionary(self):
        return self.local_communication_dictionary
