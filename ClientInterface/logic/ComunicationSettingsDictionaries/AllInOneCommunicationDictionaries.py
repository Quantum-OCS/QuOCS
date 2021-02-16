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

class AllInOneCommunicationDictionaries:

    def __init__(self, std_dictionary=None):
        # TODO Insert standard values ? , then stretch the dictionary into a line
        if std_dictionary is None:
            all_in_one_communication_dictionary = {}
            all_in_one_communication_dictionary["results_folder"] = ""
        else:
            all_in_one_communication_dictionary = std_dictionary

        self.all_in_one_communication_dictionary = all_in_one_communication_dictionary

    def get_results_folder(self):
        return self.all_in_one_communication_dictionary["results_folder"]

    def get_dictionary(self):
        return self.all_in_one_communication_dictionary
