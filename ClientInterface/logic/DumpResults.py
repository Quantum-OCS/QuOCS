"""
Class for dumping data into a file
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
import datetime

# TODO Add print set of best controls at each iteration in a json file
# TODO Create different directory based on the method (dCRAB, etc...)


class DumpResults:
    """ Dump optimization results into a file"""
    def __init__(self, client_job_name):
        """

        @param client_job_name: client interface job name
        """
        opti_method = "dCRAB"
        # Results directory path
        self.res_path = os.path.join(os.getcwd(), opti_method, client_job_name)
        # Results full data path
        self.full_data_path = os.path.join(self.res_path, "Results_QuOCS_" + client_job_name + "_FullData.txt")
        # Create the file to dump into
        open(self.full_data_path, 'w').close()

    # TODO Currently it is not called by the Communication class. Put a called there
    def print_data(self, data_dict):
        """
        Print relevant information in the results file
        @param data_dict: Dictionary with the controls
        """
        full_data_file = open(self.full_data_path, 'a')
        full_data_file.write("Time : {0} \n".format(datetime.datetime.now()))
        for element in data_dict:
            full_data_file.write("{0} : {1} \n".format(element, data_dict[element]))
        full_data_file.write("End \n")
        full_data_file.close()
