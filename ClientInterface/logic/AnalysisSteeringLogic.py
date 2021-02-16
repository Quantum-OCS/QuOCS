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

from ClientInterface.logic.WorkflowLogic import Workflow
from ClientInterface.logic.utilities.readjson import readjson


class AnalysisSteering:
    """ Main class fro the client interface with begin, run, end operations"""
    is_running = True

    def __init__(self, handle_exit_obj, opti_dict=None, comm_fom_dict=None, comm_signals_list=None):
        """

        @param handle_exit_obj: Object to handle the exit
        @param opti_dict: Dictionary with the optimization settings
        @param comm_fom_dict: Dictionary with Communication and FoM evaluation settings
        @param comm_signals_list: List with Gui signals
        """
        # Config and Communication dictionaries
        if comm_signals_list is None:
            comm_signals_list = [None, None, None]

        # TODO Adjust this part with available test json files, or put it in the constructor parameters
        if opti_dict is None:
            err_stat, opti_dict = readjson("Config3_Test.json")[1]
        if comm_fom_dict is None:
            err_stat, comm_fom_dict = readjson("Communication_Test_Local3.json")[1]

        # Get the job name set by the Client
        client_job_name = opti_dict["opti_dict"]["comm_dict"]["client_job_name"]
        # Create the workflow object. It handles the iterative optimization process
        self.wf = Workflow(handle_exit_obj, client_job_name, comm_fom_dict, comm_signals_list)
        # Save into a variable the optimal dictionary
        self.cfg_dict = opti_dict

    def begin(self):
        """ Start the communication process """
        # Initialize the communication using the optimal dictionary
        self.is_running = self.wf.initialize_communication(self.cfg_dict)

    def run(self):
        """ Run the main iterative process """
        # Get steps from workflow object
        steps_list = self.wf.steps
        # Check if the initialization was successful
        is_running = self.is_running
        # Loop until
        while is_running:
            for step in steps_list:
                if is_running:
                    is_running = step()

    def end(self):
        """ Finalize the optimization task """
        self.wf.end_communication()
