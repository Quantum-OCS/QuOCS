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

# General imports
import os
import threading
# Utility imports
from ClientInterface.logic.utilities.readjson import readjson
# Server lib imports
from OptimizationCode.Optimal_lib.OptimalAlgorithms.DirectSearchAlgorithm import DirectSearchAlgorithm as DSA
from OptimizationCode.Optimal_lib.OptimalAlgorithms.dCRABAlgorithm import dCrabAlgorithm as DC


class ServerHandler:
    """

    """
    is_running = True
    is_busy = False
    def __init__(self):
        """

        """
        self.total_allowed_jobs = 10
        self.thread_list = []
        self.curr_jobs_list = []
        self.curr_jobs = 0
        # Incoming jobs path
        self.incoming_jobs_path=os.path.join(os.getcwd(), "incoming_jobs")
        pass

    def get_incoming_jobs_list(self):
        """

        Returns
        -------

        """
        # Job list
        job_list=[]
        # Loop over all files
        with os.scandir(self.incoming_jobs_path) as ij:
            for entry in ij:
                # Read the json config file
                err_stat, user_data=readjson(entry)
                if err_stat == 0:
                    job_list.append(user_data)
                # Remove the json config file
                os.remove(entry)
        return job_list

    def check_server_status(self):
        """

        Returns
        -------

        """
        # Remaining jobs
        available_jobs = self.total_allowed_jobs - self.curr_jobs
        # Set busy or available server
        if available_jobs > 0:
            self.is_busy = False
        else:
            self.is_busy = True

    def run_job(self, job):
        """

        Parameters
        ----------
        job

        Returns
        -------

        """
        # Run or decline new jobs
        #opti_dict= job["opti_dict"]
        opti_dict = job
        thread_name = "Test1"
        # Define the threading
        t=threading.Thread(name=thread_name, target=opti_worker, kwargs=opti_dict)

        # Start the thread
        t.start()
        # Join it only for debug
        t.join()


    def update_job_number(self):
        """

        Returns
        -------

        """
        # Update the current job number
        print("Check alive threads ...")


def opti_worker(opti_dict):
    """
    Run the optimization job
    Parameters
    ----------
    opti_dict (dict) : Contains all the optimization dictionaries need for the optimal control algorithm
    options

    Returns
    -------

    """
    ## TODO Split the dictionary into 3 variables
    name_alg = opti_dict["opti_name"]
    optimal_algs_list = {"direct_search_1": DSA, "dCRAB": DC}
    ## Launch the optimizer
    opt_obj = optimal_algs_list[name_alg](opti_dict)
    try:
        if opt_obj.init_status:
            opt_obj.begin()
            opt_obj.run()
    except Exception as ex:
        print("Unhandled exception in the Server Code: {0}".format(ex.args))
    finally:
        opt_obj.end()


