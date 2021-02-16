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

import sys

from ClientInterface.logic.utilities.readjson import readjson
from ClientInterface.logic.AnalysisSteeringLogic import AnalysisSteering as AS
from ClientInterface.logic.HandleExit import HandleExit as HE

def main(argv):
    """ Main for the NoGUI QuOCS """
    ## Input dictionaries
    # TODO Handle possible input errors
    err_stat_opti_dict, opti_dict = readjson(argv[1])
    err_stat_comm_fom, comm_fom_dict = readjson(argv[2])
    if (err_stat_comm_fom + err_stat_opti_dict) > 0: return
    #
    print("Welcome to QuOCS")
    # handle exit object
    handle_exit_obj = HE()
    ## Main object depends on the communication type: AllInOne vs Local/Remote
    if comm_fom_dict["comm_dict"]["type"] == "AllInOneCommunication":
        name_alg=opti_dict["opti_dict"]["opti_name"]
        # TODO Dynamical import instead
        from OptimizationCode.Optimal_lib.OptimalAlgorithms.DirectSearchAlgorithm import DirectSearchAlgorithm as DSA
        from OptimizationCode.Optimal_lib.OptimalAlgorithms.dCRABAlgorithm import dCrabAlgorithm as DC
        optimal_algs_list={"direct_search_1": DSA, "dCRAB": DC}
        as_obj=optimal_algs_list[name_alg](handle_exit_obj=handle_exit_obj, opti_dict=opti_dict["opti_dict"], comm_fom_dict=comm_fom_dict)
    else:
        as_obj=AS(handle_exit_obj, opti_dict=opti_dict, comm_fom_dict=comm_fom_dict)
    ## Main operations
    try:
        as_obj.begin()
        as_obj.run()
    except Exception as ex:
        print("Unhandled exception in QuOCS. Error: {0}".format(ex.args))
    finally:
        as_obj.end()
        print("Bye Bye QuOCS")

if __name__=="__main__":
    #TODO Handle GUI / No GUI
    main(sys.argv)