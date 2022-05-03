# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright 2021-  QuOCS Team
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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os

from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib.handleexit.HandleExit import HandleExit
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.utils.inputoutput import readjson
from quocslib.optimalcontrolproblems.RosenbrockProblem import Rosenbrock
from quocslib.utils.BestDump import BestDump


def main(optimization_dictionary: dict):
    args_dict = {}
    # Initialize the communication object
    interface_job_name = optimization_dictionary["optimization_client_name"]
    communication_obj = AllInOneCommunication(interface_job_name=interface_job_name,
                                              FoM_obj=Rosenbrock(args_dict),
                                              handle_exit_obj=HandleExit(),
                                              dump_attribute=BestDump)
    # Get the optimizer attribute
    optimizer_attribute = dynamic_import(attribute=optimization_dictionary.setdefault("opti_algorithm_attribute", None),
                                         module_name=optimization_dictionary.setdefault("opti_algorithm_module", None),
                                         class_name=optimization_dictionary.setdefault("opti_algorithm_class", None))
    # Create the optimizer object
    optimizer_obj = optimizer_attribute(optimization_dict=optimization_dictionary, communication_obj=communication_obj)
    print("The optimizer was initialized successfully")
    optimizer_obj.begin()
    print("The optimizer begin successfully")
    optimizer_obj.run()
    print("The optimizer run successfully")
    optimizer_obj.end()
    print("The optimizer end successfully")


if __name__ == '__main__':
    main(readjson(os.path.join(os.getcwd(), "..", "algorithm_dictionary_PureParameter_CMAES_Rosenbrock.json"))[1])
