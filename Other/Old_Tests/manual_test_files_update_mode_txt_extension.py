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

from quocslib.handleexit.HandleExit import HandleExit
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.utils.inputoutput import readjson
from quocslib.utils.FilesUpdateFoM import FilesUpdateFoM
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib.utils.BestDump import BestDump


def main(optimization_dictionary: dict):
    # Initialize the communication object
    interface_job_name = optimization_dictionary["optimization_client_name"]
    FoM_obj = FilesUpdateFoM(controls_folder=".", FoM_folder=".", file_extension="txt")
    communication_obj = AllInOneCommunication(interface_job_name=interface_job_name,
                                              FoM_obj=FoM_obj,
                                              handle_exit_obj=HandleExit(),
                                              dump_attribute=BestDump)
    optimizer_attribute = dynamic_import(attribute=optimization_dictionary.setdefault("opti_algorithm_attribute", None),
                                         module_name=optimization_dictionary.setdefault("opti_algorithm_module", None),
                                         class_name=optimization_dictionary.setdefault("opti_algorithm_class", None))
    optimizer_obj = optimizer_attribute(optimization_dict=optimization_dictionary, communication_obj=communication_obj)
    print("The optimizer was initialized successfully")
    optimizer_obj.begin()
    print("The optimizer begin successfully")
    optimizer_obj.run()
    print("The optimizer run successfully")
    optimizer_obj.end()
    print("The optimizer end successfully")


if __name__ == '__main__':
    import sys
    args_number = len(sys.argv)
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        err_stat, user_data = readjson(filename)
        if err_stat == 0:
            main(user_data)
        else:
            print("File {0} does not exist".format(filename))
    else:
        print("{0} are {1} arguments".format(sys.argv, args_number))
        print("Only 2 arguments are allowed")
