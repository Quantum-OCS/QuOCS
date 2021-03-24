import os

from OptimalControlProblems.OneQubitClass import OneQubit
from quocs_optlib.handleexit.AbstractHandleExit import AbstractHandleExit
from quocs_optlib.tools.dynamicimport import dynamic_import
from quocs_optlib.tools.inputoutput import readjson
from quocs_optlib.communication.AllInOneCommunication import AllInOneCommunication


class HandleExit(AbstractHandleExit):
    pass


def main(optimization_dictionary: dict):
    args_dict = {"initial_state": "[1.0 , 0.0]", "target_state": "[1.0/np.sqrt(2), -1j/np.sqrt(2)]"}
    # Initialize the communication object
    interface_job_name = optimization_dictionary["optimization_client_name"]
    communication_obj = AllInOneCommunication(interface_job_name=interface_job_name,
                                              fom_obj=OneQubit(args_dict=args_dict), handle_exit_obj=HandleExit())
    optimizer_attribute = dynamic_import(
        attribute=optimization_dictionary.setdefault("opti_algorithm_attribute", None),
        module_name=optimization_dictionary.setdefault("opti_algorithm_module", None),
        class_name=optimization_dictionary.setdefault("opti_algorithm_class", None))
    optimizer_obj = optimizer_attribute(optimization_dict=optimization_dictionary,
                                        communication_obj=communication_obj)
    print("The optimizer was initialized successfully")
    optimizer_obj.begin()
    print("The optimizer begin successfully")
    optimizer_obj.run()
    print("The optimizer run successfully")
    optimizer_obj.end()
    print("The optimizer end successfully")


if __name__ == '__main__':
    main(readjson(os.path.join(os.getcwd(), "algorithm_dictionary_v4.json"))[1])
