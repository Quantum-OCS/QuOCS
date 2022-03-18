import os, sys

from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
from quocslib.handleexit.HandleExit import HandleExit
from quocslib.utils.dynamicimport import dynamic_import
from quocslib.utils.inputoutput import readjson
from quocslib.communication.AllInOneCommunication import AllInOneCommunication
from quocslib.optimalcontrolproblems.RosenbrockProblem import Rosenbrock
from quocslib.utils.BestDump import BestDump
from quocslib.utils.AbstractFom import AbstractFom
from scipy.optimize import rosen
import numpy as np
import pytest


class RosenFoM(AbstractFom):

    def __init__(self, args_dict:dict = None):
        if args_dict is None:
            args_dict = {}

        self.fom_list = []
        self.param_list = []
        self.save_path = ""

    def __del__(self):
        np.savetxt(os.path.join(self.save_path, 'fom.txt'), self.fom_list)
        np.savetxt(os.path.join(self.save_path, 'params.txt'), self.param_list)


    def set_save_path(self, save_path: str = ""):
        self.save_path = save_path


    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:

        fom = rosen(np.asarray(parameters))

        self.fom_list.append(fom)
        self.param_list.append(parameters)
        
        return {"FoM": fom}


def main(optimization_dictionary: dict):

    # Initialize the communication object
    interface_job_name = optimization_dictionary["optimization_client_name"]

    FoM_object = RosenFoM()

    communication_obj = AllInOneCommunication(interface_job_name=interface_job_name,
                                              fom_obj=FoM_object, handle_exit_obj=HandleExit(),
                                              dump_attribute=BestDump)

    FoM_object.set_save_path(communication_obj.results_path)

    # ----------------------------------------------------------------------------
    
    optimizer_attribute = dynamic_import(
        attribute=optimization_dictionary.setdefault("opti_algorithm_attribute", None),
        module_name=optimization_dictionary.setdefault("opti_algorithm_module", None),
        class_name=optimization_dictionary.setdefault("opti_algorithm_class", None))
    
    optimizer_obj = optimizer_attribute(optimization_dict=optimization_dictionary,
                                        communication_obj=communication_obj)

    print("The optimizer was initialized successfully")
    optimizer_obj.begin()
    print("The optimizer started successfully")
    optimizer_obj.run()
    print("The optimizer ran successfully")
    optimizer_obj.end()
    print("The optimizer finished successfully")


def test_parameter_optimization():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "opt_Rosen_NM.json"))[1]

