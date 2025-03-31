from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
from scipy.optimize import rosen
import numpy as np
import os
import pytest
import threading


class RosenFoM(AbstractFoM):
    def __init__(self, args_dict: dict = None):
        if args_dict is None:
            args_dict = {}

        self.FoM_list = []
        self.param_list = []
        self.save_path = ""

    def save_FoM(self):
        np.savetxt(os.path.join(self.save_path, 'FoM.txt'), self.FoM_list)
        np.savetxt(os.path.join(self.save_path, 'params.txt'), self.param_list)

    def set_save_path(self, save_path: str = ""):
        self.save_path = save_path

    def get_FoM(self, pulses: list = [], parameters: list = [], timegrids: list = []) -> dict:

        FoM = rosen(np.asarray(parameters))

        self.FoM_list.append(FoM)
        self.param_list.append(parameters)

        return {"FoM": FoM}


def main(optimization_dictionary: dict):

    FoM_object = RosenFoM()

    optimization_obj = Optimizer(optimization_dictionary, FoM_object)

    FoM_object.set_save_path(optimization_obj.results_path)

    optimization_obj.execute()

    FoM_object.save_FoM()


def test_parameter_optimization():
    # get the optimization settings from the json dictionary
    folder = os.path.dirname(os.path.realpath(__file__))
    optimization_dictionary = readjson(os.path.join(folder, "opt_Rosen_NM.json"))
    opt_dict_1 = optimization_dictionary.copy()
    opt_dict_2 = optimization_dictionary.copy()
    opt_dict_1["optimization_client_name"] = "Check_quick_parallel_job_1"
    opt_dict_2["optimization_client_name"] = "Check_quick_parallel_job_2"
    
    t1 = threading.Thread(target=main, args=(opt_dict_1, ))
    t2 = threading.Thread(target=main, args=(opt_dict_2, ))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
