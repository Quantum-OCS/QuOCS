import os, sys

from quocslib.optimalcontrolproblems.OneQubitProblem import OneQubit
from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
from scipy.optimize import rosen
import numpy as np
import pytest


class RosenFoM(AbstractFoM):

    def __init__(self, args_dict:dict = None):
        if args_dict is None:
            args_dict = {}

        self.FoM_list = []
        self.param_list = []
        self.save_path = ""

    # def __del__(self):
    #     np.savetxt(os.path.join(self.save_path, 'FoM.txt'), self.FoM_list)
    #     np.savetxt(os.path.join(self.save_path, 'params.txt'), self.param_list)

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




folder = os.path.dirname(os.path.realpath(__file__))
optimization_dictionary = readjson(os.path.join(folder, "opt_Rosen_NM.json"))[1]

FoM_object = RosenFoM()

optimization_obj = Optimizer(optimization_dictionary, FoM_object)

FoM_object.set_save_path(optimization_obj.results_path)

optimization_obj.execute()

FoM_object.save_FoM()



