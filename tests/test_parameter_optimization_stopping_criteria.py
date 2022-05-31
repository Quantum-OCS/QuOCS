from quocslib.utils.AbstractFoM import AbstractFoM
from quocslib.utils.inputoutput import readjson
from quocslib.Optimizer import Optimizer
from scipy.optimize import rosen
import numpy as np
import os


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

    # Try to access to the FoM list
    opt_alg_obj = optimization_obj.get_optimization_algorithm()
    best_parameters = opt_alg_obj.get_best_controls()["parameters"]
    FoM_list = opt_alg_obj.FoM_list
    iteration_number_list = opt_alg_obj.iteration_number_list


def test_parameter_optimization():
    # Create the optimization dictionary
    optimization_dictionary = {
        "optimization_client_name": "RosenbrockOptimization",
        "algorithm_settings": {
            "algorithm_name": "DirectSearch"
        }
    }
    # TODO Check the other stopping criteria as well
    dsm_settings = {
        "general_settings": {
            "dsm_algorithm_name": "NelderMead",
            "is_adaptive": False
        },
        "stopping_criteria": {
            "xatol": 1e-5,
            "frtol": 1e-5
        }
    }
    optimization_dictionary["algorithm_settings"]["dsm_settings"] = dsm_settings
    # Optimize 10 parameters
    total_number_of_parameters = 10
    parameters = []
    for index in range(total_number_of_parameters):
        parameters.append({
            "parameter_name": "Parameter{0}".format(index),
            "lower_limit": -2.0,
            "upper_limit": 2.0,
            "initial_value": 0.4,
            "amplitude_variation": 0.5
        })
    optimization_dictionary["pulses"] = []
    optimization_dictionary["parameters"] = parameters
    optimization_dictionary["times"] = []

    main(optimization_dictionary)
