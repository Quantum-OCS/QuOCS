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

import importlib
import importlib.util
from scipy import optimize
import numpy as np
#  TODO implement the file update communication part


class ReceiveFom:
    """ Class for the calculation of the figure of merit"""

    def __init__(self, fom_dict):
        """
        Initialize the object for the figure of merit calculation
        @param fom_dict:
        """
        if fom_dict["ProgramType"] == "TestClass":
            # Get the name of the module with relative imports
            python_module = fom_dict["PythonModule"]
            # Get the name of the class to ve imported
            python_class = fom_dict["PythonClass"]
            # Further args
            further_args = fom_dict.setdefault("FurtherArgs", None)
            try:
                # Get FoM class
                fom_class = getattr(importlib.import_module(python_module), python_class)
            except ImportError:
                print("Import Error")
                return
            # Get FoM object
            self.fom_obj = fom_class(args_dict=further_args)
            self.program_type = "PythonClass"
        else:
            self.program_type = None

    def get_FoM(self, pulses, paras, timegrids):
        """
        Calculate the figure of merit given the controls in input
        @param pulses: list of list of pulses
        @param paras: list of parameters
        @param timegrids: list of list of timegrids
        @return: dictionary containing the relevant quantities for the optimization
        """
        try:
            if self.program_type is None:
                fom = self.fom_eval(pulses, paras, timegrids)
            else:
                fom = self.fom_obj.fom_eval(pulses, paras, timegrids)
        except Exception as ex:
            fom = None
            error_message = "Unhandled exception in the FoM evaluation: {0}".format(ex)
            return {"FoM": fom, "ErrorMessage": error_message, "ErrorCode": -6}
        return {"FoM": fom}

    # TODO Add update files figure of merit method

    # Just a test module for FoM evaluations
    @staticmethod
    def fom_eval(dCRABPulses, dCRABParas, timegrids):
        xx = np.asarray(dCRABParas)
        fom = optimize.rosen(xx)
        return fom
